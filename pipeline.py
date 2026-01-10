"""
pipeline.py - Main orchestrator for the protein binder design pipeline.

This module implements the complete DAG:
1. RFDiffusion (backbone generation)
2. ProteinMPNN (sequence design)
3. Boltz-2 (structure validation)
4. FoldSeek (decoy identification)
5. Chai-1 (cross-reactivity check)

Final output: Ranked list of binder candidates optimized for specificity and selectivity.
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Optional

import modal

from common import (
    DATA_PATH,
    WEIGHTS_PATH,
    BackboneDesign,
    BinderCandidate,
    Boltz2Config,
    Chai1Config,
    ClusterConfig,
    CrossReactivityResult,
    DecoyHit,
    FoldSeekConfig,
    GenerationMode,
    NoveltyConfig,
    PipelineConfig,
    PipelineResult,
    PipelineStage,
    ProteinMPNNConfig,
    RFDiffusionConfig,
    SequenceDesign,
    StructurePrediction,
    TargetProtein,
    ValidationResult,
    ValidationStatus,
    app,
    base_image,
    data_volume,
    generate_design_id,
    generate_protein_id,
    generate_ulid,
    weights_volume,
)
from generators import (
    generate_sequences_parallel,
    mock_proteinmpnn,
    mock_rfdiffusion,
    run_proteinmpnn,
    run_rfdiffusion,
)
from validators import (
    check_cross_reactivity_parallel,
    check_novelty,
    cluster_by_tm_score,
    download_decoy_structures,
    mock_boltz2,
    mock_chai1,
    mock_foldseek,
    run_boltz2,
    run_chai1_batch,
    run_foldseek,
    validate_sequences_parallel,
)

# Note: Local modules are added to all images in common.py via _add_local_modules()

# =============================================================================
# Cost Estimation
# =============================================================================

# Approximate costs per GPU-hour (USD)
GPU_COSTS = {
    "A10G": 0.60,
    "L4": 0.80,
    "A100": 2.50,
    "A100-80GB": 3.50,
}

# Estimated runtime per step (seconds)
STEP_RUNTIMES = {
    "rfdiffusion": 60,  # per backbone
    "proteinmpnn": 15,  # per backbone
    "boltz2": 120,  # per sequence
    "foldseek": 30,  # per target
    "chai1": 90,  # per binder-decoy pair
}


def estimate_cost(config: PipelineConfig) -> float:
    """
    Estimate the compute cost for a pipeline run.

    Args:
        config: Pipeline configuration

    Returns:
        Estimated cost in USD
    """
    num_backbones = config.rfdiffusion.num_designs
    num_sequences_per_backbone = config.proteinmpnn.num_sequences
    total_sequences = num_backbones * num_sequences_per_backbone
    num_decoys = config.foldseek.max_hits

    # Assume 50% pass Boltz-2 validation
    passing_sequences = int(total_sequences * 0.5)

    # RFDiffusion cost (A10G)
    rfdiffusion_hours = (STEP_RUNTIMES["rfdiffusion"] * num_backbones) / 3600
    rfdiffusion_cost = rfdiffusion_hours * GPU_COSTS["A10G"]

    # ProteinMPNN cost (L4)
    proteinmpnn_hours = (STEP_RUNTIMES["proteinmpnn"] * num_backbones) / 3600
    proteinmpnn_cost = proteinmpnn_hours * GPU_COSTS["L4"]

    # Boltz-2 cost (A100)
    boltz2_hours = (STEP_RUNTIMES["boltz2"] * total_sequences) / 3600
    boltz2_cost = boltz2_hours * GPU_COSTS["A100"]

    # FoldSeek cost (CPU only, negligible)
    foldseek_cost = 0.01

    # Chai-1 cost (A100)
    chai1_pairs = passing_sequences * num_decoys
    chai1_hours = (STEP_RUNTIMES["chai1"] * chai1_pairs) / 3600
    chai1_cost = chai1_hours * GPU_COSTS["A100"]

    total_cost = (
        rfdiffusion_cost
        + proteinmpnn_cost
        + boltz2_cost
        + foldseek_cost
        + chai1_cost
    )

    return total_cost


# =============================================================================
# Pipeline Orchestrator
# =============================================================================


@app.function(
    image=base_image,
    timeout=3600,  # 1 hour max
    volumes={DATA_PATH: data_volume},
    min_containers=1,  # Keep orchestrator warm for quick starts
)
def run_pipeline(config: PipelineConfig, use_mocks: bool = False) -> PipelineResult:
    """
    Execute the complete protein binder design pipeline.

    Pipeline DAG:
    1. RFDiffusion → Generate binder backbones
    2. ProteinMPNN → Design sequences for each backbone
    3. Boltz-2 → Predict and validate structures
    4. FoldSeek → Identify structural decoys
    5. Chai-1 → Check cross-reactivity with decoys
    6. Score and rank candidates

    Args:
        config: Complete pipeline configuration
        use_mocks: If True, use mock implementations (for testing)

    Returns:
        PipelineResult with ranked binder candidates
    """
    start_time = time.time()
    run_id = generate_ulid()
    
    # Organize output: data/<PDB_ID>/entity_<N>/<YYYYMMDD>_<mode>_<ulid>/
    pdb_id = config.target.pdb_id.upper()
    entity_id = config.target.entity_id
    date_prefix = datetime.now().strftime("%Y%m%d")
    mode_str = config.mode.value
    campaign_id = f"{date_prefix}_{mode_str}_{run_id}"
    
    # Directory structure
    pdb_root = f"{DATA_PATH}/{pdb_id}"
    entity_dir = f"{pdb_root}/entity_{entity_id}"
    campaign_dir = f"{entity_dir}/{campaign_id}"
    dirs = {
        "backbones": f"{campaign_dir}/01_backbones",
        "sequences": f"{campaign_dir}/02_sequences",
        "validation_boltz": f"{campaign_dir}/03_validation/boltz",
        "validation_chai": f"{campaign_dir}/03_validation/chai",
        "metrics": f"{campaign_dir}/99_metrics",
        "pdb_root": pdb_root,
        "entity": entity_dir,
        "best": f"{entity_dir}/best_candidates",
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # Pre-flight cost check
    estimated_cost = estimate_cost(config)
    if estimated_cost > config.max_compute_usd:
        print(
            f"WARNING: Estimated cost ${estimated_cost:.2f} exceeds budget "
            f"${config.max_compute_usd:.2f}. Reducing design count."
        )
        # Scale down to fit budget
        scale_factor = config.max_compute_usd / estimated_cost
        config.rfdiffusion.num_designs = max(
            1, int(config.rfdiffusion.num_designs * scale_factor)
        )
        config.proteinmpnn.num_sequences = max(
            1, int(config.proteinmpnn.num_sequences * scale_factor)
        )

    # Consolidated pipeline log
    t = config.target
    residues = ", ".join(str(r) for r in t.hotspot_residues)
    cfg = f"{config.rfdiffusion.num_designs} × {config.proteinmpnn.num_sequences}"
    print(
        f"{t.name} ({t.pdb_id}) · Entity {t.entity_id} · "
        f"Residues {residues} · {run_id} · ${config.max_compute_usd:.2f} · {cfg}"
    )

    validation_results: list[ValidationResult] = []
    all_candidates: list[BinderCandidate] = []

    # =========================================================================
    # Phase 1: Generation (Specificity)
    # =========================================================================

    # Step 1: RFDiffusion - Backbone Generation
    print("\n=== Phase 1: Backbone Generation (RFDiffusion) ===")
    backbones = _run_backbone_generation(
        config.target,
        config.rfdiffusion,
        dirs["backbones"],
        use_mocks,
    )
    print(f"Generated {len(backbones)} backbone designs")

    validation_results.append(
        ValidationResult(
            stage=PipelineStage.BACKBONE_GENERATION,
            status=ValidationStatus.PASSED if backbones else ValidationStatus.FAILED,
            candidate_id=run_id,
            metrics={"num_backbones": len(backbones)},
        )
    )

    if not backbones:
        return _create_empty_result(run_id, config, validation_results, start_time)

    # Step 2: ProteinMPNN - Sequence Design
    print("\n=== Phase 1: Sequence Design (ProteinMPNN) ===")
    sequences = _run_sequence_design(
        backbones,
        config.proteinmpnn,
        dirs["sequences"],
        use_mocks,
    )
    print(f"Designed {len(sequences)} sequences")

    validation_results.append(
        ValidationResult(
            stage=PipelineStage.SEQUENCE_DESIGN,
            status=ValidationStatus.PASSED if sequences else ValidationStatus.FAILED,
            candidate_id=run_id,
            metrics={"num_sequences": len(sequences)},
        )
    )

    if not sequences:
        return _create_empty_result(run_id, config, validation_results, start_time)

    # =========================================================================
    # Phase 2: Validation (Specificity)
    # =========================================================================

    # Step 3: Boltz-2 - Structure Prediction & Filtering
    print("\n=== Phase 2: Structure Validation (Boltz-2) ===")
    predictions = _run_structure_validation(
        sequences,
        config.target,
        config.boltz2,
        dirs["validation_boltz"],
        use_mocks,
    )
    print(f"Validated {len(predictions)} sequences (passed filters)")

    validation_results.append(
        ValidationResult(
            stage=PipelineStage.STRUCTURE_VALIDATION,
            status=ValidationStatus.PASSED if predictions else ValidationStatus.FAILED,
            candidate_id=run_id,
            metrics={
                "num_passed": len(predictions),
                "num_tested": len(sequences),
                "pass_rate": len(predictions) / len(sequences) if sequences else 0,
            },
        )
    )

    if not predictions:
        return _create_empty_result(run_id, config, validation_results, start_time)

    # Step 4: Clustering - Diversity via TM-score (AlphaProteo SI 2.2)
    if config.cluster.tm_threshold > 0:
        print("\n=== Phase 2b: Clustering (Diversity) ===")
        predictions = cluster_by_tm_score(
            predictions,
            tm_threshold=config.cluster.tm_threshold,
            select_best=config.cluster.select_best,
        )
        print(f"After clustering: {len(predictions)} representatives")

    # Step 5: Novelty Check - pyhmmer vs UniRef50 (AlphaProteo SI 2.2)
    if config.novelty.enabled:
        print("\n=== Phase 2c: Novelty Check (pyhmmer) ===")
        # Get sequences corresponding to predictions
        pred_seq_ids = {p.sequence_id for p in predictions}
        novel_sequences = check_novelty(
            [s for s in sequences if s.sequence_id in pred_seq_ids],
            max_evalue=config.novelty.max_evalue,
            database=config.novelty.database,
        )
        # Filter predictions to only novel sequences
        novel_seq_ids = {s.sequence_id for s in novel_sequences}
        predictions = [p for p in predictions if p.sequence_id in novel_seq_ids]
        print(f"After novelty check: {len(predictions)} novel designs")
    
    if not predictions:
        print("No predictions passed clustering/novelty filters")
        return _create_empty_result(run_id, config, validation_results, start_time)

    # =========================================================================
    # Phase 3: Negative Selection (Selectivity)
    # =========================================================================

    # Step 6: FoldSeek - Find Structural Decoys
    print("\n=== Phase 3: Decoy Identification (FoldSeek) ===")
    decoys = _run_decoy_search(
        config.target,
        config.foldseek,
        f"{campaign_dir}/_decoys",
        use_mocks,
    )
    print(f"Found {len(decoys)} structural decoys (potential off-targets)")

    validation_results.append(
        ValidationResult(
            stage=PipelineStage.DECOY_SEARCH,
            status=ValidationStatus.PASSED if decoys else ValidationStatus.SKIPPED,
            candidate_id=run_id,
            metrics={"num_decoys": len(decoys)},
        )
    )

    # Step 5: Chai-1 - Cross-Reactivity Check (with Positive Control)
    positive_control_results: dict[str, CrossReactivityResult] = {}
    cross_reactivity_results: dict[str, list[CrossReactivityResult]] = {}

    # Get sequences that passed validation
    validated_sequences = [
        seq for seq in sequences
        if any(p.sequence_id == seq.sequence_id for p in predictions)
    ]

    if validated_sequences:
        print("\n=== Phase 3: Cross-Reactivity Check (Chai-1) ===")
        
        # Run positive control + decoy check
        positive_control_results, cross_reactivity_results = _run_cross_reactivity_check(
            validated_sequences,
            decoys if decoys else [],
            config.chai1,
            dirs["validation_chai"],
            use_mocks,
            target=config.target,  # Include target for positive control
        )
        
        # Report positive control results
        num_binding = sum(1 for r in positive_control_results.values() if r.plddt_interface > 50)
        print(f"Positive control: {num_binding}/{len(validated_sequences)} binders bind target")
        
        if decoys:
            print(f"Decoy check: {len(cross_reactivity_results)} sequences checked against {len(decoys)} decoys")

        validation_results.append(
            ValidationResult(
                stage=PipelineStage.CROSS_REACTIVITY,
                status=ValidationStatus.PASSED,
                candidate_id=run_id,
                metrics={
                    "num_checked": len(cross_reactivity_results),
                    "positive_control_binding": num_binding,
                },
            )
        )

    # =========================================================================
    # Final Scoring and Ranking
    # =========================================================================

    print("\n=== Final Scoring and Ranking ===")

    # Build candidate objects
    for prediction in predictions:
        # Find corresponding sequence and backbone
        sequence = next(
            (s for s in sequences if s.sequence_id == prediction.sequence_id),
            None,
        )
        if sequence is None:
            continue

        backbone = next(
            (b for b in backbones if b.design_id == sequence.backbone_id),
            None,
        )
        if backbone is None:
            continue

        # Get cross-reactivity results for this sequence
        cr_results = cross_reactivity_results.get(sequence.sequence_id, [])

        # Compute specificity using PPI score: 0.8 * ipTM + 0.2 * pTM
        # Scale to 0-100 for compatibility with existing scoring
        ppi_score = prediction.ppi_score  # Range [0, 1]
        specificity_score = ppi_score * 100 if ppi_score > 0 else prediction.plddt_interface

        # Selectivity: penalize if binder binds any decoy (using decoy PPI scores)
        max_decoy_ppi = 0.0
        if cr_results:
            decoy_ppi_scores = [r.ppi_score for r in cr_results if r.ppi_score > 0]
            if decoy_ppi_scores:
                max_decoy_ppi = max(decoy_ppi_scores)
            else:
                # Fallback to affinity-based scoring
                max_decoy_ppi = max(abs(r.predicted_affinity) / 10 for r in cr_results)
        selectivity_score = 100.0 - max_decoy_ppi * 100  # Penalize high decoy PPI

        # Final score using S(x) = α * PPI_target - β * max(PPI_decoy)
        final_score = (
            config.scoring.alpha * specificity_score
            - config.scoring.beta * max_decoy_ppi * 100
        )

        # Generate unique protein ID: <pdb_id>_E<entity_id>_<mode>_<ulid>
        candidate_id = generate_protein_id(
            config.target.pdb_id, config.target.entity_id, config.mode
        )
        
        candidate = BinderCandidate(
            candidate_id=candidate_id,
            sequence=sequence.sequence,
            backbone_design=backbone,
            sequence_design=sequence,
            structure_prediction=prediction,
            decoy_results=cr_results,
            specificity_score=specificity_score,
            selectivity_score=selectivity_score,
            final_score=final_score,
        )

        all_candidates.append(candidate)

    # Sort by final score (highest first)
    all_candidates.sort(key=lambda c: c.final_score, reverse=True)

    # Select top candidates
    top_n = min(10, len(all_candidates))
    top_candidates = all_candidates[:top_n]

    print(f"\nPipeline complete. {len(all_candidates)} candidates generated.")
    if top_candidates:
        print(f"Top candidate score: {top_candidates[0].final_score:.2f}")

    # Calculate actual runtime
    runtime_seconds = time.time() - start_time

    # Write inputs (info.json at PDB root, entity-specific config)
    _write_inputs(dirs["pdb_root"], dirs["entity"], config)

    # Write metrics CSV
    _write_metrics_csv(dirs["metrics"], all_candidates)

    # Create symlinks for best candidates
    _create_best_symlinks(dirs["best"], top_candidates, dirs["validation_boltz"])

    # Commit data volume
    data_volume.commit()

    return PipelineResult(
        run_id=run_id,
        config=config,
        candidates=all_candidates,
        top_candidates=top_candidates,
        validation_summary=validation_results,
        compute_cost_usd=estimated_cost,  # Would need actual tracking for precision
        runtime_seconds=runtime_seconds,
    )


# =============================================================================
# Pipeline Step Implementations
# =============================================================================


def _run_backbone_generation(
    target: TargetProtein,
    config: RFDiffusionConfig,
    output_dir: str,
    use_mocks: bool,
) -> list[BackboneDesign]:
    """Execute backbone generation step."""
    if use_mocks:
        return mock_rfdiffusion(target, config, output_dir)

    try:
        return run_rfdiffusion.remote(target, config, output_dir)
    except Exception as e:
        print(f"RFDiffusion failed: {e}")
        return []


def _run_sequence_design(
    backbones: list[BackboneDesign],
    config: ProteinMPNNConfig,
    output_dir: str,
    use_mocks: bool,
) -> list[SequenceDesign]:
    """Execute sequence design step for all backbones."""
    if use_mocks:
        all_sequences = []
        for backbone in backbones:
            seqs = mock_proteinmpnn(backbone, config, f"{output_dir}/{backbone.design_id}")
            all_sequences.extend(seqs)
        return all_sequences

    try:
        # Use parallel generation for efficiency
        return generate_sequences_parallel.remote(backbones, config, output_dir)
    except Exception as e:
        print(f"ProteinMPNN failed: {e}")
        return []


def _run_structure_validation(
    sequences: list[SequenceDesign],
    target: TargetProtein,
    config: Boltz2Config,
    output_dir: str,
    use_mocks: bool,
) -> list[StructurePrediction]:
    """Execute structure validation step."""
    if use_mocks:
        predictions = []
        for seq in sequences:
            pred = mock_boltz2(seq, target, config, f"{output_dir}/{seq.sequence_id}")
            if pred is not None:
                predictions.append(pred)
        return predictions

    try:
        return validate_sequences_parallel.remote(sequences, target, config, output_dir)
    except Exception as e:
        print(f"Boltz-2 validation failed: {e}")
        return []


def _run_decoy_search(
    target: TargetProtein,
    config: FoldSeekConfig,
    output_dir: str,
    use_mocks: bool,
) -> list[DecoyHit]:
    """Execute decoy search step."""
    if use_mocks:
        return mock_foldseek(target, config, output_dir)

    try:
        decoys = run_foldseek.remote(target, config, output_dir)

        # Download actual PDB structures for valid hits
        if decoys:
            decoys = download_decoy_structures.remote(
                decoys, f"{output_dir}/structures"
            )

        return decoys
    except Exception as e:
        print(f"FoldSeek failed: {e}")
        return []


def _run_cross_reactivity_check(
    sequences: list[SequenceDesign],
    decoys: list[DecoyHit],
    config: Chai1Config,
    output_dir: str,
    use_mocks: bool,
    target: Optional[TargetProtein] = None,
) -> tuple[dict[str, CrossReactivityResult], dict[str, list[CrossReactivityResult]]]:
    """
    Execute cross-reactivity check step.
    
    Returns:
        Tuple of:
        - positive_controls: dict mapping sequence_id to target binding result
        - decoy_results: dict mapping sequence_id to list of decoy binding results
    """
    positive_controls: dict[str, CrossReactivityResult] = {}
    decoy_results: dict[str, list[CrossReactivityResult]] = {}
    
    if use_mocks:
        # Mock positive control
        if target is not None:
            for seq in sequences:
                positive_controls[seq.sequence_id] = CrossReactivityResult(
                    binder_id=seq.sequence_id,
                    decoy_id="TARGET",
                    predicted_affinity=-10.0,  # Strong binding expected
                    plddt_interface=85.0,
                    binds_decoy=True,
                )
        
        # Mock decoy results
        for seq in sequences:
            seq_results = []
            for decoy in decoys:
                result = mock_chai1(
                    seq, decoy, config, f"{output_dir}/{seq.sequence_id}/{decoy.decoy_id}"
                )
                seq_results.append(result)
            decoy_results[seq.sequence_id] = seq_results
        return positive_controls, decoy_results

    try:
        return check_cross_reactivity_parallel.remote(
            sequences, decoys, config, output_dir, target
        )
    except Exception as e:
        print(f"Chai-1 cross-reactivity check failed: {e}")
        return {}, {}


def _create_empty_result(
    run_id: str,
    config: PipelineConfig,
    validation_results: list[ValidationResult],
    start_time: float,
) -> PipelineResult:
    """Create an empty result when pipeline fails early."""
    return PipelineResult(
        run_id=run_id,
        config=config,
        candidates=[],
        top_candidates=[],
        validation_summary=validation_results,
        compute_cost_usd=0.0,
        runtime_seconds=time.time() - start_time,
    )


def _write_inputs(pdb_root: str, entity_dir: str, config: PipelineConfig) -> None:
    """Write info.json at PDB root and entity-specific config."""
    import json

    t = config.target

    # PDB-level info.json (shared across entities)
    info_path = f"{pdb_root}/info.json"
    if not os.path.exists(info_path):
        with open(info_path, "w") as f:
            json.dump({"pdb_id": t.pdb_id}, f, indent=2)

    # Update info.json with entity metadata
    with open(info_path, "r") as f:
        info = json.load(f)
    entity_key = f"entity_{t.entity_id}"
    if entity_key not in info:
        info[entity_key] = {"name": t.name, "chain_id": t.chain_id}
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)

    # Entity-level config
    config_path = f"{entity_dir}/config.json"
    if not os.path.exists(config_path):
        with open(config_path, "w") as f:
            json.dump({
                "entity_id": t.entity_id,
                "name": t.name,
                "chain_id": t.chain_id,
                "hotspot_residues": t.hotspot_residues,
            }, f, indent=2)


def _write_metrics_csv(metrics_dir: str, candidates: list[BinderCandidate]) -> None:
    """Write scores_combined.csv with all candidate metrics."""
    import csv

    csv_path = f"{metrics_dir}/scores_combined.csv"
    if not candidates:
        return

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "candidate_id", "sequence", "ppi_score", "iptm", "ptm",
            "pae_interaction", "ptm_binder", "rmsd", "specificity", "selectivity", "final_score"
        ])
        for c in candidates:
            pred = c.structure_prediction
            writer.writerow([
                c.candidate_id,
                c.sequence,
                f"{pred.ppi_score:.3f}" if pred.ppi_score else "",
                f"{pred.iptm:.3f}" if pred.iptm else "",
                f"{pred.ptm:.3f}" if pred.ptm else "",
                f"{pred.pae_interaction:.2f}" if pred.pae_interaction else "",
                f"{pred.ptm_binder:.3f}" if pred.ptm_binder else "",
                f"{pred.rmsd_to_design:.2f}" if pred.rmsd_to_design else "",
                f"{c.specificity_score:.1f}",
                f"{c.selectivity_score:.1f}",
                f"{c.final_score:.2f}",
            ])


def _create_best_symlinks(
    best_dir: str, top_candidates: list[BinderCandidate], validation_dir: str
) -> None:
    """Create symlinks to top candidates in best_candidates directory."""
    import glob

    for candidate in top_candidates:
        # Find the predicted structure file
        pattern = f"{validation_dir}/{candidate.structure_prediction.sequence_id}/**/*.pdb"
        pdb_files = glob.glob(pattern, recursive=True)
        if not pdb_files:
            pattern = f"{validation_dir}/{candidate.structure_prediction.sequence_id}/**/*.cif"
            pdb_files = glob.glob(pattern, recursive=True)

        if pdb_files:
            src = pdb_files[0]
            ext = os.path.splitext(src)[1]
            dst = f"{best_dir}/{candidate.candidate_id}{ext}"
            if not os.path.exists(dst):
                try:
                    os.symlink(os.path.relpath(src, best_dir), dst)
                except OSError:
                    pass  # Symlinks may not work on all filesystems


# =============================================================================
# CLI Entry Point
# =============================================================================


def print_dry_run_summary(config: PipelineConfig) -> None:
    """
    Print a detailed human-readable overview of deployment parameters and costs.
    """
    num_backbones = config.rfdiffusion.num_designs
    num_sequences_per_backbone = config.proteinmpnn.num_sequences
    total_sequences = num_backbones * num_sequences_per_backbone
    num_decoys = config.foldseek.max_hits

    # Assume 50% pass Boltz-2 validation (same assumption as estimate_cost)
    passing_sequences = int(total_sequences * 0.5)
    chai1_pairs = passing_sequences * num_decoys

    # Calculate per-step costs
    rfdiffusion_hours = (STEP_RUNTIMES["rfdiffusion"] * num_backbones) / 3600
    rfdiffusion_cost = rfdiffusion_hours * GPU_COSTS["A10G"]

    proteinmpnn_hours = (STEP_RUNTIMES["proteinmpnn"] * num_backbones) / 3600
    proteinmpnn_cost = proteinmpnn_hours * GPU_COSTS["L4"]

    boltz2_hours = (STEP_RUNTIMES["boltz2"] * total_sequences) / 3600
    boltz2_cost = boltz2_hours * GPU_COSTS["A100"]

    foldseek_cost = 0.01

    chai1_hours = (STEP_RUNTIMES["chai1"] * chai1_pairs) / 3600
    chai1_cost = chai1_hours * GPU_COSTS["A100"]

    total_cost = rfdiffusion_cost + proteinmpnn_cost + boltz2_cost + foldseek_cost + chai1_cost

    # Estimate total runtime (sequential worst-case)
    est_runtime_sec = (
        STEP_RUNTIMES["rfdiffusion"] * num_backbones
        + STEP_RUNTIMES["proteinmpnn"] * num_backbones
        + STEP_RUNTIMES["boltz2"] * total_sequences
        + STEP_RUNTIMES["foldseek"]
        + STEP_RUNTIMES["chai1"] * chai1_pairs
    )
    est_runtime_min = est_runtime_sec / 60

    print("=" * 70)
    print("  DRY RUN - DEPLOYMENT PREVIEW")
    print("=" * 70)
    print()
    print("┌─────────────────────────────────────────────────────────────────────┐")
    print("│  PIPELINE CONFIGURATION                                             │")
    print("├─────────────────────────────────────────────────────────────────────┤")
    print(f"│  Target PDB:           {config.target.pdb_path:<43} │")
    print(f"│  Chain ID:             {config.target.chain_id:<43} │")
    print(f"│  Hotspot Residues:     {str(config.target.hotspot_residues):<43} │")
    print("├─────────────────────────────────────────────────────────────────────┤")
    print(f"│  Backbone Designs:     {num_backbones:<43} │")
    print(f"│  Sequences/Backbone:   {num_sequences_per_backbone:<43} │")
    print(f"│  Total Sequences:      {total_sequences:<43} │")
    print(f"│  Max Decoys:           {num_decoys:<43} │")
    print(f"│  Budget Limit:         ${config.max_compute_usd:<42.2f} │")
    print("└─────────────────────────────────────────────────────────────────────┘")
    print()
    print("┌─────────────────────────────────────────────────────────────────────┐")
    print("│  PHASE 1: GENERATION                                                │")
    print("├─────────────────────────────────────────────────────────────────────┤")
    print(f"│  RFDiffusion     │ GPU: A10G   │ {num_backbones:>3} runs × ~60s  │  ${rfdiffusion_cost:>6.3f}  │")
    print(f"│  ProteinMPNN     │ GPU: L4     │ {num_backbones:>3} runs × ~15s  │  ${proteinmpnn_cost:>6.3f}  │")
    print("├─────────────────────────────────────────────────────────────────────┤")
    print("│  PHASE 2: VALIDATION                                                │")
    print("├─────────────────────────────────────────────────────────────────────┤")
    print(f"│  Boltz-2         │ GPU: A100   │ {total_sequences:>3} runs × ~120s │  ${boltz2_cost:>6.3f}  │")
    print("├─────────────────────────────────────────────────────────────────────┤")
    print("│  PHASE 3: SELECTIVITY                                               │")
    print("├─────────────────────────────────────────────────────────────────────┤")
    print(f"│  FoldSeek        │ CPU         │   1 run  × ~30s  │  ${foldseek_cost:>6.3f}  │")
    print(f"│  Chai-1          │ GPU: A100   │ {chai1_pairs:>3} runs × ~90s  │  ${chai1_cost:>6.3f}  │")
    print("└─────────────────────────────────────────────────────────────────────┘")
    print()
    print("┌─────────────────────────────────────────────────────────────────────┐")
    print("│  COST SUMMARY                                                       │")
    print("├─────────────────────────────────────────────────────────────────────┤")
    budget_status = "✓ WITHIN BUDGET" if total_cost <= config.max_compute_usd else "⚠ EXCEEDS BUDGET"
    print(f"│  Estimated Total Cost:      ${total_cost:<8.2f}  {budget_status:<21} │")
    print(f"│  Estimated Runtime:         ~{est_runtime_min:<6.0f} minutes (sequential)        │")
    print(f"│  Budget Limit:              ${config.max_compute_usd:<8.2f}                           │")
    print("└─────────────────────────────────────────────────────────────────────┘")
    print()

    if total_cost > config.max_compute_usd:
        scale_factor = config.max_compute_usd / total_cost
        suggested_designs = max(1, int(num_backbones * scale_factor))
        suggested_sequences = max(1, int(num_sequences_per_backbone * scale_factor))
        print("┌─────────────────────────────────────────────────────────────────────┐")
        print("│  ⚠ BUDGET WARNING                                                   │")
        print("├─────────────────────────────────────────────────────────────────────┤")
        print(f"│  Consider reducing designs to ~{suggested_designs} and sequences to ~{suggested_sequences}           │")
        print(f"│  to stay within the ${config.max_compute_usd:.2f} budget.                                │")
        print("└─────────────────────────────────────────────────────────────────────┘")
        print()

    print("To run this pipeline, remove the --dry-run flag.")
    print()


@app.function(
    image=base_image,
    volumes={DATA_PATH: data_volume},
    timeout=60,
)
def upload_target_pdb(local_pdb_content: str, filename: str) -> str:
    """
    Upload target PDB content to the data volume.
    
    Args:
        local_pdb_content: Content of the PDB file
        filename: Original filename for naming
        
    Returns:
        Path to the uploaded file on the data volume
    """
    import os
    target_dir = f"{DATA_PATH}/targets"
    os.makedirs(target_dir, exist_ok=True)
    
    remote_path = f"{target_dir}/{filename}"
    with open(remote_path, "w") as f:
        f.write(local_pdb_content)
    
    data_volume.commit()
    return remote_path


@app.local_entrypoint()
def main(
    pdb_id: str,
    entity_id: int,
    hotspot_residues: str,
    mode: str = "bind",
    num_designs: int = 5,
    num_sequences: int = 4,
    use_mocks: bool = False,
    max_budget: float = 5.0,
    dry_run: bool = False,
):
    """
    Run the protein binder design pipeline from command line.

    Args:
        pdb_id: 4-letter PDB code (e.g., "3DI3")
        entity_id: Polymer entity ID for the target (e.g., 2 for IL7RA in 3DI3)
        hotspot_residues: Comma-separated list of hotspot residue indices
        mode: Generation mode - "bind" for binder design (default: "bind")
        num_designs: Number of backbone designs (default: 5)
        num_sequences: Sequences per backbone (default: 4)
        use_mocks: Use mock implementations for testing (default: False)
        max_budget: Maximum compute budget in USD (default: 5.0)
        dry_run: Preview deployment parameters and costs without running (default: False)
    
    Example:
        uv run modal run pipeline.py --pdb-id 3DI3 --entity-id 2 --hotspot-residues "58,80,139" --mode bind
    
    Generated proteins are named: <pdb_id>_E<entity_id>_<mode>_<ulid>
    Example: 3DI3_E2_bind_01ARZ3NDEKTSV4RRFFQ69G5FAV
    """
    from common import get_entity_info, download_pdb
    import tempfile
    import os
    
    # Parse and validate mode
    mode = mode.lower().strip()
    valid_modes = [m.value for m in GenerationMode]
    if mode not in valid_modes:
        raise ValueError(f"Invalid mode '{mode}'. Valid modes: {valid_modes}")
    generation_mode = GenerationMode(mode)
    
    # Parse hotspot residues
    hotspots = [int(r.strip()) for r in hotspot_residues.split(",")]
    
    # Validate PDB ID format
    pdb_id = pdb_id.upper().strip()
    if len(pdb_id) != 4:
        raise ValueError(f"PDB ID must be 4 characters: {pdb_id}")

    # Fetch entity info from RCSB
    print(f"Fetching entity {entity_id} info for PDB {pdb_id}...")
    entity_info = get_entity_info(pdb_id, entity_id)
    chains = entity_info.get("chains", [])
    
    if not chains:
        raise ValueError(f"No chains found for entity {entity_id} in PDB {pdb_id}")
    
    chain_id = chains[0]  # Use first chain for this entity
    entity_name = entity_info.get("description", f"{pdb_id}_entity{entity_id}")
    uniprot_ids = entity_info.get("uniprot_ids", [])
    
    print(f"  Entity: {entity_name}")
    print(f"  Chain(s): {', '.join(chains)}")
    if uniprot_ids:
        print(f"  UniProt: {', '.join(uniprot_ids)}")

    # Download PDB to temp file and upload to Modal volume
    with tempfile.TemporaryDirectory() as tmpdir:
        local_pdb = os.path.join(tmpdir, f"{pdb_id.lower()}.pdb")
        print(f"Downloading PDB {pdb_id}...")
        download_pdb(pdb_id, local_pdb)
        
        with open(local_pdb, "r") as f:
            pdb_content = f.read()
        
        remote_pdb_path = upload_target_pdb.remote(pdb_content, f"{pdb_id.lower()}.pdb")
        print(f"Uploaded to: {remote_pdb_path}")

    # Build configuration
    config = PipelineConfig(
        target=TargetProtein(
            pdb_id=pdb_id,
            entity_id=entity_id,
            hotspot_residues=hotspots,
            pdb_path=remote_pdb_path,
            chain_id=chain_id,
            name=entity_name,
        ),
        mode=generation_mode,
        rfdiffusion=RFDiffusionConfig(num_designs=num_designs),
        proteinmpnn=ProteinMPNNConfig(num_sequences=num_sequences),
        boltz2=Boltz2Config(),
        foldseek=FoldSeekConfig(),
        chai1=Chai1Config(),
        max_compute_usd=max_budget,
    )

    # Dry run mode: print summary and exit
    if dry_run:
        print_dry_run_summary(config)
        return None

    # Run pipeline
    result = run_pipeline.remote(config, use_mocks=use_mocks)

    # Print summary
    print("\n" + "=" * 60)
    print("PIPELINE RESULTS")
    print("=" * 60)
    print(f"Target: {pdb_id} entity {entity_id}")
    print(f"Run ID: {result.run_id}")
    print(f"Runtime: {result.runtime_seconds:.1f} seconds")
    print(f"Estimated cost: ${result.compute_cost_usd:.2f}")
    print(f"Total candidates: {len(result.candidates)}")
    print(f"Top candidates: {len(result.top_candidates)}")

    if result.top_candidates:
        print("\n--- TOP 5 CANDIDATES ---")
        for i, candidate in enumerate(result.top_candidates[:5]):
            pred = candidate.structure_prediction
            ppi = pred.ppi_score
            print(f"\n{i+1}. {candidate.candidate_id}")
            print(f"   Sequence: {candidate.sequence[:50]}...")
            print(f"   PPI Score: {ppi:.3f} (0.8·ipTM + 0.2·pTM)")
            print(f"   Specificity: {candidate.specificity_score:.1f}")
            print(f"   Selectivity: {candidate.selectivity_score:.1f}")
            print(f"   Final Score: {candidate.final_score:.2f}")
            # AlphaProteo metrics
            if pred.pae_interaction is not None:
                print(f"   PAE@hotspots: {pred.pae_interaction:.2f} Å (< 1.5)")
            if pred.ptm_binder is not None:
                print(f"   pTM(binder): {pred.ptm_binder:.3f} (> 0.80)")
            if pred.rmsd_to_design is not None:
                print(f"   RMSD: {pred.rmsd_to_design:.2f} Å (< 2.5)")
            if pred.iptm is not None and pred.ptm is not None:
                print(f"   ipTM: {pred.iptm:.3f}, pTM: {pred.ptm:.3f}")

            if candidate.decoy_results:
                binding_decoys = sum(1 for r in candidate.decoy_results if r.binds_decoy)
                print(f"   Binds {binding_decoys}/{len(candidate.decoy_results)} decoys")

    print("\n" + "=" * 60)
    print("Pipeline complete!")

    return result


# =============================================================================
# Programmatic Entry Point
# =============================================================================


def design_binders(
    target_pdb_path: str,
    hotspot_residues: list[int],
    chain_id: str = "A",
    num_designs: int = 5,
    num_sequences: int = 4,
    max_budget: float = 5.0,
    use_mocks: bool = False,
) -> PipelineResult:
    """
    Programmatic interface to run the binder design pipeline.

    Args:
        target_pdb_path: Path to target protein PDB file
        hotspot_residues: List of hotspot residue indices
        chain_id: Target chain ID
        num_designs: Number of backbone designs
        num_sequences: Sequences per backbone
        max_budget: Maximum compute budget in USD
        use_mocks: Use mock implementations for testing

    Returns:
        PipelineResult with ranked binder candidates
    """
    config = PipelineConfig(
        target=TargetProtein(
            pdb_path=target_pdb_path,
            chain_id=chain_id,
            hotspot_residues=hotspot_residues,
            name="target",
        ),
        rfdiffusion=RFDiffusionConfig(num_designs=num_designs),
        proteinmpnn=ProteinMPNNConfig(num_sequences=num_sequences),
        max_compute_usd=max_budget,
    )

    with modal.enable_local():
        return run_pipeline.local(config, use_mocks=use_mocks)


if __name__ == "__main__":
    # Example usage with mocks for testing
    import sys

    if len(sys.argv) > 1:
        # Use provided PDB path
        result = design_binders(
            target_pdb_path=sys.argv[1],
            hotspot_residues=[10, 15, 20],
            use_mocks=True,
        )
    else:
        print("Usage: python pipeline.py <target.pdb>")
        print("\nRunning with mock data for demonstration...")

        # Create a mock target PDB for testing
        os.makedirs("/tmp/test_pipeline", exist_ok=True)
        mock_pdb = "/tmp/test_pipeline/mock_target.pdb"
        with open(mock_pdb, "w") as f:
            f.write("HEADER    MOCK TARGET\n")
            for i in range(100):
                f.write(
                    f"ATOM  {i+1:5d}  CA  ALA A{i+1:4d}    "
                    f"{i*3.8:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C\n"
                )
            f.write("END\n")

        result = design_binders(
            target_pdb_path=mock_pdb,
            hotspot_residues=[10, 15, 20, 25, 30],
            num_designs=3,
            num_sequences=2,
            use_mocks=True,
        )

        print(f"\nGenerated {len(result.candidates)} candidates")
        if result.best_candidate:
            print(f"Best candidate: {result.best_candidate.candidate_id}")
            print(f"Best score: {result.best_candidate.final_score:.2f}")
