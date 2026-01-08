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
import uuid
from datetime import datetime
from typing import Optional

import modal

from common import (
    APP_NAME,
    DATA_PATH,
    WEIGHTS_PATH,
    BackboneDesign,
    BinderCandidate,
    Boltz2Config,
    Chai1Config,
    CrossReactivityResult,
    DecoyHit,
    FoldSeekConfig,
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
    base_image,
    data_volume,
    generate_design_id,
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
    download_decoy_structures,
    mock_boltz2,
    mock_chai1,
    mock_foldseek,
    run_boltz2,
    run_chai1_batch,
    run_foldseek,
    validate_sequences_parallel,
)

# =============================================================================
# Modal App Definition
# =============================================================================

app = modal.App(APP_NAME)

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
    keep_warm=1,  # Keep orchestrator warm for quick starts
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
    run_id = f"run_{uuid.uuid4().hex[:12]}"
    base_output_dir = f"{DATA_PATH}/{run_id}"
    os.makedirs(base_output_dir, exist_ok=True)

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

    print(f"Starting pipeline run: {run_id}")
    print(f"Estimated cost: ${estimated_cost:.2f}")
    print(f"Configuration: {config.rfdiffusion.num_designs} backbones × "
          f"{config.proteinmpnn.num_sequences} sequences each")

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
        f"{base_output_dir}/backbones",
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
        f"{base_output_dir}/sequences",
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
        f"{base_output_dir}/predictions",
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

    # =========================================================================
    # Phase 3: Negative Selection (Selectivity)
    # =========================================================================

    # Step 4: FoldSeek - Find Structural Decoys
    print("\n=== Phase 3: Decoy Identification (FoldSeek) ===")
    decoys = _run_decoy_search(
        config.target,
        config.foldseek,
        f"{base_output_dir}/decoys",
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

    # Step 5: Chai-1 - Cross-Reactivity Check
    cross_reactivity_results: dict[str, list[CrossReactivityResult]] = {}

    if decoys:
        print("\n=== Phase 3: Cross-Reactivity Check (Chai-1) ===")
        # Get sequences that passed validation
        validated_sequences = [
            seq for seq in sequences
            if any(p.sequence_id == seq.sequence_id for p in predictions)
        ]

        cross_reactivity_results = _run_cross_reactivity_check(
            validated_sequences,
            decoys,
            config.chai1,
            f"{base_output_dir}/cross_reactivity",
            use_mocks,
        )
        print(f"Checked cross-reactivity for {len(cross_reactivity_results)} sequences")

        validation_results.append(
            ValidationResult(
                stage=PipelineStage.CROSS_REACTIVITY,
                status=ValidationStatus.PASSED,
                candidate_id=run_id,
                metrics={"num_checked": len(cross_reactivity_results)},
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

        # Compute scores
        specificity_score = prediction.plddt_interface

        # Selectivity: penalize if binder binds any decoy
        max_decoy_affinity = 0.0
        if cr_results:
            max_decoy_affinity = max(abs(r.predicted_affinity) for r in cr_results)
        selectivity_score = 100.0 - max_decoy_affinity * 10  # Heuristic scaling

        # Final score using the S(x) formula
        final_score = (
            config.scoring.alpha * specificity_score
            - config.scoring.beta * max_decoy_affinity
        )

        candidate = BinderCandidate(
            candidate_id=generate_design_id("candidate"),
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
) -> dict[str, list[CrossReactivityResult]]:
    """Execute cross-reactivity check step."""
    if use_mocks:
        results: dict[str, list[CrossReactivityResult]] = {}
        for seq in sequences:
            seq_results = []
            for decoy in decoys:
                result = mock_chai1(
                    seq, decoy, config, f"{output_dir}/{seq.sequence_id}/{decoy.decoy_id}"
                )
                seq_results.append(result)
            results[seq.sequence_id] = seq_results
        return results

    try:
        return check_cross_reactivity_parallel.remote(
            sequences, decoys, config, output_dir
        )
    except Exception as e:
        print(f"Chai-1 cross-reactivity check failed: {e}")
        return {}


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


# =============================================================================
# CLI Entry Point
# =============================================================================


@app.local_entrypoint()
def main(
    target_pdb: str,
    hotspot_residues: str,
    chain_id: str = "A",
    num_designs: int = 5,
    num_sequences: int = 4,
    use_mocks: bool = False,
    max_budget: float = 5.0,
):
    """
    Run the protein binder design pipeline from command line.

    Args:
        target_pdb: Path to target protein PDB file
        hotspot_residues: Comma-separated list of hotspot residue indices
        chain_id: Target chain ID (default: A)
        num_designs: Number of backbone designs (default: 5)
        num_sequences: Sequences per backbone (default: 4)
        use_mocks: Use mock implementations for testing (default: False)
        max_budget: Maximum compute budget in USD (default: 5.0)
    """
    # Parse hotspot residues
    hotspots = [int(r.strip()) for r in hotspot_residues.split(",")]

    # Build configuration
    config = PipelineConfig(
        target=TargetProtein(
            pdb_path=target_pdb,
            chain_id=chain_id,
            hotspot_residues=hotspots,
            name="target",
        ),
        rfdiffusion=RFDiffusionConfig(num_designs=num_designs),
        proteinmpnn=ProteinMPNNConfig(num_sequences=num_sequences),
        boltz2=Boltz2Config(),
        foldseek=FoldSeekConfig(),
        chai1=Chai1Config(),
        max_compute_usd=max_budget,
    )

    print("=" * 60)
    print("PROTEIN BINDER DESIGN PIPELINE")
    print("=" * 60)
    print(f"Target PDB: {target_pdb}")
    print(f"Hotspot residues: {hotspots}")
    print(f"Designs: {num_designs} backbones × {num_sequences} sequences")
    print(f"Budget: ${max_budget:.2f}")
    print("=" * 60)

    # Run pipeline
    result = run_pipeline.remote(config, use_mocks=use_mocks)

    # Print summary
    print("\n" + "=" * 60)
    print("PIPELINE RESULTS")
    print("=" * 60)
    print(f"Run ID: {result.run_id}")
    print(f"Runtime: {result.runtime_seconds:.1f} seconds")
    print(f"Estimated cost: ${result.compute_cost_usd:.2f}")
    print(f"Total candidates: {len(result.candidates)}")
    print(f"Top candidates: {len(result.top_candidates)}")

    if result.top_candidates:
        print("\n--- TOP 5 CANDIDATES ---")
        for i, candidate in enumerate(result.top_candidates[:5]):
            print(f"\n{i+1}. {candidate.candidate_id}")
            print(f"   Sequence: {candidate.sequence[:50]}...")
            print(f"   Specificity: {candidate.specificity_score:.1f}")
            print(f"   Selectivity: {candidate.selectivity_score:.1f}")
            print(f"   Final Score: {candidate.final_score:.2f}")
            print(f"   i-pLDDT: {candidate.structure_prediction.plddt_interface:.1f}")
            print(f"   PAE: {candidate.structure_prediction.pae_interface:.1f}")

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
