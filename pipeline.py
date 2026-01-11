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
    AdaptiveGenerationConfig,
    BackboneDesign,
    BackboneFilterConfig,
    BinderCandidate,
    Boltz2Config,
    Chai1Config,
    CrossReactivityResult,
    DecoyHit,
    FoldSeekConfig,
    GenerationMode,
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
    generate_protein_id,
    generate_ulid,
    load_config_from_yaml,
)
from state_tree import (
    PipelineStateTree,
    NodeStatus,
    create_state_tree,
)
from generators import (
    filter_backbones_by_quality,
    generate_sequences_parallel,
    mock_proteinmpnn,
    mock_rfdiffusion,
    run_rfdiffusion,
)
from validators import (
    check_cross_reactivity_parallel,
    check_novelty,
    cluster_by_tm_score,
    download_decoy_structures,
    get_cached_decoys,
    mock_boltz2,
    mock_chai1,
    mock_foldseek,
    run_foldseek,
    save_decoys_to_cache,
    validate_sequences_parallel,
)

# Note: Local modules are added to all images in common.py via _add_local_modules()

# =============================================================================
# Cost Estimation
# =============================================================================

# =============================================================================
# Modal Pricing (per second) - https://modal.com/pricing
# =============================================================================

MODAL_GPU_COST_PER_SEC = {
    "H100": 0.001097,
    "A100-80GB": 0.000694,
    "A100": 0.000583,  # 40GB variant
    "L40S": 0.000542,
    "A10G": 0.000306,
    "L4": 0.000222,
    "T4": 0.000164,
}

MODAL_CPU_COST_PER_CORE_SEC = 0.0000131
MODAL_MEMORY_COST_PER_GIB_SEC = 0.00000222

# Default timeout ceilings per step (seconds) - from @app.function decorators
# These represent worst-case billing durations when no limits are specified
DEFAULT_STEP_TIMEOUTS = {
    "rfdiffusion": 600,   # generators.py: run_rfdiffusion timeout=600
    "proteinmpnn": 300,   # generators.py: run_proteinmpnn timeout=300
    "boltz2": 900,        # validators.py: run_boltz2 timeout=900
    "foldseek": 120,      # validators.py: download_decoy_structures timeout=120
    "chai1": 900,         # validators.py: run_chai1 timeout=900
}

# Default max_designs per step when no limits are specified
DEFAULT_MAX_DESIGNS = {
    "rfdiffusion": 4,     # Max backbones to generate
    "proteinmpnn": 16,    # Max total sequences
    "boltz2": 3,          # Max sequences to validate (Boltz-2 @ $0.74/seq)
    "foldseek": 2,        # Max decoys
    "chai1": 3,           # Max binder-decoy pairs (Chai-1 @ $0.74/pair)
}

# Resource allocation per step - from @app.function decorators
STEP_RESOURCES = {
    "rfdiffusion": {"gpu": "A10G", "cpu_cores": 2, "memory_gib": 16},
    "proteinmpnn": {"gpu": "L4", "cpu_cores": 2, "memory_gib": 8},
    "boltz2": {"gpu": "A100-80GB", "cpu_cores": 4, "memory_gib": 32},
    "foldseek": {"gpu": None, "cpu_cores": 2, "memory_gib": 8},
    "chai1": {"gpu": "A100-80GB", "cpu_cores": 4, "memory_gib": 32},
}


def _calculate_step_cost(step: str, duration_sec: float) -> float:
    """Calculate cost for a single step including GPU, CPU, and memory."""
    resources = STEP_RESOURCES[step]
    
    # GPU cost
    gpu_cost = 0.0
    if resources["gpu"]:
        gpu_cost = MODAL_GPU_COST_PER_SEC[resources["gpu"]] * duration_sec
    
    # CPU cost
    cpu_cost = MODAL_CPU_COST_PER_CORE_SEC * resources["cpu_cores"] * duration_sec
    
    # Memory cost
    mem_cost = MODAL_MEMORY_COST_PER_GIB_SEC * resources["memory_gib"] * duration_sec
    
    return gpu_cost + cpu_cost + mem_cost


def estimate_cost(config: PipelineConfig) -> dict:
    """
    Estimate the worst-case compute cost ceiling for a pipeline run.
    
    Uses timeout ceilings and max_designs limits from config.limits to provide
    a deterministic upper bound based on configured limits.
    
    Cost calculation uses min(requested, limit) for each dimension:
    - Timeouts: Uses config.limits timeout or default
    - Max designs: Uses min(requested designs, config.limits max_designs)

    Args:
        config: Pipeline configuration

    Returns:
        Dict with per-step costs, limits used, and total
    """
    limits = config.limits
    
    # ==========================================================================
    # Tree Degree Indexing: limits represent branching factor at each level
    # ==========================================================================
    
    # Level 1: RFDiffusion - backbones per target (degree from root)
    requested_backbones = config.rfdiffusion.num_designs
    max_backbones_per_target = limits.get_max_designs("rfdiffusion")
    effective_backbones = min(requested_backbones, max_backbones_per_target)
    
    # Level 2: ProteinMPNN - sequences per backbone (degree per backbone)
    requested_seqs_per_backbone = config.proteinmpnn.num_sequences
    max_seqs_per_backbone = limits.get_max_designs("proteinmpnn")
    effective_seqs_per_backbone = min(requested_seqs_per_backbone, max_seqs_per_backbone)
    
    # Derived: total sequences in tree
    effective_sequences = effective_backbones * effective_seqs_per_backbone
    
    # Level 3: Boltz-2 - total cap on validations (cost control, not degree)
    max_boltz2 = limits.get_max_designs("boltz2")
    effective_boltz2 = min(effective_sequences, max_boltz2)
    
    # FoldSeek: decoys per target (degree from target)
    requested_decoys = config.foldseek.max_hits
    max_decoys_per_target = limits.get_max_designs("foldseek")
    effective_decoys = min(requested_decoys, max_decoys_per_target)
    
    # Level 4: Chai-1 - max sequences to check (each checked against ALL decoys)
    max_chai1_sequences = limits.get_max_designs("chai1")
    effective_chai1_sequences = min(effective_boltz2, max_chai1_sequences)
    effective_chai1_pairs = effective_chai1_sequences * effective_decoys
    
    # Get effective timeouts from limits
    rfdiffusion_timeout = limits.get_timeout("rfdiffusion")
    proteinmpnn_timeout = limits.get_timeout("proteinmpnn")
    boltz2_timeout = limits.get_timeout("boltz2")
    foldseek_timeout = limits.get_timeout("foldseek")
    chai1_timeout = limits.get_timeout("chai1")

    # Calculate per-step costs using timeout ceilings (worst-case billing)
    costs = {}
    
    # RFDiffusion (A10G) - timeout for batch generation (all backbones in one call)
    rfdiffusion_sec = rfdiffusion_timeout
    costs["rfdiffusion"] = _calculate_step_cost("rfdiffusion", rfdiffusion_sec)
    
    # ProteinMPNN (L4) - timeout per backbone
    proteinmpnn_sec = proteinmpnn_timeout * effective_backbones
    costs["proteinmpnn"] = _calculate_step_cost("proteinmpnn", proteinmpnn_sec)
    
    # Boltz-2 (A100) - timeout per sequence
    boltz2_sec = boltz2_timeout * effective_boltz2
    costs["boltz2"] = _calculate_step_cost("boltz2", boltz2_sec)
    
    # FoldSeek (CPU only) - single timeout
    foldseek_sec = foldseek_timeout
    costs["foldseek"] = _calculate_step_cost("foldseek", foldseek_sec)
    
    # Chai-1 (A100) - timeout per binder-decoy pair
    chai1_sec = chai1_timeout * effective_chai1_pairs
    costs["chai1"] = _calculate_step_cost("chai1", chai1_sec)
    
    costs["total"] = sum(costs.values())
    
    # Include effective counts and timeouts for reporting
    costs["_effective"] = {
        "backbones": effective_backbones,
        "seqs_per_backbone": effective_seqs_per_backbone,
        "sequences": effective_sequences,
        "boltz2_validations": effective_boltz2,
        "decoys": effective_decoys,
        "chai1_sequences": effective_chai1_sequences,
        "chai1_pairs": effective_chai1_pairs,
    }
    costs["_timeouts"] = {
        "rfdiffusion": rfdiffusion_timeout,
        "proteinmpnn": proteinmpnn_timeout,
        "boltz2": boltz2_timeout,
        "foldseek": foldseek_timeout,
        "chai1": chai1_timeout,
    }
    costs["_runtime_sec"] = rfdiffusion_sec + proteinmpnn_sec + boltz2_sec + foldseek_sec + chai1_sec
    
    return costs


# =============================================================================
# Stage Logging
# =============================================================================


def _log_stage_metrics(
    stage_name: str,
    duration_sec: float,
    candidates: int,
    ceiling_sec: float,
    ceiling_cost: float,
) -> dict:
    """
    Log actual vs. ceiling metrics for a pipeline stage.
    
    Args:
        stage_name: Name of the stage (e.g., "RFDiffusion")
        duration_sec: Actual duration in seconds
        candidates: Number of candidates produced
        ceiling_sec: Ceiling timeout in seconds
        ceiling_cost: Ceiling cost in USD
    
    Returns:
        Dict with stage metrics for inclusion in run manifest
    """
    actual_cost = _calculate_step_cost(stage_name.lower().replace("-", ""), duration_sec)
    utilization = (duration_sec / ceiling_sec * 100) if ceiling_sec > 0 else 0
    
    print(f"  ├─ Duration: {duration_sec:.1f}s / {ceiling_sec:.0f}s ({utilization:.0f}% of ceiling)")
    print(f"  ├─ Candidates: {candidates}")
    print(f"  └─ Est. cost: ${actual_cost:.3f} / ${ceiling_cost:.3f} ceiling")
    
    return {
        "stage": stage_name,
        "duration_sec": duration_sec,
        "ceiling_sec": ceiling_sec,
        "candidates": candidates,
        "actual_cost_usd": actual_cost,
        "ceiling_cost_usd": ceiling_cost,
        "utilization_pct": utilization,
    }


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
    
    # Initialize state tree for observability
    state_tree = create_state_tree(
        run_id=run_id,
        config=config.model_dump(mode="json") if hasattr(config, 'model_dump') else None
    )
    
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

    # ==========================================================================
    # ENFORCE HARD LIMITS from config.limits (Tree Degree Indexing)
    # ==========================================================================
    # The pipeline creates a tree structure. Limits represent the DEGREE
    # (branching factor) at each node level:
    #
    #   Target (root)
    #   ├── [RFDiffusion] → degree = max backbones per target
    #   │   └── [ProteinMPNN] → degree = max sequences per backbone
    #   │       └── [Boltz-2] → 1:1 mapping, capped by total (cost control)
    #   │           └── [Chai-1] → degree = decoys per validated sequence
    #   └── [FoldSeek] → degree = max decoys per target
    #
    # ==========================================================================
    limits = config.limits
    
    # Level 1: RFDiffusion - max backbones per target (degree from root)
    max_backbones_per_target = limits.get_max_designs("rfdiffusion")
    if config.rfdiffusion.num_designs > max_backbones_per_target:
        print(f"[LIMIT] RFDiffusion: {config.rfdiffusion.num_designs} → {max_backbones_per_target} backbones/target")
        config.rfdiffusion.num_designs = max_backbones_per_target
    
    # Level 2: ProteinMPNN - max sequences per backbone (degree per backbone node)
    max_seqs_per_backbone = limits.get_max_designs("proteinmpnn")
    if config.proteinmpnn.num_sequences > max_seqs_per_backbone:
        print(f"[LIMIT] ProteinMPNN: {config.proteinmpnn.num_sequences} → {max_seqs_per_backbone} seqs/backbone")
        config.proteinmpnn.num_sequences = max_seqs_per_backbone
    
    # Derived: total sequences in tree
    total_sequences = config.rfdiffusion.num_designs * config.proteinmpnn.num_sequences
    
    # Level 3: Boltz-2 - max validations total (cost control, not degree)
    # This caps how many leaf nodes we validate, not branching factor
    max_boltz2_validations = limits.get_max_designs("boltz2")
    
    # FoldSeek: max decoys per target (degree from target node)
    max_decoys_per_target = limits.get_max_designs("foldseek")
    if config.foldseek.max_hits > max_decoys_per_target:
        print(f"[LIMIT] FoldSeek: {config.foldseek.max_hits} → {max_decoys_per_target} decoys/target")
        config.foldseek.max_hits = max_decoys_per_target
    
    # Level 4: Chai-1 - max validated sequences to check (cost control)
    # Each validated sequence is checked against ALL decoys
    # Limit controls how many sequences proceed to cross-reactivity check
    max_chai1_sequences = limits.get_max_designs("chai1")
    
    # Pre-flight cost check (only if budget is set)
    cost_estimate = estimate_cost(config)
    if config.max_compute_usd is not None and cost_estimate["total"] > config.max_compute_usd:
        print(
            f"WARNING: Estimated cost ${cost_estimate['total']:.2f} exceeds budget "
            f"${config.max_compute_usd:.2f}. Reducing design count."
        )
        # Scale down to fit budget
        scale_factor = config.max_compute_usd / cost_estimate["total"]
        config.rfdiffusion.num_designs = max(
            1, int(config.rfdiffusion.num_designs * scale_factor)
        )
        config.proteinmpnn.num_sequences = max(
            1, int(config.proteinmpnn.num_sequences * scale_factor)
        )
        # Recalculate after scaling
        cost_estimate = estimate_cost(config)

    # Pipeline configuration
    t = config.target
    print(f"\n{'='*70}")
    print(f"PIPELINE RUN: {run_id}")
    print(f"{'='*70}")
    print(f"Target: {t.name}")
    print(f"PDB ID: {t.pdb_id} | Entity ID: {t.entity_id}")
    print(f"Hotspot residues: {', '.join(str(r) for r in t.hotspot_residues)}")
    print("\nConfiguration:")
    print(f"├─ Backbones: {config.rfdiffusion.num_designs} (max: {max_backbones_per_target})")
    print(f"│  └─ Sequences: {config.proteinmpnn.num_sequences}/backbone (max: {max_seqs_per_backbone}) → {total_sequences} total")
    print(f"│     └─ Validations: up to {max_boltz2_validations}")
    print(f"│        └─ Chai-1: up to {max_chai1_sequences} seqs × {config.foldseek.max_hits} decoys")
    print(f"└─ Decoys: {config.foldseek.max_hits} (max: {max_decoys_per_target})")
    print(f"\nCost ceiling: ${cost_estimate['total']:.2f}")
    print(f"  ├─ RFDiffusion (A10G):  ${cost_estimate['rfdiffusion']:.3f}")
    print(f"  ├─ ProteinMPNN (L4):    ${cost_estimate['proteinmpnn']:.3f}")
    print(f"  ├─ Boltz-2 (A100):      ${cost_estimate['boltz2']:.3f}")
    print(f"  ├─ FoldSeek (CPU):      ${cost_estimate['foldseek']:.3f}")
    print(f"  └─ Chai-1 (A100):       ${cost_estimate['chai1']:.3f}")

    # Write run manifest to campaign directory
    _write_run_manifest(campaign_dir, run_id, config, cost_estimate)

    # Add target node to state tree
    target_node_id = state_tree.add_target(
        pdb_id=t.pdb_id,
        entity_id=t.entity_id,
        name=t.name,
        hotspot_residues=t.hotspot_residues,
    )
    state_tree.set_status(target_node_id, NodeStatus.COMPLETED)
    
    # Set ceiling (passive observability) values in state tree
    # These represent worst-case estimates based on timeouts × max_designs
    state_tree.set_ceilings(
        run_id,
        ceiling_cost_usd=cost_estimate["total"],
        ceiling_timing_sec=cost_estimate["_runtime_sec"],
    )
    state_tree.set_metrics(run_id, {
        "cost_estimate": {
            "total": cost_estimate["total"],
            "rfdiffusion": cost_estimate["rfdiffusion"],
            "proteinmpnn": cost_estimate["proteinmpnn"],
            "boltz2": cost_estimate["boltz2"],
            "foldseek": cost_estimate["foldseek"],
            "chai1": cost_estimate["chai1"],
        },
        "effective_limits": cost_estimate["_effective"],
        "timeouts": cost_estimate["_timeouts"],
        "max_compute_usd": config.max_compute_usd if config.max_compute_usd is not None else "unlimited",
    })

    validation_results: list[ValidationResult] = []
    all_candidates: list[BinderCandidate] = []
    stage_metrics: list[dict] = []  # Track actual vs ceiling for each stage

    # =========================================================================
    # Phase 1: Generation (Specificity)
    # =========================================================================

    # Step 1: RFDiffusion - Backbone Generation
    # Use adaptive generation if enabled, otherwise standard batch generation
    if config.adaptive.enabled and not use_mocks:
        print("\n=== Phase 1: Adaptive Generation (RFDiffusion + ProteinMPNN + Boltz-2) ===")
        stage_start = time.time()
        backbones, sequences, predictions = _run_adaptive_generation(
            config.target,
            config.rfdiffusion,
            config.proteinmpnn,
            config.boltz2,
            config.adaptive,
            config.backbone_filter,
            dirs,
            max_boltz2_per_batch=max_boltz2_validations,
        )
        stage_duration = time.time() - stage_start
        print(f"Adaptive generation complete: {len(backbones)} backbones, {len(sequences)} sequences, {len(predictions)} validated")
        print(f"  └─ Total duration: {stage_duration:.1f}s")
        
        # Add adaptive generation nodes to state tree
        for backbone in backbones:
            backbone_node_id = state_tree.add_backbone(
                design_id=backbone.design_id,
                parent_id=target_node_id,
                pdb_path=backbone.pdb_path,
                binder_length=backbone.binder_length,
                rfdiffusion_score=backbone.rfdiffusion_score,
            )
            state_tree.end_timing(backbone_node_id, NodeStatus.COMPLETED)
        
        for seq in sequences:
            seq_node_id = state_tree.add_sequence(
                sequence_id=seq.sequence_id,
                backbone_id=seq.backbone_id,
                sequence=seq.sequence,
                score=seq.score,
                fasta_path=seq.fasta_path,
            )
            # Mark as completed or filtered based on whether it has a prediction
            has_prediction = any(p.sequence_id == seq.sequence_id for p in predictions)
            state_tree.end_timing(
                seq_node_id,
                status=NodeStatus.COMPLETED if has_prediction else NodeStatus.FILTERED,
            )
        
        for pred in predictions:
            pred_node_id = state_tree.add_prediction(
                prediction_id=pred.prediction_id,
                sequence_id=pred.sequence_id,
                pdb_path=pred.pdb_path,
                plddt_overall=pred.plddt_overall,
                plddt_interface=pred.plddt_interface,
                pae_interface=pred.pae_interface,
                ptm=pred.ptm,
                iptm=pred.iptm,
                pae_interaction=pred.pae_interaction,
                ptm_binder=pred.ptm_binder,
                rmsd_to_design=pred.rmsd_to_design,
            )
            state_tree.end_timing(pred_node_id, NodeStatus.COMPLETED)
        
        # Set aggregate cost for adaptive stage
        state_tree.set_metrics(run_id, {
            "adaptive_mode": True,
            "adaptive_duration_sec": stage_duration,
        })
        
        validation_results.append(
            ValidationResult(
                stage=PipelineStage.BACKBONE_GENERATION,
                status=ValidationStatus.PASSED if backbones else ValidationStatus.FAILED,
                candidate_id=run_id,
                metrics={"num_backbones": len(backbones), "adaptive_mode": True, "duration_sec": stage_duration},
            )
        )
        validation_results.append(
            ValidationResult(
                stage=PipelineStage.SEQUENCE_DESIGN,
                status=ValidationStatus.PASSED if sequences else ValidationStatus.FAILED,
                candidate_id=run_id,
                metrics={"num_sequences": len(sequences)},
            )
        )
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
            return _create_empty_result(run_id, config, validation_results, start_time, state_tree)
    else:
        # Standard batch generation
        print("\n=== Phase 1: Backbone Generation (RFDiffusion) ===")
        print(f"[START] {time.strftime('%H:%M:%S')} | Generating {config.rfdiffusion.num_designs} backbones")
        stage_start = time.time()
        backbones = _run_backbone_generation(
            config.target,
            config.rfdiffusion,
            dirs["backbones"],
            use_mocks,
        )
        stage_duration = time.time() - stage_start
        print(f"[DONE]  {time.strftime('%H:%M:%S')} | Output: {len(backbones)} backbones | Duration: {stage_duration:.1f}s")
        stage_metrics.append(_log_stage_metrics(
            "rfdiffusion", stage_duration, len(backbones),
            cost_estimate["_timeouts"]["rfdiffusion"], cost_estimate["rfdiffusion"]
        ))
        
        # Add backbone nodes to state tree
        for backbone in backbones:
            backbone_node_id = state_tree.add_backbone(
                design_id=backbone.design_id,
                parent_id=target_node_id,
                pdb_path=backbone.pdb_path,
                binder_length=backbone.binder_length,
                rfdiffusion_score=backbone.rfdiffusion_score,
            )
            state_tree.end_timing(
                backbone_node_id,
                status=NodeStatus.COMPLETED,
                cost_usd=cost_estimate["rfdiffusion"] / max(1, len(backbones)),
            )
        
        # Step 1b: Backbone Quality Filter (#2 optimization)
        if config.backbone_filter.enabled and backbones:
            original_backbone_ids = {b.design_id for b in backbones}
            backbones = filter_backbones_by_quality(
                backbones,
                min_score=config.backbone_filter.min_score,
                max_keep=config.backbone_filter.max_keep,
            )
            # Mark filtered backbones in state tree
            kept_ids = {b.design_id for b in backbones}
            for bb_id in original_backbone_ids - kept_ids:
                state_tree.set_status(bb_id, NodeStatus.FILTERED)

        validation_results.append(
            ValidationResult(
                stage=PipelineStage.BACKBONE_GENERATION,
                status=ValidationStatus.PASSED if backbones else ValidationStatus.FAILED,
                candidate_id=run_id,
                metrics={"num_backbones": len(backbones)},
            )
        )

        if not backbones:
            return _create_empty_result(run_id, config, validation_results, start_time, state_tree)

        # Step 2: ProteinMPNN - Sequence Design
        print("\n=== Phase 1: Sequence Design (ProteinMPNN) ===")
        print(f"[START] {time.strftime('%H:%M:%S')} | Input: {len(backbones)} backbones × {config.proteinmpnn.num_sequences} seqs each")
        stage_start = time.time()
        sequences = _run_sequence_design(
            backbones,
            config.proteinmpnn,
            dirs["sequences"],
            use_mocks,
        )
        stage_duration = time.time() - stage_start
        print(f"[DONE]  {time.strftime('%H:%M:%S')} | Output: {len(sequences)} sequences | Duration: {stage_duration:.1f}s")
        stage_metrics.append(_log_stage_metrics(
            "proteinmpnn", stage_duration, len(sequences),
            cost_estimate["_timeouts"]["proteinmpnn"] * len(backbones), cost_estimate["proteinmpnn"]
        ))
        
        # Add sequence nodes to state tree
        for seq in sequences:
            seq_node_id = state_tree.add_sequence(
                sequence_id=seq.sequence_id,
                backbone_id=seq.backbone_id,
                sequence=seq.sequence,
                score=seq.score,
                fasta_path=seq.fasta_path,
            )
            state_tree.end_timing(
                seq_node_id,
                status=NodeStatus.COMPLETED,
                cost_usd=cost_estimate["proteinmpnn"] / max(1, len(sequences)),
            )
        
        predictions = None  # Will be computed in Phase 2
        
        validation_results.append(
            ValidationResult(
                stage=PipelineStage.SEQUENCE_DESIGN,
                status=ValidationStatus.PASSED if sequences else ValidationStatus.FAILED,
                candidate_id=run_id,
                metrics={"num_sequences": len(sequences)},
            )
        )

        if not sequences:
            return _create_empty_result(run_id, config, validation_results, start_time, state_tree)

        # =========================================================================
        # Phase 2: Validation (Specificity) - Standard Mode
        # =========================================================================

        # Step 3: Boltz-2 - Structure Prediction & Filtering
        # ENFORCE Boltz-2 limit: select TOP N sequences by ProteinMPNN score (descending)
        if len(sequences) > max_boltz2_validations:
            # Sort by ProteinMPNN score descending (higher score = better)
            sequences_ranked = sorted(sequences, key=lambda s: s.score, reverse=True)
            sequences_to_validate = sequences_ranked[:max_boltz2_validations]
            skipped_sequences = sequences_ranked[max_boltz2_validations:]
            top_scores = [f"{s.score:.2f}" for s in sequences_to_validate]
            print(f"[LIMIT] Boltz-2: {len(sequences)} → {max_boltz2_validations} sequences (top by score: {', '.join(top_scores)})")
            # Mark skipped sequences in state tree
            for seq in skipped_sequences:
                state_tree.set_status(seq.sequence_id, NodeStatus.SKIPPED)
                state_tree.set_metrics(seq.sequence_id, {"skipped_by": "boltz2_limit", "score": seq.score})
        else:
            sequences_to_validate = sequences
        
        print("\n=== Phase 2: Structure Validation (Boltz-2) ===")
        print(f"[START] {time.strftime('%H:%M:%S')} | Input: {len(sequences_to_validate)} sequences")
        stage_start = time.time()
        predictions = _run_structure_validation(
            sequences_to_validate,
            config.target,
            config.boltz2,
            dirs["validation_boltz"],
            use_mocks,
        )
        stage_duration = time.time() - stage_start
        print(f"[DONE]  {time.strftime('%H:%M:%S')} | Output: {len(predictions)} passed | Duration: {stage_duration:.1f}s")
        stage_metrics.append(_log_stage_metrics(
            "boltz2", stage_duration, len(predictions),
            cost_estimate["_timeouts"]["boltz2"] * len(sequences_to_validate), cost_estimate["boltz2"]
        ))
        
        # Add prediction nodes to state tree
        for pred in predictions:
            pred_node_id = state_tree.add_prediction(
                prediction_id=pred.prediction_id,
                sequence_id=pred.sequence_id,
                pdb_path=pred.pdb_path,
                plddt_overall=pred.plddt_overall,
                plddt_interface=pred.plddt_interface,
                pae_interface=pred.pae_interface,
                ptm=pred.ptm,
                iptm=pred.iptm,
                pae_interaction=pred.pae_interaction,
                ptm_binder=pred.ptm_binder,
                rmsd_to_design=pred.rmsd_to_design,
            )
            state_tree.end_timing(
                pred_node_id,
                status=NodeStatus.COMPLETED,
                cost_usd=cost_estimate["boltz2"] / max(1, len(predictions)),
            )
        
        # Mark sequences that failed validation as filtered
        validated_seq_ids = {p.sequence_id for p in predictions}
        for seq in sequences_to_validate:
            if seq.sequence_id not in validated_seq_ids:
                state_tree.set_status(seq.sequence_id, NodeStatus.FILTERED)

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
            return _create_empty_result(run_id, config, validation_results, start_time, state_tree)

    # Step 4: Clustering - Diversity via TM-score (AlphaProteo SI 2.2)
    if config.cluster.tm_threshold > 0:
        print("\n=== Phase 2b: Clustering (Diversity) ===")
        pre_cluster_ids = {p.prediction_id for p in predictions}
        predictions = cluster_by_tm_score(
            predictions,
            tm_threshold=config.cluster.tm_threshold,
            select_best=config.cluster.select_best,
        )
        post_cluster_ids = {p.prediction_id for p in predictions}
        # Mark clustered-out predictions as filtered
        for pred_id in pre_cluster_ids - post_cluster_ids:
            state_tree.set_status(pred_id, NodeStatus.FILTERED)
            state_tree.set_metrics(pred_id, {"filtered_by": "clustering"})
        print(f"After clustering: {len(predictions)} representatives")

    # Step 5: Novelty Check - pyhmmer vs UniRef50 (AlphaProteo SI 2.2)
    if config.novelty.enabled:
        print("\n=== Phase 2c: Novelty Check (pyhmmer) ===")
        # Get sequences corresponding to predictions
        pred_seq_ids = {p.sequence_id for p in predictions}
        pre_novelty_pred_ids = {p.prediction_id for p in predictions}
        novel_sequences = check_novelty(
            [s for s in sequences if s.sequence_id in pred_seq_ids],
            max_evalue=config.novelty.max_evalue,
            database=config.novelty.database,
        )
        # Filter predictions to only novel sequences
        novel_seq_ids = {s.sequence_id for s in novel_sequences}
        predictions = [p for p in predictions if p.sequence_id in novel_seq_ids]
        post_novelty_pred_ids = {p.prediction_id for p in predictions}
        # Mark non-novel predictions as filtered
        for pred_id in pre_novelty_pred_ids - post_novelty_pred_ids:
            state_tree.set_status(pred_id, NodeStatus.FILTERED)
            state_tree.set_metrics(pred_id, {"filtered_by": "novelty_check"})
        print(f"After novelty check: {len(predictions)} novel designs")
    
    if not predictions:
        print("No predictions passed clustering/novelty filters")
        return _create_empty_result(run_id, config, validation_results, start_time, state_tree)

    # =========================================================================
    # Phase 3: Negative Selection (Selectivity)
    # =========================================================================

    # Step 6: FoldSeek - Find Structural Decoys (with caching #5 optimization)
    print("\n=== Phase 3: Decoy Identification (FoldSeek) ===")
    print(f"[START] {time.strftime('%H:%M:%S')} | Searching for up to {config.foldseek.max_hits} decoys")
    stage_start = time.time()
    decoys = _run_decoy_search(
        config.target,
        config.foldseek,
        f"{campaign_dir}/_decoys",
        use_mocks,
    )
    stage_duration = time.time() - stage_start
    print(f"[DONE]  {time.strftime('%H:%M:%S')} | Output: {len(decoys)} decoys | Duration: {stage_duration:.1f}s")
    stage_metrics.append(_log_stage_metrics(
        "foldseek", stage_duration, len(decoys),
        cost_estimate["_timeouts"]["foldseek"], cost_estimate["foldseek"]
    ))
    
    # Add decoy nodes to state tree
    for decoy in decoys:
        decoy_node_id = state_tree.add_decoy(
            decoy_id=decoy.decoy_id,
            target_id=target_node_id,
            pdb_path=decoy.pdb_path,
            evalue=decoy.evalue,
            tm_score=decoy.tm_score,
            aligned_length=decoy.aligned_length,
            sequence_identity=decoy.sequence_identity,
        )
        state_tree.end_timing(
            decoy_node_id,
            status=NodeStatus.COMPLETED,
            cost_usd=cost_estimate["foldseek"] / max(1, len(decoys)),
        )

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
        # ENFORCE Chai-1 limit: select TOP N sequences by PPI score (descending)
        # Each sequence is checked against ALL decoys
        num_decoys = len(decoys) if decoys else 0
        
        if len(validated_sequences) > max_chai1_sequences:
            # Build mapping from sequence_id to PPI score from Boltz-2 predictions
            ppi_scores = {p.sequence_id: p.ppi_score for p in predictions}
            
            # Sort by PPI score descending (higher = better target binding)
            validated_sequences_ranked = sorted(
                validated_sequences,
                key=lambda s: ppi_scores.get(s.sequence_id, 0),
                reverse=True
            )
            validated_sequences = validated_sequences_ranked[:max_chai1_sequences]
            skipped_for_chai1 = validated_sequences_ranked[max_chai1_sequences:]
            top_ppi = [f"{ppi_scores.get(s.sequence_id, 0):.3f}" for s in validated_sequences]
            print(f"[LIMIT] Chai-1: {len(validated_sequences_ranked)} → {max_chai1_sequences} sequences (top by PPI: {', '.join(top_ppi)})")
            # Mark skipped sequences in state tree
            for seq in skipped_for_chai1:
                state_tree.set_metrics(seq.sequence_id, {
                    "skipped_chai1": True,
                    "ppi_score": ppi_scores.get(seq.sequence_id, 0),
                })
        
        print("\n=== Phase 3: Cross-Reactivity Check (Chai-1) ===")
        print(f"[START] {time.strftime('%H:%M:%S')} | Input: {len(validated_sequences)} sequences × {num_decoys} decoys")
        stage_start = time.time()
        
        # Run positive control + decoy check
        positive_control_results, cross_reactivity_results = _run_cross_reactivity_check(
            validated_sequences,
            decoys if decoys else [],
            config.chai1,
            dirs["validation_chai"],
            use_mocks,
            target=config.target,  # Include target for positive control
        )
        stage_duration = time.time() - stage_start
        
        # Report positive control results
        num_binding = sum(1 for r in positive_control_results.values() if r.plddt_interface > 50)
        print(f"[DONE]  {time.strftime('%H:%M:%S')} | Positive control: {num_binding}/{len(validated_sequences)} bind target | Duration: {stage_duration:.1f}s")
        
        if decoys:
            print(f"        Decoy check: {len(cross_reactivity_results)} sequences checked against {num_decoys} decoys")
        
        # Log Chai-1 metrics
        num_pairs = len(validated_sequences) * num_decoys if decoys else len(validated_sequences)
        stage_metrics.append(_log_stage_metrics(
            "chai1", stage_duration, num_pairs,
            cost_estimate["_timeouts"]["chai1"] * cost_estimate["_effective"]["chai1_pairs"], 
            cost_estimate["chai1"]
        ))
        
        # Add positive control results to state tree (target binding check)
        for seq_id, pc_result in positive_control_results.items():
            pc_node_id = state_tree.add_cross_reactivity(
                binder_id=seq_id,
                decoy_id="TARGET_POSITIVE_CONTROL",
                predicted_affinity=pc_result.predicted_affinity,
                plddt_interface=pc_result.plddt_interface,
                binds_decoy=pc_result.binds_decoy,
                ptm=pc_result.ptm,
                iptm=pc_result.iptm,
                chain_pair_iptm=pc_result.chain_pair_iptm,
            )
            state_tree.set_metrics(pc_node_id, {"is_positive_control": True})
            state_tree.end_timing(pc_node_id, status=NodeStatus.COMPLETED)
        
        # Add cross-reactivity nodes to state tree (decoy checks)
        for seq_id, cr_results in cross_reactivity_results.items():
            for cr in cr_results:
                cr_node_id = state_tree.add_cross_reactivity(
                    binder_id=seq_id,
                    decoy_id=cr.decoy_id,
                    predicted_affinity=cr.predicted_affinity,
                    plddt_interface=cr.plddt_interface,
                    binds_decoy=cr.binds_decoy,
                    ptm=cr.ptm,
                    iptm=cr.iptm,
                    chain_pair_iptm=cr.chain_pair_iptm,
                )
                state_tree.end_timing(
                    cr_node_id,
                    status=NodeStatus.COMPLETED,
                    cost_usd=cost_estimate["chai1"] / max(1, num_pairs),
                )

        validation_results.append(
            ValidationResult(
                stage=PipelineStage.CROSS_REACTIVITY,
                status=ValidationStatus.PASSED,
                candidate_id=run_id,
                metrics={
                    "num_checked": len(cross_reactivity_results),
                    "positive_control_binding": num_binding,
                    "duration_sec": stage_duration,
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
        
        # Add candidate node to state tree
        state_tree.add_candidate(
            candidate_id=candidate_id,
            prediction_id=prediction.prediction_id,
            specificity_score=specificity_score,
            selectivity_score=selectivity_score,
            final_score=final_score,
        )

    # Sort by final score (highest first)
    all_candidates.sort(key=lambda c: c.final_score, reverse=True)

    # Select top candidates
    top_n = min(10, len(all_candidates))
    top_candidates = all_candidates[:top_n]

    print(f"\nPipeline complete. {len(all_candidates)} candidates generated.")
    if top_candidates:
        print(f"Top candidate score: {top_candidates[0].final_score:.2f}")

    # Calculate actual runtime (wall-clock)
    runtime_seconds = time.time() - start_time
    
    # Finalize state tree first to aggregate all metrics
    state_tree.finalize(success=True)
    
    # Get actual metrics from state tree (single source of truth)
    actual_cost = state_tree.get_total_cost()
    actual_timing = state_tree.get_total_timing()
    timing_by_stage = state_tree.get_timing_by_stage()
    cost_by_stage = state_tree.get_cost_by_stage()
    ceiling_cost = cost_estimate["total"]
    
    # Print stage metrics summary (derived from state tree)
    print("\n=== Stage Metrics Summary (Actual vs Ceiling) ===")
    print(f"{'Stage':<15} {'Time (actual/ceil)':<20} {'Cost (actual/ceil)':<25}")
    print("-" * 60)
    
    for stage in ["rfdiffusion", "proteinmpnn", "boltz2", "foldseek", "chai1"]:
        stage_time = timing_by_stage.get(stage, 0.0)
        stage_cost = cost_by_stage.get(stage, 0.0)
        ceiling_time = cost_estimate["_timeouts"].get(stage, 0)
        # Calculate ceiling cost for this stage
        stage_ceiling = cost_estimate.get(stage, 0.0)
        
        time_str = f"{stage_time:.1f}s / {ceiling_time}s"
        cost_str = f"${stage_cost:.3f} / ${stage_ceiling:.3f}"
        print(f"{stage:<15} {time_str:<20} {cost_str:<25}")
    
    print("-" * 60)
    print(f"{'TOTAL':<15} {actual_timing:.1f}s / {cost_estimate['_runtime_sec']:.0f}s      ${actual_cost:.3f} / ${ceiling_cost:.3f}")
    
    savings = ceiling_cost - actual_cost
    if savings > 0:
        print(f"  Savings: ${savings:.3f} ({savings/ceiling_cost*100:.0f}% under ceiling)")
    
    print(f"\nWall-clock runtime: {runtime_seconds:.1f}s")
    
    # Add final run metrics to state tree
    state_tree.set_metrics(run_id, {
        "runtime_seconds": runtime_seconds,
        "total_candidates": len(all_candidates),
        "top_candidates_count": len(top_candidates),
        "best_score": top_candidates[0].final_score if top_candidates else None,
        "best_candidate_id": top_candidates[0].candidate_id if top_candidates else None,
        "ceiling_cost_usd": ceiling_cost,
        "savings_usd": savings if savings > 0 else 0,
        "savings_pct": (savings / ceiling_cost * 100) if ceiling_cost > 0 and savings > 0 else 0,
    })

    # Write inputs (info.json at PDB root, entity-specific config)
    _write_inputs(dirs["pdb_root"], dirs["entity"], config)

    # Write metrics CSV
    _write_metrics_csv(dirs["metrics"], all_candidates)

    # Write stage metrics (actual vs ceiling) - now uses state tree data
    _write_stage_metrics_from_tree(campaign_dir, state_tree, cost_estimate, runtime_seconds)

    # Create symlinks for best candidates
    _create_best_symlinks(dirs["best"], top_candidates, dirs["validation_boltz"])
    
    # Export state tree
    state_tree_path = f"{campaign_dir}/state_tree.json"
    state_tree.to_json(state_tree_path)
    print(f"\nState tree exported to: {state_tree_path}")
    print(state_tree.summary())
    
    # Also export Graphviz visualization
    graphviz_path = f"{campaign_dir}/state_tree.dot"
    state_tree.to_graphviz(graphviz_path)

    # Commit data volume
    data_volume.commit()

    return PipelineResult(
        run_id=run_id,
        config=config,
        candidates=all_candidates,
        top_candidates=top_candidates,
        validation_summary=validation_results,
        compute_cost_usd=actual_cost,  # Actual cost from state tree
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


def _run_adaptive_generation(
    target: TargetProtein,
    rfdiffusion_config: RFDiffusionConfig,
    proteinmpnn_config: ProteinMPNNConfig,
    boltz2_config: Boltz2Config,
    adaptive_config: AdaptiveGenerationConfig,
    backbone_filter_config: BackboneFilterConfig,
    dirs: dict[str, str],
    max_boltz2_per_batch: Optional[int] = None,
) -> tuple[list[BackboneDesign], list[SequenceDesign], list[StructurePrediction]]:
    """
    Adaptive generation with early termination using micro-batching (#3 optimization).
    
    Uses GPU-efficient micro-batching: generates a batch of backbones, processes
    all sequences in parallel, validates in parallel, then checks if threshold met.
    This preserves GPU throughput while enabling early termination between batches.
    
    Args:
        target: Target protein specification
        rfdiffusion_config: RFDiffusion configuration
        proteinmpnn_config: ProteinMPNN configuration
        boltz2_config: Boltz-2 configuration
        adaptive_config: Adaptive generation settings
        backbone_filter_config: Backbone quality filter settings
        dirs: Output directories
    
    Returns:
        Tuple of (backbones, sequences, validated_predictions)
    """
    all_backbones: list[BackboneDesign] = []
    all_sequences: list[SequenceDesign] = []
    validated_predictions: list[StructurePrediction] = []
    
    batch_num = 0
    
    print(f"Adaptive mode: target {adaptive_config.min_validated_candidates} validated candidates")
    print(f"  Micro-batch size: {adaptive_config.batch_size} backbones, max batches: {adaptive_config.max_batches}")
    print("  (GPU-efficient: full batch processing, early termination between batches)")
    
    while (len(validated_predictions) < adaptive_config.min_validated_candidates 
           and batch_num < adaptive_config.max_batches):
        
        batch_num += 1
        print(f"\n--- Micro-batch {batch_num}/{adaptive_config.max_batches} ---")
        
        # =====================================================================
        # Step 1: Generate batch of backbones (GPU-efficient batch processing)
        # =====================================================================
        batch_rfdiffusion_config = RFDiffusionConfig(
            num_designs=adaptive_config.batch_size,
            binder_length_min=rfdiffusion_config.binder_length_min,
            binder_length_max=rfdiffusion_config.binder_length_max,
            noise_scale=rfdiffusion_config.noise_scale,
            num_diffusion_steps=rfdiffusion_config.num_diffusion_steps,
        )
        
        batch_output_dir = f"{dirs['backbones']}/batch_{batch_num}"
        
        try:
            batch_backbones = run_rfdiffusion.remote(target, batch_rfdiffusion_config, batch_output_dir)
        except Exception as e:
            print(f"  RFDiffusion batch failed: {e}")
            continue
        
        if not batch_backbones:
            print(f"  No backbones generated in batch {batch_num}")
            continue
        
        print(f"  Step 1: Generated {len(batch_backbones)} backbones")
        
        # Apply backbone quality filter
        if backbone_filter_config.enabled:
            batch_backbones = filter_backbones_by_quality(
                batch_backbones,
                min_score=backbone_filter_config.min_score,
                max_keep=backbone_filter_config.max_keep,
            )
        
        if not batch_backbones:
            print(f"  All backbones filtered out in batch {batch_num}")
            continue
        
        all_backbones.extend(batch_backbones)
        
        # =====================================================================
        # Step 2: Design sequences for ALL backbones in parallel (GPU-efficient)
        # =====================================================================
        try:
            batch_sequences = generate_sequences_parallel.remote(
                batch_backbones, 
                proteinmpnn_config, 
                f"{dirs['sequences']}/batch_{batch_num}"
            )
        except Exception as e:
            print(f"  ProteinMPNN batch failed: {e}")
            continue
        
        if not batch_sequences:
            print(f"  No sequences designed in batch {batch_num}")
            continue
        
        print(f"  Step 2: Designed {len(batch_sequences)} sequences ({len(batch_sequences)//len(batch_backbones)} per backbone)")
        all_sequences.extend(batch_sequences)
        
        # =====================================================================
        # Step 3: Validate sequences (GPU-efficient, score-ranked selection)
        # =====================================================================
        # Apply per-batch limit with score ranking if specified
        if max_boltz2_per_batch and len(batch_sequences) > max_boltz2_per_batch:
            # Sort by ProteinMPNN score descending (higher = better)
            batch_sequences_ranked = sorted(batch_sequences, key=lambda s: s.score, reverse=True)
            sequences_for_validation = batch_sequences_ranked[:max_boltz2_per_batch]
            top_scores = [f"{s.score:.2f}" for s in sequences_for_validation]
            print(f"  [LIMIT] Boltz-2: {len(batch_sequences)} → {max_boltz2_per_batch} seqs (top by score: {', '.join(top_scores)})")
        else:
            sequences_for_validation = batch_sequences
        
        try:
            batch_predictions = validate_sequences_parallel.remote(
                sequences_for_validation,
                target,
                boltz2_config,
                f"{dirs['validation_boltz']}/batch_{batch_num}"
            )
        except Exception as e:
            print(f"  Boltz-2 batch failed: {e}")
            continue
        
        if batch_predictions:
            validated_predictions.extend(batch_predictions)
            print(f"  Step 3: Validated {len(batch_predictions)}/{len(sequences_for_validation)} sequences")
            print(f"  Running total: {len(validated_predictions)}/{adaptive_config.min_validated_candidates} validated candidates")
        else:
            print("  Step 3: No sequences passed validation in this batch")
        
        # =====================================================================
        # Check if we have enough validated candidates to stop
        # =====================================================================
        if len(validated_predictions) >= adaptive_config.min_validated_candidates:
            print(f"\n  ✓ Early termination: reached {len(validated_predictions)} validated candidates")
            break
    
    # Summary
    total_backbones = len(all_backbones)
    max_possible_backbones = adaptive_config.max_batches * adaptive_config.batch_size
    max_possible_sequences = max_possible_backbones * proteinmpnn_config.num_sequences
    
    print("\nAdaptive generation summary:")
    print(f"  Micro-batches used: {batch_num}/{adaptive_config.max_batches}")
    print(f"  Backbones generated: {total_backbones}")
    print(f"  Sequences designed: {len(all_sequences)}")
    print(f"  Validated candidates: {len(validated_predictions)}")
    
    if batch_num < adaptive_config.max_batches:
        saved_backbones = max_possible_backbones - total_backbones
        saved_sequences = max_possible_sequences - len(all_sequences)
        if saved_backbones > 0 or saved_sequences > 0:
            print("  Savings from early termination:")
            print(f"    Skipped: {saved_backbones} backbones, {saved_sequences} sequences")
            # Estimate cost savings (rough)
            backbone_cost_per = 600 * MODAL_GPU_COST_PER_SEC["A10G"]  # 600s timeout
            sequence_cost_per = 300 * MODAL_GPU_COST_PER_SEC["L4"] / proteinmpnn_config.num_sequences
            validation_cost_per = 900 * MODAL_GPU_COST_PER_SEC["A100"]
            saved_cost = (saved_backbones * backbone_cost_per + 
                         saved_sequences * (sequence_cost_per + validation_cost_per))
            print(f"    Est. cost saved: ${saved_cost:.2f}")
    
    return all_backbones, all_sequences, validated_predictions


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
    """Execute decoy search step with TTL caching via Modal Dict."""
    if use_mocks:
        return mock_foldseek(target, config, output_dir)

    try:
        # Check cache first if enabled (uses Modal Dict - no cache_dir needed)
        if config.cache_results:
            cached_decoys = get_cached_decoys(target)
            if cached_decoys is not None:
                # Verify PDB structures exist, re-download if missing
                valid_decoys = [d for d in cached_decoys if os.path.exists(d.pdb_path)]
                
                if len(valid_decoys) == len(cached_decoys):
                    return cached_decoys
                else:
                    # Re-download missing structures
                    cached_decoys = download_decoy_structures.remote(
                        cached_decoys, f"{output_dir}/structures"
                    )
                    return cached_decoys
        
        # Run FoldSeek
        decoys = run_foldseek.remote(target, config, output_dir)

        # Download actual PDB structures for valid hits
        if decoys:
            decoys = download_decoy_structures.remote(
                decoys, f"{output_dir}/structures"
            )
            
            # Cache results for future runs (TTL: 7 days, refreshed on read)
            if config.cache_results:
                save_decoys_to_cache(target, decoys)

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
    state_tree: Optional[PipelineStateTree] = None,
) -> PipelineResult:
    """Create an empty result when pipeline fails early."""
    # Finalize state tree if provided
    if state_tree is not None:
        state_tree.finalize(success=False)
    
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


def _write_run_manifest(
    campaign_dir: str, run_id: str, config: PipelineConfig, cost_estimate: dict
) -> None:
    """Write run_manifest.json with complete run configuration and cost breakdown."""
    import json
    from datetime import datetime

    t = config.target
    manifest = {
        "ulid": run_id,
        "timestamp": datetime.now().isoformat(),
        "target": {
            "pdb_id": t.pdb_id,
            "entity_id": t.entity_id,
            "name": t.name,
            "chain_id": t.chain_id,
            "hotspot_residues": t.hotspot_residues,
        },
        "configuration": {
            "mode": config.mode.value,
            "num_backbones": config.rfdiffusion.num_designs,
            "num_sequences_per_backbone": config.proteinmpnn.num_sequences,
            "total_sequences": config.rfdiffusion.num_designs * config.proteinmpnn.num_sequences,
            "max_decoys": config.foldseek.max_hits,
        },
        "thresholds": {
            "boltz2": {
                "max_pae_interaction": config.boltz2.max_pae_interaction,
                "min_ptm_binder": config.boltz2.min_ptm_binder,
                "max_rmsd": config.boltz2.max_rmsd,
            },
            "chai1": {
                "min_chain_pair_iptm": config.chai1.min_chain_pair_iptm,
            },
        },
        "cost_ceiling": {
            "total": cost_estimate["total"],
            "rfdiffusion": cost_estimate["rfdiffusion"],
            "proteinmpnn": cost_estimate["proteinmpnn"],
            "boltz2": cost_estimate["boltz2"],
            "foldseek": cost_estimate["foldseek"],
            "chai1": cost_estimate["chai1"],
        },
    }

    manifest_path = f"{campaign_dir}/run_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


def _write_stage_metrics(campaign_dir: str, stage_metrics: list[dict], runtime_seconds: float) -> None:
    """Write stage_metrics.json with actual vs ceiling costs for each stage (legacy)."""
    import json
    
    if not stage_metrics:
        return
    
    total_actual = sum(m["actual_cost_usd"] for m in stage_metrics)
    total_ceiling = sum(m["ceiling_cost_usd"] for m in stage_metrics)
    
    metrics = {
        "runtime_seconds": runtime_seconds,
        "total_actual_cost_usd": total_actual,
        "total_ceiling_cost_usd": total_ceiling,
        "savings_usd": total_ceiling - total_actual,
        "savings_pct": (total_ceiling - total_actual) / total_ceiling * 100 if total_ceiling > 0 else 0,
        "stages": stage_metrics,
    }
    
    metrics_path = f"{campaign_dir}/stage_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)


def _write_stage_metrics_from_tree(
    campaign_dir: str, 
    state_tree: PipelineStateTree, 
    cost_estimate: dict, 
    runtime_seconds: float
) -> None:
    """
    Write stage_metrics.json with actual vs ceiling costs derived from state tree.
    
    This version uses the state tree as the single source of truth for actual
    timing and cost metrics, ensuring consistency across all outputs.
    """
    import json
    
    timing_by_stage = state_tree.get_timing_by_stage()
    cost_by_stage = state_tree.get_cost_by_stage()
    
    stages = []
    for stage in ["rfdiffusion", "proteinmpnn", "boltz2", "foldseek", "chai1"]:
        stages.append({
            "stage": stage,
            "actual_timing_sec": timing_by_stage.get(stage, 0.0),
            "ceiling_timing_sec": cost_estimate["_timeouts"].get(stage, 0),
            "actual_cost_usd": cost_by_stage.get(stage, 0.0),
            "ceiling_cost_usd": cost_estimate.get(stage, 0.0),
        })
    
    metrics = {
        "runtime_seconds": runtime_seconds,
        "total_actual_timing_sec": state_tree.get_total_timing(),
        "total_ceiling_timing_sec": cost_estimate["_runtime_sec"],
        "total_actual_cost_usd": state_tree.get_total_cost(),
        "total_ceiling_cost_usd": cost_estimate["total"],
        "savings_usd": cost_estimate["total"] - state_tree.get_total_cost(),
        "savings_pct": (
            (cost_estimate["total"] - state_tree.get_total_cost()) / cost_estimate["total"] * 100
            if cost_estimate["total"] > 0 else 0
        ),
        "stages": stages,
    }
    
    metrics_path = f"{campaign_dir}/stage_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)


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
    Uses limit-based ceiling forecasting for deterministic cost estimates.
    """
    # Get accurate cost estimates with limit enforcement
    costs = estimate_cost(config)
    
    # Extract effective counts (after applying limits)
    eff = costs["_effective"]
    timeouts = costs["_timeouts"]
    
    # Requested vs effective (limited) values
    req_backbones = config.rfdiffusion.num_designs
    req_decoys = config.foldseek.max_hits
    
    eff_backbones = eff["backbones"]
    eff_seqs_per_backbone = eff["seqs_per_backbone"]
    eff_sequences = eff["sequences"]
    eff_boltz2 = eff["boltz2_validations"]
    eff_decoys = eff["decoys"]
    eff_chai1_seqs = eff["chai1_sequences"]
    eff_chai1 = eff["chai1_pairs"]
    
    # Calculate time breakdown
    rfdiffusion_sec = timeouts["rfdiffusion"]
    proteinmpnn_sec = timeouts["proteinmpnn"] * eff_backbones
    boltz2_sec = timeouts["boltz2"] * eff_boltz2
    foldseek_sec = timeouts["foldseek"]
    chai1_sec = timeouts["chai1"] * eff_chai1
    
    est_runtime_sec = costs["_runtime_sec"]
    est_runtime_min = est_runtime_sec / 60

    t = config.target
    limits = config.limits
    
    print()
    print("=" * 70)
    print("DRY RUN — DEPLOYMENT PREVIEW (Limit-Based Ceiling Forecast)")
    print("=" * 70)
    print()
    print("TARGET")
    print("-" * 70)
    print(f"Name: {t.name}")
    print(f"PDB ID: {t.pdb_id}")
    print(f"Entity ID: {t.entity_id}")
    print(f"Hotspot residues: {', '.join(str(r) for r in t.hotspot_residues)}")
    print()
    
    print("CONFIGURATION")
    print("-" * 70)
    
    # Show as tree structure
    backbone_limited = req_backbones > eff_backbones
    seq_per_bb_limited = config.proteinmpnn.num_sequences > eff_seqs_per_backbone
    decoy_limited = req_decoys > eff_decoys
    
    backbone_str = f"{req_backbones} → {eff_backbones}" if backbone_limited else str(eff_backbones)
    seq_per_bb_str = f"{config.proteinmpnn.num_sequences} → {eff_seqs_per_backbone}" if seq_per_bb_limited else str(eff_seqs_per_backbone)
    decoy_str = f"{req_decoys} → {eff_decoys}" if decoy_limited else str(eff_decoys)
    
    def cap_tag(capped: bool) -> str:
        return " (capped)" if capped else ""
    
    print(f"├─ Backbones: {backbone_str}{cap_tag(backbone_limited)}")
    print(f"│  └─ Sequences: {seq_per_bb_str}/backbone{cap_tag(seq_per_bb_limited)} → {eff_sequences} total")
    print(f"│     └─ Validations: {eff_boltz2}")
    print(f"│        └─ Chai-1: {eff_chai1_seqs} seqs × {eff_decoys} decoys = {eff_chai1} pairs")
    print(f"└─ Decoys: {decoy_str}{cap_tag(decoy_limited)}")
    print()
    
    print("STAGE LIMITS")
    print("-" * 70)
    print(f"{'Stage':<20} {'Timeout':<15} {'Max Designs':<15}")
    print("-" * 70)
    for stage in ["rfdiffusion", "proteinmpnn", "boltz2", "foldseek", "chai1"]:
        timeout = limits.get_timeout(stage)
        max_designs = limits.get_max_designs(stage)
        timeout_str = f"{timeout}s"
        print(f"{stage:<20} {timeout_str:<15} {max_designs:<15}")
    print()
    
    print("COST BREAKDOWN (Modal pricing, ceiling estimate)")
    print("-" * 70)
    print(f"{'Step':<20} {'GPU':<12} {'Runs':<10} {'Time':<12} {'Cost':>12}")
    print("-" * 70)
    cost_rfd = costs["rfdiffusion"]
    cost_mpnn = costs["proteinmpnn"]
    cost_boltz = costs["boltz2"]
    cost_fseek = costs["foldseek"]
    cost_chai = costs["chai1"]
    cost_total = costs["total"]
    print(f"{'RFDiffusion':<20} {'A10G':<12} {eff_backbones:<10} {rfdiffusion_sec:>6}s {'$' + f'{cost_rfd:.3f}':>12}")
    print(f"{'ProteinMPNN':<20} {'L4':<12} {eff_backbones:<10} {proteinmpnn_sec:>6}s {'$' + f'{cost_mpnn:.3f}':>12}")
    print(f"{'Boltz-2':<20} {'A100-80GB':<12} {eff_boltz2:<10} {boltz2_sec:>6}s {'$' + f'{cost_boltz:.3f}':>12}")
    print(f"{'FoldSeek':<20} {'CPU':<12} {1:<10} {foldseek_sec:>6}s {'$' + f'{cost_fseek:.3f}':>12}")
    print(f"{'Chai-1':<20} {'A100-80GB':<12} {eff_chai1:<10} {chai1_sec:>6}s {'$' + f'{cost_chai:.3f}':>12}")
    print("-" * 70)
    print(f"{'CEILING TOTAL':<20} {'':<12} {'':<10} {int(est_runtime_sec):>6}s {'$' + f'{cost_total:.2f}':>12}")
    print()
    
    if config.max_compute_usd is not None:
        budget_status = "✓ WITHIN BUDGET" if costs["total"] <= config.max_compute_usd else "✗ EXCEEDS BUDGET"
        print(f"Budget: ${config.max_compute_usd:.2f}  [{budget_status}]")
    else:
        print("Budget: unlimited")
    print(f"Max runtime: ~{est_runtime_min:.0f} minutes (sequential worst-case)")
    print()

    if config.max_compute_usd is not None and costs["total"] > config.max_compute_usd:
        scale_factor = config.max_compute_usd / costs["total"]
        suggested_designs = max(1, int(req_backbones * scale_factor))
        suggested_sequences = max(1, int(config.proteinmpnn.num_sequences * scale_factor))
        print("⚠ BUDGET WARNING")
        print("-" * 70)
        print("Options to reduce cost:")
        print(f"  1. Reduce designs: --num-designs {suggested_designs} --num-sequences {suggested_sequences}")
        print("  2. Lower stage limits in YAML config (limits.boltz2.max_designs, etc.)")
        print(f"  3. Increase budget: --max-budget {cost_total * 1.1:.2f}")
        print()

    print("To run: remove --dry-run flag")
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
    pdb_id: str = None,
    entity_id: int = None,
    hotspot_residues: str = None,
    config: str = None,
    mode: str = "bind",
    num_designs: int = 5,
    num_sequences: int = 4,
    use_mocks: bool = False,
    max_budget: float = None,
    dry_run: bool = False,
):
    """
    Run the protein binder design pipeline from command line.

    Args:
        pdb_id: 4-letter PDB code (e.g., "3DI3") - required unless --config is provided
        entity_id: Polymer entity ID for the target (e.g., 2 for IL7RA in 3DI3)
        hotspot_residues: Comma-separated list of hotspot residue indices
        config: Path to YAML configuration file (overrides other arguments)
        mode: Generation mode - "bind" for binder design (default: "bind")
        num_designs: Number of backbone designs (default: 5)
        num_sequences: Sequences per backbone (default: 4)
        use_mocks: Use mock implementations for testing (default: False)
        max_budget: Maximum compute budget in USD (default: None = no limit)
        dry_run: Preview deployment parameters and costs without running (default: False)
    
    Example with CLI args:
        uv run modal run pipeline.py --pdb-id 3DI3 --entity-id 2 --hotspot-residues "58,80,139" --mode bind
    
    Example with YAML config:
        uv run modal run pipeline.py --config config.yaml
        uv run modal run pipeline.py --config config.yaml --dry-run
    
    Generated proteins are named: <pdb_id>_E<entity_id>_<mode>_<ulid>
    Example: 3DI3_E2_bind_01ARZ3NDEKTSV4RRFFQ69G5FAV
    """
    from common import get_entity_info, download_pdb, load_config_from_yaml
    import tempfile
    import os
    
    # Load config from YAML if provided
    if config is not None:
        print(f"Loading configuration from: {config}")
        pipeline_config = load_config_from_yaml(config)
        
        # Extract target info for entity lookup
        t = pipeline_config.target
        pdb_id = t.pdb_id
        entity_id = t.entity_id
        
        # Fetch entity info from RCSB (local API call, no Modal)
        print(f"Fetching entity {entity_id} info for PDB {pdb_id}...")
        entity_info = get_entity_info(pdb_id, entity_id)
        chains = entity_info.get("chains", [])
        
        if not chains:
            raise ValueError(f"No chains found for entity {entity_id} in PDB {pdb_id}")
        
        chain_id = chains[0]
        entity_name = entity_info.get("description", f"{pdb_id}_entity{entity_id}")
        uniprot_ids = entity_info.get("uniprot_ids", [])
        
        print(f"  Entity: {entity_name}")
        print(f"  Chain(s): {', '.join(chains)}")
        if uniprot_ids:
            print(f"  UniProt: {', '.join(uniprot_ids)}")
        
        # Update target with resolved chain info
        pipeline_config.target.chain_id = chain_id
        pipeline_config.target.name = entity_name
        
        if dry_run:
            pipeline_config.target.pdb_path = f"/data/targets/{pdb_id.lower()}.pdb"  # Placeholder
            print_dry_run_summary(pipeline_config)
            return None
        
        # Download PDB and upload to Modal volume
        with tempfile.TemporaryDirectory() as tmpdir:
            local_pdb = os.path.join(tmpdir, f"{pdb_id.lower()}.pdb")
            print(f"Downloading PDB {pdb_id}...")
            download_pdb(pdb_id, local_pdb)
            
            with open(local_pdb, "r") as f:
                pdb_content = f.read()
            
            remote_pdb_path = upload_target_pdb.remote(pdb_content, f"{pdb_id.lower()}.pdb")
            print(f"Uploaded to: {remote_pdb_path}")
        
        pipeline_config.target.pdb_path = remote_pdb_path
        
        # Run pipeline with YAML config
        result = run_pipeline.remote(pipeline_config, use_mocks=use_mocks)
        
    else:
        # Require CLI arguments if no config file
        if pdb_id is None or entity_id is None or hotspot_residues is None:
            raise ValueError(
                "Either --config must be provided, or all of: --pdb-id, --entity-id, --hotspot-residues"
            )
        
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

        # Fetch entity info from RCSB (local API call, no Modal)
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

        # Dry run mode: print summary and exit (before any Modal calls)
        if dry_run:
            # Build config with placeholder path for cost estimation
            pipeline_config = PipelineConfig(
                target=TargetProtein(
                    pdb_id=pdb_id,
                    entity_id=entity_id,
                    hotspot_residues=hotspots,
                    pdb_path=f"/data/targets/{pdb_id.lower()}.pdb",  # Placeholder
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
            print_dry_run_summary(pipeline_config)
            return None

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
        pipeline_config = PipelineConfig(
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

        # Run pipeline
        result = run_pipeline.remote(pipeline_config, use_mocks=use_mocks)

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
    max_budget: Optional[float] = None,
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
        max_budget: Maximum compute budget in USD (None = no limit)
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
    # Standalone CLI that can run without Modal for dry-run mode
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(
        description="Protein binder design pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run with CLI args (no Modal, no GPU, just cost estimation):
  python pipeline.py --pdb-id 3DI3 --entity-id 2 --hotspot-residues 58,80,139 --dry-run

  # Dry run with YAML config:
  python pipeline.py --config config.yaml --dry-run

  # Full run (use 'modal run' instead):
  modal run pipeline.py --pdb-id 3DI3 --entity-id 2 --hotspot-residues 58,80,139
  modal run pipeline.py --config config.yaml
        """
    )
    parser.add_argument("--config", help="Path to YAML configuration file")
    parser.add_argument("--pdb-id", help="4-letter PDB code (e.g., 3DI3)")
    parser.add_argument("--entity-id", type=int, help="Polymer entity ID")
    parser.add_argument("--hotspot-residues", help="Comma-separated residue indices")
    parser.add_argument("--mode", default="bind", help="Generation mode (default: bind)")
    parser.add_argument("--num-designs", type=int, default=5, help="Number of backbones (default: 5)")
    parser.add_argument("--num-sequences", type=int, default=4, help="Sequences per backbone (default: 4)")
    parser.add_argument("--max-budget", type=float, default=None, help="Max budget in USD (default: no limit)")
    parser.add_argument("--dry-run", action="store_true", help="Preview costs without running")
    
    args = parser.parse_args()
    
    # Validate that we have either --config or the required CLI args
    if args.config is None and (args.pdb_id is None or args.entity_id is None or args.hotspot_residues is None):
        print("Error: Either --config must be provided, or all of: --pdb-id, --entity-id, --hotspot-residues")
        sys.exit(1)
    
    # For dry-run, we avoid importing Modal entirely
    if args.dry_run:
        # Import only what we need for cost estimation (no Modal)
        from common import get_entity_info, GenerationMode, load_config_from_yaml
        
        if args.config:
            # Load from YAML config
            print(f"Loading configuration from: {args.config}")
            try:
                config = load_config_from_yaml(args.config)
            except Exception as e:
                print(f"Error loading config: {e}")
                sys.exit(1)
            
            pdb_id = config.target.pdb_id
            entity_id = config.target.entity_id
            
            # Fetch entity info (local HTTP call, no Modal)
            print(f"Fetching entity {entity_id} info for PDB {pdb_id}...")
            try:
                entity_info = get_entity_info(pdb_id, entity_id)
            except Exception as e:
                print(f"Error fetching entity info: {e}")
                sys.exit(1)
            
            chains = entity_info.get("chains", [])
            if not chains:
                print(f"Error: No chains found for entity {entity_id} in PDB {pdb_id}")
                sys.exit(1)
            
            chain_id = chains[0]
            entity_name = entity_info.get("description", f"{pdb_id}_entity{entity_id}")
            uniprot_ids = entity_info.get("uniprot_ids", [])
            
            print(f"  Entity: {entity_name}")
            print(f"  Chain(s): {', '.join(chains)}")
            if uniprot_ids:
                print(f"  UniProt: {', '.join(uniprot_ids)}")
            
            # Update target with resolved chain info
            config.target.chain_id = chain_id
            config.target.name = entity_name
            config.target.pdb_path = f"/data/targets/{pdb_id.lower()}.pdb"  # Placeholder
            
        else:
            # Build config from CLI args
            pdb_id = args.pdb_id.upper().strip()
            if len(pdb_id) != 4:
                print(f"Error: PDB ID must be 4 characters: {pdb_id}")
                sys.exit(1)
            
            hotspots = [int(r.strip()) for r in args.hotspot_residues.split(",")]
            
            mode = args.mode.lower().strip()
            valid_modes = [m.value for m in GenerationMode]
            if mode not in valid_modes:
                print(f"Error: Invalid mode '{mode}'. Valid modes: {valid_modes}")
                sys.exit(1)
            generation_mode = GenerationMode(mode)
            
            # Fetch entity info (local HTTP call, no Modal)
            print(f"Fetching entity {args.entity_id} info for PDB {pdb_id}...")
            try:
                entity_info = get_entity_info(pdb_id, args.entity_id)
            except Exception as e:
                print(f"Error fetching entity info: {e}")
                sys.exit(1)
            
            chains = entity_info.get("chains", [])
            if not chains:
                print(f"Error: No chains found for entity {args.entity_id} in PDB {pdb_id}")
                sys.exit(1)
            
            chain_id = chains[0]
            entity_name = entity_info.get("description", f"{pdb_id}_entity{args.entity_id}")
            uniprot_ids = entity_info.get("uniprot_ids", [])
            
            print(f"  Entity: {entity_name}")
            print(f"  Chain(s): {', '.join(chains)}")
            if uniprot_ids:
                print(f"  UniProt: {', '.join(uniprot_ids)}")
            
            # Build config for cost estimation (import Pydantic models only)
            from common import (
                PipelineConfig, TargetProtein, RFDiffusionConfig,
                ProteinMPNNConfig, Boltz2Config, FoldSeekConfig, Chai1Config
            )
            
            config = PipelineConfig(
                target=TargetProtein(
                    pdb_id=pdb_id,
                    entity_id=args.entity_id,
                    hotspot_residues=hotspots,
                    pdb_path=f"/data/targets/{pdb_id.lower()}.pdb",  # Placeholder
                    chain_id=chain_id,
                    name=entity_name,
                ),
                mode=generation_mode,
                rfdiffusion=RFDiffusionConfig(num_designs=args.num_designs),
                proteinmpnn=ProteinMPNNConfig(num_sequences=args.num_sequences),
                boltz2=Boltz2Config(),
                foldseek=FoldSeekConfig(),
                chai1=Chai1Config(),
                max_compute_usd=args.max_budget,
            )
        
        # Print cost summary (local function, no Modal)
        print_dry_run_summary(config)
        sys.exit(0)
    
    else:
        # Non-dry-run: tell user to use modal run
        if args.config:
            print("Error: For actual pipeline runs, use 'modal run':")
            print(f"  modal run pipeline.py --config {args.config}")
            print("\nFor dry-run (cost estimation only):")
            print(f"  python pipeline.py --config {args.config} --dry-run")
        else:
            print("Error: For actual pipeline runs, use 'modal run':")
            print(f"  modal run pipeline.py --pdb-id {args.pdb_id} --entity-id {args.entity_id} \\")
            print(f"    --hotspot-residues \"{args.hotspot_residues}\"")
            print("\nFor dry-run (cost estimation only):")
            print(f"  python pipeline.py --pdb-id {args.pdb_id} --entity-id {args.entity_id} \\")
            print(f"    --hotspot-residues \"{args.hotspot_residues}\" --dry-run")
        sys.exit(1)
