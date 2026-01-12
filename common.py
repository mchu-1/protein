"""
common.py - Pydantic models and Modal image definitions for protein binder design pipeline.

This module defines:
- Data schemas for pipeline I/O (BinderCandidate, ValidationResult, etc.)
- Modal image configurations with pre-installed dependencies
- Shared volumes for model weights
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Optional

import modal
from pydantic import BaseModel, Field, field_validator

# =============================================================================
# Modal Configuration
# =============================================================================

APP_NAME = "protein-binder-pipeline"

# Shared volumes for model weights (pre-downloaded, not at runtime)
weights_volume = modal.Volume.from_name("binder-weights", create_if_missing=True)
data_volume = modal.Volume.from_name("binder-data", create_if_missing=True)

# TTL key-value cache for FoldSeek decoy results (7-day TTL, refreshed on read)
# Uses Modal Dict for cloud-native caching with automatic LRU-like expiry
foldseek_cache = modal.Dict.from_name("foldseek-decoy-cache", create_if_missing=True)

# Cache for 3Di structural fingerprints (structural memoization)
# Key: PDB ID + entity ID, Value: {backbone_hash: 3di_fingerprint}
structural_cache = modal.Dict.from_name("structural-3di-cache", create_if_missing=True)

WEIGHTS_PATH = "/weights"
DATA_PATH = "/data"

# Single app instance shared across all modules
app = modal.App(APP_NAME)

# =============================================================================
# Base Modal Images
# =============================================================================

# Base image with common scientific Python dependencies
base_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "biopython>=1.81",
        "pydantic>=2.0.0",
        "scipy>=1.11.0",
        "networkx>=3.0",  # State tree graph representation & beam pruning
        "peptides>=0.3.2",  # Solubility filtering (net charge, pI)
        "biotite>=1.0.0",  # SAP stickiness calculation
        "pyhmmer>=0.10.0",  # Novelty check vs UniRef50 (phmmer mode)
        "pyyaml>=6.0",      # For loading config.yaml
    )
)

# RFDiffusion image
# The official rosettacommons/rfdiffusion image uses Python 3.9, but Modal requires 3.10+.
# Solution: Use from_registry with add_python to overlay 3.11, then reinstall dependencies
# under the new Python interpreter so they're available at runtime.
# Note: The container has a venv at /app/RFdiffusion/.venv that we must deactivate.
rfdiffusion_image = (
    modal.Image.from_registry(
        "rosettacommons/rfdiffusion:latest",
        add_python="3.11",  # Overlay Python 3.11 for Modal compatibility
    )
    .entrypoint([])  # Clear the container's ENTRYPOINT to avoid conflicts with Modal
    # Deactivate venv BEFORE pip installs by setting PATH to use system Python
    .env({
        "VIRTUAL_ENV": "",
        "PATH": "/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:/usr/sbin:/sbin",
        "DGLBACKEND": "pytorch",
    })
    # Reinstall RFDiffusion's core dependencies under Python 3.11
    # These are already in the container but installed for Python 3.9
    .pip_install(
        "torch==2.1.0",  # Pin torch for DGL/CUDA compatibility
        "numpy>=1.24.0,<2.0.0",  # RFDiffusion requires numpy <2
        "scipy>=1.11.0",
        "biopython>=1.81",
        "pydantic>=2.0.0",
        "hydra-core>=1.3.0",
        "omegaconf>=2.3.0",
        "e3nn",
        "opt_einsum",
        "icecream",
        "pyrsistent",  # Required by RFDiffusion symmetry module
        "pyyaml>=6.0",  # Required for common.py config loading
    )
    # Install DGL with CUDA 12.1 support (must match torch CUDA version)
    .run_commands(
        "pip install dgl -f https://data.dgl.ai/wheels/torch-2.1/cu121/repo.html"
    )
    # Install se3-transformer and rfdiffusion from the container's source
    .run_commands(
        "pip install /app/RFdiffusion/env/SE3Transformer",
        "pip install -e /app/RFdiffusion",
    )
)

# ProteinMPNN image
# Downloads ProteinMPNN from GitHub and installs dependencies
PROTEINMPNN_PATH = "/app/ProteinMPNN"
proteinmpnn_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "git-lfs", "wget")
    .pip_install(
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "biopython>=1.81",
        "pydantic>=2.0.0",
        "pyyaml>=6.0",  # Required for common.py config loading
    )
    # Clone ProteinMPNN repository (includes model weights)
    .run_commands(
        "git lfs install",
        f"git clone https://github.com/dauparas/ProteinMPNN.git {PROTEINMPNN_PATH}",
        f"ls -la {PROTEINMPNN_PATH}/vanilla_model_weights/",  # Verify weights exist
    )
    .env({"PROTEINMPNN_PATH": PROTEINMPNN_PATH})
)

# Boltz-2 image with structure prediction dependencies
# Uses the boltz CLI for structure prediction
# Based on Modal's working example: https://modal.com/docs/examples/boltz1
boltz2_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "wget", "build-essential")
    .pip_install(
        "boltz==0.4.0",  # Use stable version compatible with Modal
        "biopython>=1.81",
        "pydantic>=2.0.0",
        "biotite>=1.0.0",  # For robust RMSD calculation
        "scipy>=1.11.0",   # Required by biotite
        "pyyaml>=6.0",      # Required for common.py config loading
    )
)

# Chai-1 image for docking/structure prediction
# chai_lab requires specific torch version with CUDA and additional dependencies
chai1_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget", "build-essential")
    .pip_install(
        "numpy>=1.24.0,<2.0.0",
        "scipy>=1.11.0",
        "biopython>=1.81",
        "pydantic>=2.0.0",
        "einops>=0.7.0",
        "modelcif",  # For CIF file handling
        "gemmi",  # Molecular structure library
        "pyyaml>=6.0",  # Required for common.py config loading
        index_url="https://pypi.org/simple",
    )
    .run_commands(
        # Install PyTorch with CUDA support
        "pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu121",
        # Install chai_lab after torch
        "pip install chai_lab",
    )
    .env({
        # Suppress pandera deprecation warning from chai_lab dependencies
        "DISABLE_PANDERA_IMPORT_WARNING": "True",
        # Reduce verbosity of progress bars
        "TQDM_DISABLE": "1",
    })
)

# FoldSeek image for proteome scanning
foldseek_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("wget", "tar", "curl")
    .run_commands(
        "wget https://mmseqs.com/foldseek/foldseek-linux-avx2.tar.gz",
        "tar xzf foldseek-linux-avx2.tar.gz",
        "mv foldseek/bin/foldseek /usr/local/bin/",
        "rm -rf foldseek foldseek-linux-avx2.tar.gz",
    )
    .pip_install(
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "biopython>=1.81",
        "pydantic>=2.0.0",
        "pyyaml>=6.0",  # Required for common.py config loading
    )
)

# ESMFold image for orthogonal validation
# Uses ESM-2 protein language model to predict structure from sequence
# Provides architectural diversity vs. RFDiffusion/Boltz-2 (diffusion models)
esmfold_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget", "build-essential")
    # Install PyTorch FIRST (transformers requires >= 2.1)
    .pip_install(
        "torch>=2.1.0",
        "numpy>=1.24.0,<2.0.0",
    )
    # Then install other dependencies
    .pip_install(
        "scipy>=1.11.0",
        "biotite>=1.0.0",  # For fast RMSD calculations
        "pydantic>=2.0.0",
        "transformers>=4.30.0",  # For ESMFold via HuggingFace
        "fair-esm",  # Facebook ESM library (includes ESMFold)
        "pyyaml>=6.0",  # Required for common.py config loading
    )
    # Pre-download ESMFold model weights to bake into image (avoids 15GB download on every run)
    .run_commands(
        "python -c 'from transformers import EsmForProteinFolding; "
        "model = EsmForProteinFolding.from_pretrained(\"facebook/esmfold_v1\"); "
        "print(\"ESMFold weights cached successfully\")'",
    )
    .env({
        "TRANSFORMERS_CACHE": "/root/.cache/huggingface",
        "HF_HOME": "/root/.cache/huggingface",
    })
)


def _add_local_modules(image: modal.Image) -> modal.Image:
    """Add local Python modules to a Modal image for cross-module imports."""
    return (
        image
        .add_local_file("common.py", remote_path="/root/common.py")
        .add_local_file("generators.py", remote_path="/root/generators.py")
        .add_local_file("validators.py", remote_path="/root/validators.py")
        .add_local_file("state_tree.py", remote_path="/root/state_tree.py")
        .add_local_file("optimizers.py", remote_path="/root/optimizers.py")
        .add_local_file("config.yaml", remote_path="/root/config.yaml")
    )


# Apply local modules to all images so functions can import each other
base_image = _add_local_modules(base_image)
rfdiffusion_image = _add_local_modules(rfdiffusion_image)
proteinmpnn_image = _add_local_modules(proteinmpnn_image)
boltz2_image = _add_local_modules(boltz2_image)
chai1_image = _add_local_modules(chai1_image)
foldseek_image = _add_local_modules(foldseek_image)
esmfold_image = _add_local_modules(esmfold_image)


# =============================================================================
# Enums
# =============================================================================


class PipelineStage(str, Enum):
    """Stages in the binder design pipeline."""

    BACKBONE_GENERATION = "backbone_generation"
    SEQUENCE_DESIGN = "sequence_design"
    STRUCTURE_VALIDATION = "structure_validation"
    DECOY_SEARCH = "decoy_search"
    CROSS_REACTIVITY = "cross_reactivity"


class ValidationStatus(str, Enum):
    """Status of validation checks."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


class GenerationMode(str, Enum):
    """Semantic generation modes for protein design."""

    BIND = "bind"  # Design binders to target


# =============================================================================
# Pydantic Models - Input Schemas
# =============================================================================


class TargetProtein(BaseModel):
    """Input target protein specification.
    
    Uses PDB ID + Entity ID for unambiguous target identification.
    Entity ID maps to specific polymer chains within the PDB structure.
    """

    pdb_id: str = Field(..., description="4-letter PDB code (e.g., '3DI3')")
    entity_id: int = Field(..., description="Polymer entity ID (e.g., 1, 2)")
    hotspot_residues: list[int] = Field(
        default_factory=list,
        description="Residue indices defining the binding interface",
    )
    
    # Computed after download/initialization
    pdb_path: Optional[str] = Field(default=None, description="Path to downloaded PDB file")
    chain_id: Optional[str] = Field(default=None, description="Chain ID mapped from entity")
    name: Optional[str] = Field(default=None, description="Human-readable name")

    @field_validator("pdb_id")
    @classmethod
    def validate_pdb_id(cls, v: str) -> str:
        v = v.upper().strip()
        if len(v) != 4:
            raise ValueError(f"PDB ID must be 4 characters: {v}")
        return v

    @field_validator("hotspot_residues")
    @classmethod
    def validate_hotspots(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError("At least one hotspot residue must be specified")
        if any(r < 1 for r in v):
            raise ValueError("Residue indices must be >= 1")
        return sorted(set(v))


class RFDiffusionConfig(BaseModel):
    """Configuration for RFDiffusion backbone generation."""

    num_designs: int = Field(default=10, description="Number of backbones to generate")
    binder_length_min: int = Field(default=50, description="Minimum binder length")
    binder_length_max: int = Field(default=100, description="Maximum binder length")
    noise_scale: float = Field(default=1.0, description="Diffusion noise scale")
    num_diffusion_steps: int = Field(default=50, description="Number of diffusion steps")

    @property
    def contigmap(self) -> str:
        """Generate contigmap string for RFDiffusion."""
        return f"[{self.binder_length_min}-{self.binder_length_max}/0 ]"


class ProteinMPNNConfig(BaseModel):
    """Configuration for ProteinMPNN sequence design."""

    num_sequences: int = Field(default=8, description="Sequences per backbone")
    temperature: float = Field(default=0.2, description="Sampling temperature")
    backbone_noise: float = Field(default=0.0, description="Backbone coordinate noise")


class ESMFoldConfig(BaseModel):
    """Configuration for ESMFold orthogonal validation (gatekeeper).
    
    ESMFold provides an orthogonal architectural signal to detect hallucinations:
    - RFDiffusion/Boltz-2: Diffusion-based, geometry-driven
    - ESMFold: Protein language model, evolutionary data (UniRef)
    
    If ESMFold (sequence-only) cannot recover the RFDiffusion backbone,
    the design is likely an adversarial example.
    """
    
    enabled: bool = Field(default=True, description="Enable ESMFold gatekeeper")
    min_plddt: float = Field(default=70.0, description="Min mean pLDDT (confidence)")
    max_rmsd: float = Field(default=2.5, description="Max RMSD to RFDiffusion backbone (consistency)")
    save_predictions: bool = Field(default=False, description="Save ESMFold PDB predictions")


class Boltz2Config(BaseModel):
    """Configuration for Boltz-2 structure prediction.
    
    Thresholds based on AlphaProteo SI 2.2 (optimized AF3 metrics):
    1. min_pae_interaction < 1.5 Å: Anchor lock at hotspots
    2. ptm_binder > 0.80: Binder fold quality
    3. rmsd < 2.5 Å: Self-consistency vs RFDiffusion
    """

    num_recycles: int = Field(default=3, description="Number of recycling iterations")
    
    # AlphaProteo thresholds (SI 2.2)
    max_pae_interaction: float = Field(default=1.5, description="Max PAE at hotspots (Anchor Lock)")
    min_ptm_binder: float = Field(default=0.80, description="Min pTM for binder-only (Fold Quality)")
    max_rmsd: float = Field(default=2.5, description="Max RMSD vs RFDiffusion backbone (Self-Consistency)")


class FoldSeekConfig(BaseModel):
    """Configuration for FoldSeek proteome scanning.
    
    Note: Fewer decoys (5) reduces expensive Chai-1 calls while still catching top risks.
    """

    database: str = Field(default="pdb100", description="Database to search (pdb100, afdb50, etc.)")
    max_hits: int = Field(default=5, description="Maximum decoys (fewer = cheaper)")
    evalue_threshold: float = Field(default=1e-3, description="E-value cutoff")
    cache_results: bool = Field(default=True, description="Cache decoy results by target (saves repeated FoldSeek calls)")


class Chai1Config(BaseModel):
    """Configuration for Chai-1 cross-reactivity check (single-sequence mode)."""

    num_samples: int = Field(default=1, description="Samples per binder-decoy pair")
    min_chain_pair_iptm: float = Field(default=0.5, description="chain_pair_iptm threshold for cross-reactivity")
    num_recycles: int = Field(default=3, description="Number of recycling iterations")
    tiered_checking: bool = Field(default=True, description="Use tiered decoy checking (Tier 1/2/3) for cost savings")
    tier1_min_tm: float = Field(default=0.7, description="Min TM-score for Tier 1 (highest risk)")
    tier2_min_tm: float = Field(default=0.5, description="Min TM-score for Tier 2")
    tier2_max_decoys: int = Field(default=2, description="Max decoys in Tier 2")


class ClusterConfig(BaseModel):
    """Configuration for TM-score based clustering (diversity)."""

    tm_threshold: float = Field(default=0.7, description="TM-score threshold for clustering")
    select_best: bool = Field(default=True, description="Select best representative per cluster")


class NoveltyConfig(BaseModel):
    """Configuration for novelty check via pyhmmer vs UniRef50.
    
    Uses UniRef50 sequence database to filter sequences with high similarity to
    existing proteins. Critical for:
    - Patentability (IP): Avoid sequences too similar to known/patented proteins
    - Safety (Immunogenicity): Flag sequences similar to human proteins
    
    UniRef50 is a ~12GB compressed / ~50GB uncompressed FASTA database.
    It is persisted in a Modal Volume and downloaded only once.
    """

    max_evalue: float = Field(default=1e-6, description="Max E-value to consider a hit (lower = stricter)")
    enabled: bool = Field(default=True, description="Enable novelty filtering")
    auto_download: bool = Field(default=True, description="Auto-download UniRef50 database if not present")


class ScoringWeights(BaseModel):
    """Weights for the scoring function S(x)."""

    alpha: float = Field(default=1.0, description="Weight for interface pLDDT (specificity)")
    beta: float = Field(default=0.5, description="Weight for max decoy affinity (selectivity penalty)")


class BackboneFilterConfig(BaseModel):
    """Configuration for backbone quality pre-screening (#2 optimization).
    
    Filters low-quality RFDiffusion backbones before ProteinMPNN to save downstream costs.
    """

    enabled: bool = Field(default=True, description="Enable backbone quality filtering")
    min_score: float = Field(default=0.4, description="Min RFDiffusion confidence score")
    max_keep: Optional[int] = Field(default=None, description="Max backbones to keep (None = no limit)")


class AdaptiveGenerationConfig(BaseModel):
    """Configuration for adaptive generation with early termination (#3 optimization).
    
    Uses micro-batching to preserve GPU throughput while enabling early termination.
    Generates backbones in batches, validates all, then decides whether to continue.
    
    Stops generation early when enough high-quality candidates are found,
    saving significant compute on RFDiffusion, ProteinMPNN, and Boltz-2.
    """

    enabled: bool = Field(default=True, description="Enable adaptive generation")
    min_validated_candidates: int = Field(default=3, description="Stop when this many pass Boltz-2")
    batch_size: int = Field(default=4, description="Backbones to generate per micro-batch (GPU-efficient)")


class SolubilityFilterConfig(BaseModel):
    """Configuration for lookahead solubility filtering (Post-ProteinMPNN).
    
    Uses peptides.py to check net charge and isoelectric point (pI) to ensure
    good solubility of generated sequences before expensive structure prediction.
    
    Charge gates: Uses absolute charge to ensure minimum repulsion (prevents clumping)
    while rejecting super-charged sequences that won't fold properly.
    
    pI gate: Forbids the "dead zone" near physiological pH (6.0-8.0) where proteins
    have minimal net charge and tend to aggregate.
    
    Economics: Prunes ~20-40% of sequences that would likely aggregate in solution,
    saving Boltz-2 and Chai-1 compute costs.
    """

    enabled: bool = Field(default=True, description="Enable solubility pre-screening")
    min_abs_charge_ph7: float = Field(default=3.0, description="Minimum |net charge| at pH 7 (repulsion to prevent clumping)")
    max_abs_charge_ph7: float = Field(default=12.0, description="Maximum |net charge| at pH 7 (reject unfoldable super-charged)")
    forbidden_pi_min: float = Field(default=6.0, description="Lower bound of forbidden pI zone (dead zone near pH 7.4)")
    forbidden_pi_max: float = Field(default=8.0, description="Upper bound of forbidden pI zone (dead zone near pH 7.4)")


class StructuralMemoizationConfig(BaseModel):
    """Configuration for structural memoization via 3Di hashing (Post-RFDiffusion).
    
    Uses mini3di (ported from Foldseek) to generate structural fingerprints.
    Detects 'structural twins' from different random seeds and skips redundant
    sequence/folding computations.
    
    Economics: Can skip 10-30% of redundant backbone processing when generating
    many designs from the same target.
    """

    enabled: bool = Field(default=True, description="Enable structural memoization")
    similarity_threshold: float = Field(default=0.9, description="3Di similarity threshold to consider as 'twin'")
    cache_ttl_hours: int = Field(default=168, description="TTL for structural cache (default: 7 days)")


class BeamPruningConfig(BaseModel):
    """Configuration for greedy beam pruning (Throughout pipeline).
    
    Uses NetworkX-based dynamic tree pruning with beam search to limit fan-out.
    Ensures 1 backbone only sprouts N sequences instead of unlimited, keeping
    total tree nodes under control.
    
    Economics: Critical for cost control. Without pruning, costs scale O(n³).
    """

    enabled: bool = Field(default=True, description="Enable beam pruning")
    max_tree_nodes: int = Field(default=500, description="Maximum total nodes in state tree")
    sequences_per_backbone: int = Field(default=5, description="Max sequences to spawn per backbone")
    predictions_per_sequence: int = Field(default=1, description="Max predictions per sequence (usually 1)")
    prune_by_score: bool = Field(default=True, description="Prune lowest-scoring siblings when limit hit")


class StickinessFilterConfig(BaseModel):
    """Configuration for SAP (Spatial Aggregation Propensity) stickiness check.
    
    Uses biotite.structure.CellList for efficient spatial queries to compute
    surface-exposed hydrophobic patches. Prunes binders that are generally
    'sticky' and will aggregate in a test tube.
    
    Economics: Filters out ~10-20% of sequences that would fail wet-lab validation
    before expensive affinity checks (Chai-1).
    """

    enabled: bool = Field(default=True, description="Enable SAP stickiness filtering")
    max_sap_score: float = Field(default=0.50, description="Maximum SAP score. Relaxed to 0.50 to allow CDR-like loops.")
    hydrophobic_residues: str = Field(default="AVILMFYW", description="Residues considered hydrophobic")


class HardwareConfig(BaseModel):
    """Configuration for hardware resources (GPUs) for cost optimization.
    
    Allows selecting optimal GPU types based on model size and budget.
    Note: Changing these requires updating the code to read them dynamically, 
    or they serve as documentation for the hardcoded values if dynamic allocation is limited.
    """
    
    rfdiffusion_gpu: str = Field(default="A10G", description="GPU for RFDiffusion")
    proteinmpnn_gpu: str = Field(default="L4", description="GPU for ProteinMPNN")
    esmfold_gpu: str = Field(default="T4", description="GPU for ESMFold")
    boltz2_gpu: str = Field(default="A100", description="GPU for Boltz-2 (A100=40GB, A100-80GB=80GB)")
    chai1_gpu: str = Field(default="A100", description="GPU for Chai-1 (A100=40GB, A100-80GB=80GB)")



# =============================================================================
# Stage Limits Configuration
# =============================================================================


class StageLimits(BaseModel):
    """Limits for a single pipeline stage.
    
    Used to enforce timeouts per stage.
    If not specified, defaults are used.
    """
    
    timeout_seconds: Optional[int] = Field(
        default=None, 
        description="Max seconds for this stage (None = use default)"
    )


class PipelineLimits(BaseModel):
    """Limits for all pipeline stages.
    
    Enforces timeouts per stage.
    Defaults are used if not specified in YAML config.
    
    Default timeouts are based on Modal @app.function decorators.
    """
    
    # Stage-specific limits
    rfdiffusion: StageLimits = Field(default_factory=StageLimits)
    proteinmpnn: StageLimits = Field(default_factory=StageLimits)
    esmfold: StageLimits = Field(default_factory=StageLimits)
    boltz2: StageLimits = Field(default_factory=StageLimits)
    foldseek: StageLimits = Field(default_factory=StageLimits)
    chai1: StageLimits = Field(default_factory=StageLimits)
    
    # Default timeout values (seconds) - from Modal @app.function decorators
    # Kept relaxed to avoid premature termination of valid runs
    _default_timeouts: dict = {
        "rfdiffusion": 600,   # 10 min for batch generation
        "proteinmpnn": 300,   # 5 min per backbone
        "esmfold": 300,       # 5 min per sequence (gatekeeper)
        "boltz2": 900,        # 15 min per sequence
        "foldseek": 120,      # 2 min for proteome search
        "chai1": 900,         # 15 min per binder-decoy pair
    }
    
    def get_timeout(self, stage: str) -> int:
        """Get timeout for a stage, using default if not specified."""
        stage_limits = getattr(self, stage, StageLimits())
        if stage_limits.timeout_seconds is not None:
            return stage_limits.timeout_seconds
        return self._default_timeouts.get(stage, 600)


class PipelineConfig(BaseModel):
    """Complete pipeline configuration."""

    target: TargetProtein
    mode: GenerationMode = Field(default=GenerationMode.BIND, description="Semantic generation mode")
    rfdiffusion: RFDiffusionConfig = Field(default_factory=RFDiffusionConfig)
    proteinmpnn: ProteinMPNNConfig = Field(default_factory=ProteinMPNNConfig)
    esmfold: ESMFoldConfig = Field(default_factory=ESMFoldConfig, description="ESMFold orthogonal validation (gatekeeper)")
    boltz2: Boltz2Config = Field(default_factory=Boltz2Config)
    foldseek: FoldSeekConfig = Field(default_factory=FoldSeekConfig)
    chai1: Chai1Config = Field(default_factory=Chai1Config)
    cluster: ClusterConfig = Field(default_factory=ClusterConfig, description="TM-score clustering for diversity")
    novelty: NoveltyConfig = Field(default_factory=NoveltyConfig, description="Novelty check vs UniRef50")
    scoring: ScoringWeights = Field(default_factory=ScoringWeights)

    # Hardware configuration
    hardware: HardwareConfig = Field(default_factory=HardwareConfig, description="GPU hardware selection")

    # Cost optimization configs
    backbone_filter: BackboneFilterConfig = Field(default_factory=BackboneFilterConfig, description="Backbone quality pre-screening")
    adaptive: AdaptiveGenerationConfig = Field(default_factory=AdaptiveGenerationConfig, description="Adaptive generation with early termination")
    
    # New optimizations based on state tree
    solubility_filter: SolubilityFilterConfig = Field(default_factory=SolubilityFilterConfig, description="Lookahead solubility filtering")
    structural_memoization: StructuralMemoizationConfig = Field(default_factory=StructuralMemoizationConfig, description="3Di structural hashing for deduplication")
    beam_pruning: BeamPruningConfig = Field(default_factory=BeamPruningConfig, description="Greedy beam pruning to limit tree size")
    stickiness_filter: StickinessFilterConfig = Field(default_factory=StickinessFilterConfig, description="SAP stickiness check")

    # Stage limits (timeouts per stage)
    limits: PipelineLimits = Field(default_factory=PipelineLimits, description="Per-stage timeout limits")


def _load_global_config() -> PipelineConfig:
    """
    Load configuration globally for use in Modal decorators.
    
    This is required because @app.function decorators are evaluated at definition time,
    so we need access to the config (GPUs, timeouts) before the pipeline actually runs.
    """
    import yaml
    
    # Try finding config.yaml in common locations
    candidates = ["config.yaml", "/root/config.yaml"]
    config_path = None
    
    for path in candidates:
        if os.path.exists(path):
            config_path = path
            break
            
    if not config_path:
        # Fallback to defaults if file not found (e.g. during some remote builds)
        print("Warning: config.yaml not found, using defaults")
        # Create a valid dummy target to satisfy validation
        dummy_target = TargetProtein(pdb_id="XXXX", entity_id=1)
        return PipelineConfig(target=dummy_target)
        
    try:
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)
        return PipelineConfig(**data)
    except Exception as e:
        print(f"Error loading global config: {e}")
        # Return defaults on error to prevent import crash
        dummy_target = TargetProtein(pdb_id="XXXX", entity_id=1)
        return PipelineConfig(target=dummy_target)

# Export global config for decorators
GLOBAL_CONFIG = _load_global_config()


# =============================================================================
# Pydantic Models - Intermediate/Output Schemas
# =============================================================================


class BackboneDesign(BaseModel):
    """Output from RFDiffusion backbone generation."""

    design_id: str = Field(..., description="Unique identifier for this design")
    pdb_path: str = Field(..., description="Path to generated backbone PDB")
    target_pdb_path: str = Field(..., description="Path to target PDB used")
    hotspot_residues: list[int] = Field(..., description="Hotspot residues used")
    binder_length: int = Field(..., description="Length of designed binder")
    binder_chain_id: str = Field(default="B", description="Chain ID of the designed binder")
    rfdiffusion_score: Optional[float] = Field(None, description="RFDiffusion confidence")


class SequenceDesign(BaseModel):
    """Output from ProteinMPNN sequence design."""

    sequence_id: str = Field(..., description="Unique identifier for this sequence")
    backbone_id: str = Field(..., description="Parent backbone design ID")
    sequence: str = Field(..., description="Amino acid sequence (one-letter codes)")
    fasta_path: str = Field(..., description="Path to FASTA file")
    score: float = Field(..., description="ProteinMPNN log-likelihood score")
    binder_chain_id: str = Field(default="B", description="Chain ID of the designed binder")
    recovery: Optional[float] = Field(None, description="Sequence recovery if applicable")
    backbone_pdb: Optional[str] = Field(None, description="Path to RFDiffusion backbone PDB (for RMSD)")

    @field_validator("sequence")
    @classmethod
    def validate_sequence(cls, v: str) -> str:
        valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
        if not all(aa in valid_aa for aa in v.upper()):
            raise ValueError(f"Invalid amino acid in sequence: {v}")
        return v.upper()


class StructurePrediction(BaseModel):
    """Output from Boltz-2 structure prediction.
    
    Includes AlphaProteo SI 2.2 metrics for on-target scoring.
    """

    prediction_id: str = Field(..., description="Unique identifier")
    sequence_id: str = Field(..., description="Parent sequence design ID")
    pdb_path: str = Field(..., description="Path to predicted complex PDB")
    
    # Standard metrics
    plddt_overall: float = Field(..., description="Overall pLDDT score")
    plddt_interface: float = Field(..., description="Interface pLDDT (i-pLDDT)")
    pae_interface: float = Field(..., description="Interface PAE score")
    ptm: Optional[float] = Field(None, description="pTM score (complex)")
    iptm: Optional[float] = Field(None, description="ipTM score")
    
    # AlphaProteo SI 2.2 metrics
    pae_interaction: Optional[float] = Field(None, description="Min PAE at hotspot residues (Anchor Lock)")
    ptm_binder: Optional[float] = Field(None, description="pTM for binder-only (Fold Quality)")
    rmsd_to_design: Optional[float] = Field(None, description="RMSD vs RFDiffusion backbone (Self-Consistency)")
    binder_chain_id: str = Field(default="B", description="Chain ID of the designed binder")

    @property
    def ppi_score(self) -> float:
        """
        Quantification of protein-protein interaction quality.
        
        Uses weighted average: 0.8 * ipTM + 0.2 * pTM
        
        Returns:
            PPI score in range [0, 1], or 0.0 if metrics unavailable.
        """
        if self.iptm is not None and self.ptm is not None:
            return 0.8 * self.iptm + 0.2 * self.ptm
        elif self.iptm is not None:
            return self.iptm
        elif self.ptm is not None:
            return self.ptm
        return 0.0


class DecoyHit(BaseModel):
    """A structural homolog (potential off-target) found by FoldSeek."""

    decoy_id: str = Field(..., description="Identifier (PDB ID or UniProt)")
    pdb_path: str = Field(..., description="Path to decoy structure")
    evalue: float = Field(..., description="E-value from FoldSeek")
    tm_score: float = Field(..., description="TM-score alignment")
    aligned_length: int = Field(..., description="Number of aligned residues")
    sequence_identity: float = Field(..., description="Sequence identity in alignment")


class CrossReactivityResult(BaseModel):
    """Output from Chai-1 cross-reactivity check (single-sequence mode)."""

    binder_id: str = Field(..., description="Binder sequence ID")
    decoy_id: str = Field(..., description="Decoy protein ID")
    predicted_affinity: float = Field(..., description="Predicted binding affinity (lower = tighter)")
    plddt_interface: float = Field(..., description="Interface pLDDT with decoy")
    binds_decoy: bool = Field(..., description="Whether binder likely binds decoy")
    ptm: Optional[float] = Field(None, description="pTM score")
    iptm: Optional[float] = Field(None, description="ipTM score")
    chain_pair_iptm: Optional[float] = Field(None, description="Chain-pair ipTM (off-target threshold: >0.5)")

    @property
    def ppi_score(self) -> float:
        """
        Quantification of protein-protein interaction quality.
        
        Uses weighted average: 0.8 * ipTM + 0.2 * pTM
        
        Returns:
            PPI score in range [0, 1], or 0.0 if metrics unavailable.
        """
        if self.iptm is not None and self.ptm is not None:
            return 0.8 * self.iptm + 0.2 * self.ptm
        elif self.iptm is not None:
            return self.iptm
        elif self.ptm is not None:
            return self.ptm
        return 0.0


class BinderCandidate(BaseModel):
    """A complete binder candidate with all validation results."""

    candidate_id: str = Field(..., description="Unique candidate identifier")
    sequence: str = Field(..., description="Final amino acid sequence")
    backbone_design: BackboneDesign = Field(..., description="Backbone generation result")
    sequence_design: SequenceDesign = Field(..., description="Sequence design result")
    structure_prediction: StructurePrediction = Field(..., description="Structure validation result")
    decoy_results: list[CrossReactivityResult] = Field(
        default_factory=list, description="Cross-reactivity checks"
    )

    # Computed scores
    specificity_score: float = Field(..., description="Target binding score (higher = better, Boltz-2)")
    chai1_specificity_score: Optional[float] = Field(None, description="Target binding score (higher = better, Chai-1)")
    selectivity_score: float = Field(..., description="Off-target avoidance (higher = better)")
    final_score: float = Field(..., description="Combined S(x) score")

    def compute_final_score(self, alpha: float = 1.0, beta: float = 0.5) -> float:
        """
        Compute the selection function S(x).
        
        S(x) = α * MIN(pLDDT_interface_Boltz, pLDDT_interface_Chai) - β * max_{d ∈ D}(Affinity(x, d))
        
        Uses the minimum specificity score between Boltz-2 and Chai-1 (if available)
        to be conservative and avoid hallucinations from a single model.
        """
        # Specificity: Use min of Boltz-2 and Chai-1 (if available)
        # This provides a "consensus" confidence check
        spec = self.specificity_score
        if self.chai1_specificity_score is not None and self.chai1_specificity_score > 0:
            spec = min(self.specificity_score, self.chai1_specificity_score)

        max_decoy_affinity = 0.0
        if self.decoy_results:
            max_decoy_affinity = max(r.predicted_affinity for r in self.decoy_results)

        return alpha * spec - beta * max_decoy_affinity


class ValidationResult(BaseModel):
    """Result of a validation step."""

    stage: PipelineStage = Field(..., description="Pipeline stage")
    status: ValidationStatus = Field(..., description="Validation status")
    candidate_id: str = Field(..., description="Candidate being validated")
    metrics: dict = Field(default_factory=dict, description="Stage-specific metrics")
    error_message: Optional[str] = Field(None, description="Error message if failed")


class PipelineResult(BaseModel):
    """Final output of the complete pipeline."""

    run_id: str = Field(..., description="Unique run identifier")
    config: PipelineConfig = Field(..., description="Configuration used")
    candidates: list[BinderCandidate] = Field(..., description="All validated candidates")
    top_candidates: list[BinderCandidate] = Field(..., description="Top-ranked candidates")
    validation_summary: list[ValidationResult] = Field(..., description="All validation results")
    compute_cost_usd: float = Field(..., description="Estimated compute cost")
    runtime_seconds: float = Field(..., description="Total runtime")

    @property
    def best_candidate(self) -> Optional[BinderCandidate]:
        """Return the highest-scoring candidate."""
        if not self.top_candidates:
            return None
        return max(self.top_candidates, key=lambda c: c.final_score)


# =============================================================================
# YAML Configuration Loading
# =============================================================================


def load_config_from_yaml(yaml_path: str) -> PipelineConfig:
    """
    Load pipeline configuration from a YAML file.
    
    The YAML file should contain a complete or partial pipeline configuration.
    Missing fields use default values.
    
    Args:
        yaml_path: Path to the YAML configuration file
        
    Returns:
        PipelineConfig object
        
    Raises:
        FileNotFoundError: If the YAML file doesn't exist
        ValueError: If the YAML is malformed or contains invalid values
    """
    import yaml
    
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Config file not found: {yaml_path}")
    
    with open(yaml_path, "r") as f:
        raw_config = yaml.safe_load(f)
    
    if raw_config is None:
        raise ValueError(f"Empty or invalid YAML file: {yaml_path}")
    
    # Handle mode enum conversion
    if "mode" in raw_config and isinstance(raw_config["mode"], str):
        raw_config["mode"] = GenerationMode(raw_config["mode"].lower())
    
    # Validate and create PipelineConfig
    try:
        config = PipelineConfig(**raw_config)
        return config
    except Exception as e:
        raise ValueError(f"Invalid configuration in {yaml_path}: {e}")


def save_config_to_yaml(config: PipelineConfig, yaml_path: str) -> None:
    """
    Save a pipeline configuration to a YAML file.
    
    Args:
        config: PipelineConfig object to save
        yaml_path: Path to save the YAML file
    """
    import yaml
    
    # Convert to dict, handling enums
    config_dict = config.model_dump(mode="json")
    
    os.makedirs(os.path.dirname(yaml_path) or ".", exist_ok=True)
    with open(yaml_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


# =============================================================================
# Utility Functions
# =============================================================================


# Crockford's Base32 alphabet (excludes I, L, O, U to avoid ambiguity)
_ULID_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def generate_ulid() -> str:
    """
    Generate a ULID (Universally Unique Lexicographically Sortable Identifier).
    
    Format: 10 chars timestamp (ms) + 16 chars randomness = 26 chars total
    Uses Crockford's Base32 encoding.
    
    Returns:
        26-character ULID string (e.g., "01ARZ3NDEKTSV4RRFFQ69G5FAV")
    """
    import time
    import random
    
    # Timestamp: milliseconds since Unix epoch (48 bits -> 10 chars in base32)
    timestamp_ms = int(time.time() * 1000)
    
    # Encode timestamp (10 characters)
    timestamp_chars = []
    for _ in range(10):
        timestamp_chars.append(_ULID_ALPHABET[timestamp_ms & 0x1F])
        timestamp_ms >>= 5
    timestamp_str = "".join(reversed(timestamp_chars))
    
    # Randomness: 80 bits -> 16 chars in base32
    random_chars = []
    for _ in range(16):
        random_chars.append(_ULID_ALPHABET[random.randint(0, 31)])
    random_str = "".join(random_chars)
    
    return timestamp_str + random_str


def generate_protein_id(pdb_id: str, entity_id: int, mode: GenerationMode) -> str:
    """
    Generate a unique identifier for a designed protein.
    
    Format: <pdb_id>_E<entity_id>_<mode>_<ulid>
    Example: 3DI3_E2_bind_01ARZ3NDEKTSV4RRFFQ69G5FAV
    
    Args:
        pdb_id: Source PDB ID (e.g., "3DI3")
        entity_id: Entity number (e.g., 2)
        mode: Generation mode (e.g., GenerationMode.BIND)
    
    Returns:
        Unique protein identifier
    """
    ulid = generate_ulid()
    return f"{pdb_id.upper()}_E{entity_id}_{mode.value}_{ulid}"


def generate_design_id(prefix: str = "design") -> str:
    """Generate a unique design identifier (legacy format)."""
    import uuid

    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def parse_pdb_sequence(pdb_path: str, chain_id: str = "A") -> str:
    """Extract amino acid sequence from a PDB file."""
    from Bio.PDB import PDBParser
    from Bio.SeqUtils import seq1

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)

    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                residues = [res for res in chain if res.id[0] == " "]
                return "".join(seq1(res.resname) for res in residues)

    raise ValueError(f"Chain {chain_id} not found in {pdb_path}")


def write_fasta(sequence: str, header: str, output_path: str) -> None:
    """Write a sequence to a FASTA file."""
    with open(output_path, "w") as f:
        f.write(f">{header}\n")
        # Write sequence in lines of 80 characters
        for i in range(0, len(sequence), 80):
            f.write(sequence[i : i + 80] + "\n")


# =============================================================================
# PDB/Entity Mapping Functions
# =============================================================================


def get_entity_chain_mapping(pdb_id: str) -> dict[int, list[str]]:
    """
    Get mapping from entity ID to chain IDs for a PDB entry.
    
    Uses RCSB PDB API to look up entity-chain relationships.
    
    Args:
        pdb_id: 4-letter PDB code (e.g., "3DI3")
    
    Returns:
        Dictionary mapping entity_id -> list of chain_ids
        e.g., {1: ["A", "C"], 2: ["B", "D"]}
    """
    import json
    import urllib.request
    import urllib.error
    
    entity_chains: dict[int, list[str]] = {}
    
    try:
        url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id.upper()}"
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read().decode())
            
            container = data.get("rcsb_entry_container_identifiers", {})
            polymer_entity_ids = container.get("polymer_entity_ids", [])
            
            for entity_id in polymer_entity_ids:
                entity_url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id.upper()}/{entity_id}"
                try:
                    with urllib.request.urlopen(entity_url, timeout=10) as entity_response:
                        entity_data = json.loads(entity_response.read().decode())
                        
                        entity_container = entity_data.get("rcsb_polymer_entity_container_identifiers", {})
                        auth_asym_ids = entity_container.get("auth_asym_ids", [])
                        
                        if auth_asym_ids:
                            entity_chains[int(entity_id)] = auth_asym_ids
                except Exception:
                    continue
                    
    except urllib.error.HTTPError as e:
        raise ValueError(f"Could not fetch entity mapping for PDB {pdb_id}: HTTP {e.code}")
    except Exception as e:
        raise ValueError(f"Could not fetch entity mapping for PDB {pdb_id}: {e}")
    
    return entity_chains


def get_entity_info(pdb_id: str, entity_id: int) -> dict:
    """
    Get information about a specific entity in a PDB entry.
    
    Args:
        pdb_id: 4-letter PDB code
        entity_id: Polymer entity ID
    
    Returns:
        Dictionary with entity metadata (name, UniProt IDs, etc.)
    """
    import json
    import urllib.request
    
    try:
        url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id.upper()}/{entity_id}"
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read().decode())
            
            info = {
                "entity_id": entity_id,
                "chains": data.get("rcsb_polymer_entity_container_identifiers", {}).get("auth_asym_ids", []),
                "description": data.get("rcsb_polymer_entity", {}).get("pdbx_description", ""),
                "uniprot_ids": data.get("rcsb_polymer_entity_container_identifiers", {}).get("uniprot_ids", []),
            }
            return info
    except Exception as e:
        raise ValueError(f"Could not fetch entity {entity_id} info for PDB {pdb_id}: {e}")


def download_pdb(pdb_id: str, output_path: str) -> str:
    """
    Download a PDB file from RCSB.
    
    Args:
        pdb_id: 4-letter PDB code
        output_path: Path to save the PDB file
    
    Returns:
        Path to the downloaded file
    """
    import urllib.request
    
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    urllib.request.urlretrieve(url, output_path)
    return output_path


def initialize_target(
    pdb_id: str,
    entity_id: int,
    hotspot_residues: list[int],
    output_dir: str,
) -> TargetProtein:
    """
    Initialize a TargetProtein by downloading the PDB and resolving entity to chain.
    
    Args:
        pdb_id: 4-letter PDB code (e.g., "3DI3")
        entity_id: Polymer entity ID (e.g., 2 for the receptor)
        hotspot_residues: List of hotspot residue indices
        output_dir: Directory to save downloaded PDB
    
    Returns:
        Fully initialized TargetProtein with pdb_path and chain_id set
    """
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get entity info
    entity_info = get_entity_info(pdb_id, entity_id)
    chains = entity_info.get("chains", [])
    
    if not chains:
        raise ValueError(f"No chains found for entity {entity_id} in PDB {pdb_id}")
    
    # Use first chain for this entity
    chain_id = chains[0]
    
    # Download PDB
    pdb_path = os.path.join(output_dir, f"{pdb_id.lower()}.pdb")
    download_pdb(pdb_id, pdb_path)
    
    # Create target
    target = TargetProtein(
        pdb_id=pdb_id,
        entity_id=entity_id,
        hotspot_residues=hotspot_residues,
        pdb_path=pdb_path,
        chain_id=chain_id,
        name=entity_info.get("description", f"{pdb_id}_entity{entity_id}"),
    )
    
    return target
