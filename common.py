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
from pathlib import Path
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
    )
)


def _add_local_modules(image: modal.Image) -> modal.Image:
    """Add local Python modules to a Modal image for cross-module imports."""
    return (
        image
        .add_local_file("common.py", remote_path="/root/common.py")
        .add_local_file("generators.py", remote_path="/root/generators.py")
        .add_local_file("validators.py", remote_path="/root/validators.py")
    )


# Apply local modules to all images so functions can import each other
base_image = _add_local_modules(base_image)
rfdiffusion_image = _add_local_modules(rfdiffusion_image)
proteinmpnn_image = _add_local_modules(proteinmpnn_image)
boltz2_image = _add_local_modules(boltz2_image)
chai1_image = _add_local_modules(chai1_image)
foldseek_image = _add_local_modules(foldseek_image)


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
    entity_id: int = Field(..., ge=1, description="Polymer entity ID (e.g., 1, 2)")
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

    num_designs: int = Field(default=10, ge=1, le=100, description="Number of backbones to generate")
    binder_length_min: int = Field(default=50, ge=30, description="Minimum binder length")
    binder_length_max: int = Field(default=100, le=200, description="Maximum binder length")
    noise_scale: float = Field(default=1.0, ge=0.1, le=2.0, description="Diffusion noise scale")
    num_diffusion_steps: int = Field(default=50, ge=10, le=200, description="Number of diffusion steps")

    @property
    def contigmap(self) -> str:
        """Generate contigmap string for RFDiffusion."""
        return f"[{self.binder_length_min}-{self.binder_length_max}/0 ]"


class ProteinMPNNConfig(BaseModel):
    """Configuration for ProteinMPNN sequence design."""

    num_sequences: int = Field(default=8, ge=1, le=64, description="Sequences per backbone")
    temperature: float = Field(default=0.2, ge=0.01, le=1.0, description="Sampling temperature")
    backbone_noise: float = Field(default=0.0, ge=0.0, le=0.5, description="Backbone coordinate noise")


class Boltz2Config(BaseModel):
    """Configuration for Boltz-2 structure prediction.
    
    Thresholds based on AlphaProteo SI 2.2 (optimized AF3 metrics):
    - min_pae_interaction < 1.5 Å: Anchor lock at hotspots
    - ptm_binder > 0.80: Binder fold quality
    - rmsd < 2.5 Å: Self-consistency vs RFDiffusion
    """

    num_recycles: int = Field(default=3, ge=1, le=10, description="Number of recycling iterations")
    
    # AlphaProteo thresholds (SI 2.2)
    max_pae_interaction: float = Field(default=1.5, ge=0.0, le=31.0, description="Max PAE at hotspots (Anchor Lock)")
    min_ptm_binder: float = Field(default=0.80, ge=0.0, le=1.0, description="Min pTM for binder-only (Fold Quality)")
    max_rmsd: float = Field(default=2.5, ge=0.0, le=10.0, description="Max RMSD vs RFDiffusion backbone (Self-Consistency)")
    
    # Legacy thresholds (still applied)
    min_iplddt: float = Field(default=0.8, ge=0.0, le=1.0, description="Minimum interface pLDDT threshold")
    max_pae: float = Field(default=5.0, ge=0.0, le=31.0, description="Maximum overall PAE threshold")


class FoldSeekConfig(BaseModel):
    """Configuration for FoldSeek proteome scanning.
    
    Note: Fewer decoys (5) reduces expensive Chai-1 calls while still catching top risks.
    """

    database: str = Field(default="pdb100", description="Database to search (pdb100, afdb50, etc.)")
    max_hits: int = Field(default=5, ge=1, le=100, description="Maximum decoys (fewer = cheaper)")
    evalue_threshold: float = Field(default=1e-3, ge=0.0, description="E-value cutoff")


class Chai1Config(BaseModel):
    """Configuration for Chai-1 cross-reactivity check (single-sequence mode)."""

    num_samples: int = Field(default=1, ge=1, le=5, description="Samples per binder-decoy pair")
    min_chain_pair_iptm: float = Field(default=0.5, ge=0.0, le=1.0, description="chain_pair_iptm threshold for cross-reactivity")


class ClusterConfig(BaseModel):
    """Configuration for TM-score based clustering (diversity)."""

    tm_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="TM-score threshold for clustering")
    select_best: bool = Field(default=True, description="Select best representative per cluster")


class NoveltyConfig(BaseModel):
    """Configuration for novelty check via pyhmmer vs UniRef50."""

    database: str = Field(default="uniref50", description="HMM database for novelty check")
    max_evalue: float = Field(default=1e-5, ge=0.0, description="Max E-value to consider a hit (lower = stricter)")
    enabled: bool = Field(default=True, description="Enable novelty filtering")


class ScoringWeights(BaseModel):
    """Weights for the scoring function S(x)."""

    alpha: float = Field(default=1.0, ge=0.0, description="Weight for interface pLDDT (specificity)")
    beta: float = Field(default=0.5, ge=0.0, description="Weight for max decoy affinity (selectivity penalty)")


class PipelineConfig(BaseModel):
    """Complete pipeline configuration."""

    target: TargetProtein
    mode: GenerationMode = Field(default=GenerationMode.BIND, description="Semantic generation mode")
    rfdiffusion: RFDiffusionConfig = Field(default_factory=RFDiffusionConfig)
    proteinmpnn: ProteinMPNNConfig = Field(default_factory=ProteinMPNNConfig)
    boltz2: Boltz2Config = Field(default_factory=Boltz2Config)
    foldseek: FoldSeekConfig = Field(default_factory=FoldSeekConfig)
    chai1: Chai1Config = Field(default_factory=Chai1Config)
    cluster: ClusterConfig = Field(default_factory=ClusterConfig, description="TM-score clustering for diversity")
    novelty: NoveltyConfig = Field(default_factory=NoveltyConfig, description="Novelty check vs UniRef50")
    scoring: ScoringWeights = Field(default_factory=ScoringWeights)

    # Budget control
    max_compute_usd: float = Field(default=5.0, ge=0.1, le=10.0, description="Maximum compute budget in USD")


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
    rfdiffusion_score: Optional[float] = Field(None, description="RFDiffusion confidence")


class SequenceDesign(BaseModel):
    """Output from ProteinMPNN sequence design."""

    sequence_id: str = Field(..., description="Unique identifier for this sequence")
    backbone_id: str = Field(..., description="Parent backbone design ID")
    sequence: str = Field(..., description="Amino acid sequence (one-letter codes)")
    fasta_path: str = Field(..., description="Path to FASTA file")
    score: float = Field(..., description="ProteinMPNN log-likelihood score")
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
    plddt_overall: float = Field(..., ge=0.0, le=100.0, description="Overall pLDDT score")
    plddt_interface: float = Field(..., ge=0.0, le=100.0, description="Interface pLDDT (i-pLDDT)")
    pae_interface: float = Field(..., ge=0.0, description="Interface PAE score")
    ptm: Optional[float] = Field(None, ge=0.0, le=1.0, description="pTM score (complex)")
    iptm: Optional[float] = Field(None, ge=0.0, le=1.0, description="ipTM score")
    
    # AlphaProteo SI 2.2 metrics
    pae_interaction: Optional[float] = Field(None, ge=0.0, description="Min PAE at hotspot residues (Anchor Lock)")
    ptm_binder: Optional[float] = Field(None, ge=0.0, le=1.0, description="pTM for binder-only (Fold Quality)")
    rmsd_to_design: Optional[float] = Field(None, ge=0.0, description="RMSD vs RFDiffusion backbone (Self-Consistency)")

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
    tm_score: float = Field(..., ge=0.0, le=1.0, description="TM-score alignment")
    aligned_length: int = Field(..., description="Number of aligned residues")
    sequence_identity: float = Field(..., ge=0.0, le=1.0, description="Sequence identity in alignment")


class CrossReactivityResult(BaseModel):
    """Output from Chai-1 cross-reactivity check (single-sequence mode)."""

    binder_id: str = Field(..., description="Binder sequence ID")
    decoy_id: str = Field(..., description="Decoy protein ID")
    predicted_affinity: float = Field(..., description="Predicted binding affinity (lower = tighter)")
    plddt_interface: float = Field(..., ge=0.0, le=100.0, description="Interface pLDDT with decoy")
    binds_decoy: bool = Field(..., description="Whether binder likely binds decoy")
    ptm: Optional[float] = Field(None, ge=0.0, le=1.0, description="pTM score")
    iptm: Optional[float] = Field(None, ge=0.0, le=1.0, description="ipTM score")
    chain_pair_iptm: Optional[float] = Field(None, ge=0.0, le=1.0, description="Chain-pair ipTM (off-target threshold: >0.5)")

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
    specificity_score: float = Field(..., description="Target binding score (higher = better)")
    selectivity_score: float = Field(..., description="Off-target avoidance (higher = better)")
    final_score: float = Field(..., description="Combined S(x) score")

    def compute_final_score(self, alpha: float = 1.0, beta: float = 0.5) -> float:
        """
        Compute the selection function S(x).
        
        S(x) = α * pLDDT_interface(x) - β * max_{d ∈ D}(Affinity(x, d))
        """
        max_decoy_affinity = 0.0
        if self.decoy_results:
            max_decoy_affinity = max(r.predicted_affinity for r in self.decoy_results)

        return alpha * self.specificity_score - beta * max_decoy_affinity


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
