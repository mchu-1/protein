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

# RFDiffusion image with PyTorch and SE3 dependencies
rfdiffusion_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "wget", "build-essential")
    .pip_install(
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "biopython>=1.81",
        "pydantic>=2.0.0",
        "hydra-core>=1.3.0",
        "omegaconf>=2.3.0",
        "icecream>=2.1.0",
        "e3nn>=0.5.0",
        "wandb>=0.15.0",
        "opt_einsum>=3.3.0",
    )
    .run_commands(
        "pip install git+https://github.com/RosettaCommons/RFdiffusion.git@main"
    )
)

# ProteinMPNN image (lighter weight)
proteinmpnn_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install(
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "biopython>=1.81",
        "pydantic>=2.0.0",
    )
)

# Boltz-2 image with structure prediction dependencies
boltz2_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "wget", "build-essential")
    .pip_install(
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "biopython>=1.81",
        "pydantic>=2.0.0",
        "einops>=0.7.0",
        "fair-esm>=2.0.0",
    )
)

# Chai-1 image for docking
chai1_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "wget", "build-essential")
    .pip_install(
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "biopython>=1.81",
        "pydantic>=2.0.0",
        "einops>=0.7.0",
    )
)

# FoldSeek image for proteome scanning
foldseek_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("wget", "tar")
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


# =============================================================================
# Pydantic Models - Input Schemas
# =============================================================================


class TargetProtein(BaseModel):
    """Input target protein specification."""

    pdb_path: str = Field(..., description="Path to target PDB structure file")
    chain_id: str = Field(default="A", description="Chain ID of interest")
    hotspot_residues: list[int] = Field(
        default_factory=list,
        description="Residue indices defining the binding interface",
    )
    name: str = Field(default="target", description="Human-readable name for target")

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
    """Configuration for Boltz-2 structure prediction."""

    num_recycles: int = Field(default=3, ge=1, le=10, description="Number of recycling iterations")
    min_iplddt: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum interface pLDDT threshold")
    max_pae: float = Field(default=10.0, ge=0.0, le=31.0, description="Maximum PAE threshold")


class FoldSeekConfig(BaseModel):
    """Configuration for FoldSeek proteome scanning."""

    database: str = Field(default="pdb100", description="Database to search (pdb100, afdb50, etc.)")
    max_hits: int = Field(default=10, ge=1, le=100, description="Maximum number of structural homologs")
    evalue_threshold: float = Field(default=1e-3, ge=0.0, description="E-value cutoff")


class Chai1Config(BaseModel):
    """Configuration for Chai-1 cross-reactivity check."""

    num_samples: int = Field(default=1, ge=1, le=5, description="Samples per binder-decoy pair")


class ScoringWeights(BaseModel):
    """Weights for the scoring function S(x)."""

    alpha: float = Field(default=1.0, ge=0.0, description="Weight for interface pLDDT (specificity)")
    beta: float = Field(default=0.5, ge=0.0, description="Weight for max decoy affinity (selectivity penalty)")


class PipelineConfig(BaseModel):
    """Complete pipeline configuration."""

    target: TargetProtein
    rfdiffusion: RFDiffusionConfig = Field(default_factory=RFDiffusionConfig)
    proteinmpnn: ProteinMPNNConfig = Field(default_factory=ProteinMPNNConfig)
    boltz2: Boltz2Config = Field(default_factory=Boltz2Config)
    foldseek: FoldSeekConfig = Field(default_factory=FoldSeekConfig)
    chai1: Chai1Config = Field(default_factory=Chai1Config)
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

    @field_validator("sequence")
    @classmethod
    def validate_sequence(cls, v: str) -> str:
        valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
        if not all(aa in valid_aa for aa in v.upper()):
            raise ValueError(f"Invalid amino acid in sequence: {v}")
        return v.upper()


class StructurePrediction(BaseModel):
    """Output from Boltz-2 structure prediction."""

    prediction_id: str = Field(..., description="Unique identifier")
    sequence_id: str = Field(..., description="Parent sequence design ID")
    pdb_path: str = Field(..., description="Path to predicted complex PDB")
    plddt_overall: float = Field(..., ge=0.0, le=100.0, description="Overall pLDDT score")
    plddt_interface: float = Field(..., ge=0.0, le=100.0, description="Interface pLDDT (i-pLDDT)")
    pae_interface: float = Field(..., ge=0.0, description="Interface PAE score")
    ptm: Optional[float] = Field(None, ge=0.0, le=1.0, description="pTM score")
    iptm: Optional[float] = Field(None, ge=0.0, le=1.0, description="ipTM score")


class DecoyHit(BaseModel):
    """A structural homolog (potential off-target) found by FoldSeek."""

    decoy_id: str = Field(..., description="Identifier (PDB ID or UniProt)")
    pdb_path: str = Field(..., description="Path to decoy structure")
    evalue: float = Field(..., description="E-value from FoldSeek")
    tm_score: float = Field(..., ge=0.0, le=1.0, description="TM-score alignment")
    aligned_length: int = Field(..., description="Number of aligned residues")
    sequence_identity: float = Field(..., ge=0.0, le=1.0, description="Sequence identity in alignment")


class CrossReactivityResult(BaseModel):
    """Output from Chai-1 cross-reactivity check."""

    binder_id: str = Field(..., description="Binder sequence ID")
    decoy_id: str = Field(..., description="Decoy protein ID")
    predicted_affinity: float = Field(..., description="Predicted binding affinity (lower = tighter)")
    plddt_interface: float = Field(..., ge=0.0, le=100.0, description="Interface pLDDT with decoy")
    binds_decoy: bool = Field(..., description="Whether binder likely binds decoy")


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


def generate_design_id(prefix: str = "design") -> str:
    """Generate a unique design identifier."""
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
