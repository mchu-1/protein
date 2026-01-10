"""
generators.py - RFDiffusion and ProteinMPNN Modal functions for backbone and sequence generation.

This module implements:
- RFDiffusion backbone generation (Phase 1, Step 1)
- ProteinMPNN sequence design (Phase 1, Step 2)
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import modal

from common import (
    DATA_PATH,
    WEIGHTS_PATH,
    BackboneDesign,
    ProteinMPNNConfig,
    RFDiffusionConfig,
    SequenceDesign,
    TargetProtein,
    app,
    data_volume,
    generate_design_id,
    proteinmpnn_image,
    rfdiffusion_image,
    weights_volume,
    write_fasta,
)

# =============================================================================
# RFDiffusion - Backbone Generation
# =============================================================================


@app.function(
    image=rfdiffusion_image,
    gpu="A10G",  # Cost-effective GPU sufficient for RFDiffusion
    timeout=600,
    volumes={WEIGHTS_PATH: weights_volume, DATA_PATH: data_volume},
)
def run_rfdiffusion(
    target: TargetProtein,
    config: RFDiffusionConfig,
    output_dir: str,
) -> list[BackboneDesign]:
    """
    Generate protein binder backbones using RFDiffusion.

    Args:
        target: Target protein specification with hotspot residues
        config: RFDiffusion configuration parameters
        output_dir: Directory to store output PDB files

    Returns:
        List of BackboneDesign objects for each generated backbone
    """
    import numpy as np

    os.makedirs(output_dir, exist_ok=True)

    # Build hotspot string for RFDiffusion (e.g., "A10,A15,A20")
    hotspot_str = ",".join(
        f"{target.chain_id}{res}" for res in target.hotspot_residues
    )

    # Determine target chain residue range from PDB file
    min_res, max_res = _get_chain_residue_range(target.pdb_path, target.chain_id)
    
    # Build contigmap: defines binder length range and target interaction
    # Format: [binder_length/0 target_chain_residues]
    # Use actual residue numbering from PDB (may not start at 1)
    contigmap = f"[{config.binder_length_min}-{config.binder_length_max}/0 {target.chain_id}{min_res}-{max_res}]"

    designs: list[BackboneDesign] = []

    try:
        # RFDiffusion inference command
        # The official container has models at /app/RFdiffusion/models or mounted at WEIGHTS_PATH
        model_dir = (
            f"{WEIGHTS_PATH}/rfdiffusion"
            if os.path.exists(f"{WEIGHTS_PATH}/rfdiffusion")
            else "/app/RFdiffusion/models"
        )
        # Run inference script directly (not as module) since RFDiffusion isn't structured for -m
        inference_script = "/app/RFdiffusion/scripts/run_inference.py"
        cmd = [
            "python",
            inference_script,
            f"inference.input_pdb={target.pdb_path}",
            f"inference.output_prefix={output_dir}/design",
            f"inference.num_designs={config.num_designs}",
            f"contigmap.contigs={contigmap}",
            f"ppi.hotspot_res=[{hotspot_str}]",
            f"diffuser.T={config.num_diffusion_steps}",
            f"denoiser.noise_scale_ca={config.noise_scale}",
            f"denoiser.noise_scale_frame={config.noise_scale}",
            f"inference.model_directory_path={model_dir}",
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd="/",
        )

        if result.returncode != 0:
            print(f"RFDiffusion stderr: {result.stderr}")
            # Try to continue with any designs that were generated

        # Collect generated designs
        for i in range(config.num_designs):
            pdb_path = f"{output_dir}/design_{i}.pdb"
            if os.path.exists(pdb_path):
                design_id = generate_design_id("backbone")

                # Extract binder length from PDB
                binder_length = _count_binder_residues(pdb_path, "B")

                designs.append(
                    BackboneDesign(
                        design_id=design_id,
                        pdb_path=pdb_path,
                        target_pdb_path=target.pdb_path,
                        hotspot_residues=target.hotspot_residues,
                        binder_length=binder_length,
                        rfdiffusion_score=None,  # Score extracted if available
                    )
                )

    except Exception as e:
        print(f"RFDiffusion error: {e}")
        # Return empty list - graceful failure as per coding standards
        return []

    # Commit volume changes
    data_volume.commit()

    return designs


@app.function(
    image=rfdiffusion_image,
    gpu="A10G",
    timeout=300,
    volumes={WEIGHTS_PATH: weights_volume, DATA_PATH: data_volume},
)
def run_rfdiffusion_single(
    target: TargetProtein,
    config: RFDiffusionConfig,
    design_index: int,
    output_dir: str,
) -> Optional[BackboneDesign]:
    """
    Generate a single backbone design. Useful for parallel generation.

    Args:
        target: Target protein specification
        config: RFDiffusion configuration
        design_index: Index of this design (for naming)
        output_dir: Output directory

    Returns:
        BackboneDesign if successful, None otherwise
    """
    # Modify config to generate single design
    single_config = RFDiffusionConfig(
        num_designs=1,
        binder_length_min=config.binder_length_min,
        binder_length_max=config.binder_length_max,
        noise_scale=config.noise_scale,
        num_diffusion_steps=config.num_diffusion_steps,
    )

    designs = run_rfdiffusion.local(target, single_config, output_dir)

    if designs:
        return designs[0]
    return None


def _count_binder_residues(pdb_path: str, chain_id: str = "B") -> int:
    """Count residues in the binder chain of a PDB file."""
    try:
        from Bio.PDB import PDBParser

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_path)

        for model in structure:
            for chain in model:
                if chain.id == chain_id:
                    return len([r for r in chain if r.id[0] == " "])
    except Exception:
        pass

    # Fallback: count ATOM lines with specified chain
    count = 0
    seen_residues = set()
    with open(pdb_path, "r") as f:
        for line in f:
            if line.startswith("ATOM") and len(line) > 25:
                if line[21] == chain_id:
                    res_id = line[22:27].strip()
                    if res_id not in seen_residues:
                        seen_residues.add(res_id)
                        count += 1
    return count


def _get_chain_residue_range(pdb_path: str, chain_id: str) -> tuple[int, int]:
    """Get the (min, max) residue numbers for a chain in a PDB file."""
    min_res = float('inf')
    max_res = 0
    try:
        with open(pdb_path, "r") as f:
            for line in f:
                if line.startswith("ATOM") and len(line) > 25:
                    if line[21] == chain_id:
                        try:
                            res_num = int(line[22:26].strip())
                            min_res = min(min_res, res_num)
                            max_res = max(max_res, res_num)
                        except ValueError:
                            pass
    except Exception:
        pass
    
    # Default to reasonable values if we couldn't parse
    if min_res == float('inf'):
        min_res = 1
    if max_res == 0:
        max_res = 200
    
    return (int(min_res), max_res)


# =============================================================================
# ProteinMPNN - Sequence Design
# =============================================================================


@app.function(
    image=proteinmpnn_image,
    gpu="L4",  # Lightweight GPU sufficient for ProteinMPNN
    timeout=300,
    volumes={WEIGHTS_PATH: weights_volume, DATA_PATH: data_volume},
)
def run_proteinmpnn(
    backbone: BackboneDesign,
    config: ProteinMPNNConfig,
    output_dir: str,
) -> list[SequenceDesign]:
    """
    Design amino acid sequences for a given backbone structure using ProteinMPNN.

    Args:
        backbone: BackboneDesign from RFDiffusion
        config: ProteinMPNN configuration
        output_dir: Directory to store output FASTA files

    Returns:
        List of SequenceDesign objects
    """
    os.makedirs(output_dir, exist_ok=True)

    sequences: list[SequenceDesign] = []

    try:
        # Prepare chains to design (design binder chain B, fix target chain A)
        chains_to_design = "B"

        # ProteinMPNN path from environment (set in image) or fallback
        proteinmpnn_path = os.environ.get("PROTEINMPNN_PATH", "/app/ProteinMPNN")
        
        # ProteinMPNN command
        # Note: ProteinMPNN designs all chains by default; use --pdb_path_chains to specify
        # which chains to include. For binder design, we include both target (A) and binder (B)
        cmd = [
            "python",
            f"{proteinmpnn_path}/protein_mpnn_run.py",
            "--pdb_path",
            backbone.pdb_path,
            "--pdb_path_chains",
            chains_to_design,  # Only design chain B (binder), chain A (target) is fixed
            "--out_folder",
            output_dir,
            "--num_seq_per_target",
            str(config.num_sequences),
            "--sampling_temp",
            str(config.temperature),
            "--backbone_noise",
            str(config.backbone_noise),
            "--model_name",
            "v_48_020",  # Standard ProteinMPNN model
            "--path_to_model_weights",
            f"{proteinmpnn_path}/vanilla_model_weights",
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"ProteinMPNN stderr: {result.stderr}")

        # Parse output FASTA file
        output_fasta = f"{output_dir}/seqs/{Path(backbone.pdb_path).stem}.fa"
        if os.path.exists(output_fasta):
            sequences = _parse_proteinmpnn_output(
                output_fasta, backbone.design_id, output_dir
            )
        else:
            # Try alternative output location
            alt_fasta = f"{output_dir}/{Path(backbone.pdb_path).stem}.fa"
            if os.path.exists(alt_fasta):
                sequences = _parse_proteinmpnn_output(
                    alt_fasta, backbone.design_id, output_dir
                )

    except Exception as e:
        print(f"ProteinMPNN error: {e}")
        return []

    # Commit volume changes
    data_volume.commit()

    return sequences


@app.function(
    image=proteinmpnn_image,
    gpu="L4",
    timeout=300,
    volumes={WEIGHTS_PATH: weights_volume, DATA_PATH: data_volume},
)
def run_proteinmpnn_batch(
    backbones: list[BackboneDesign],
    config: ProteinMPNNConfig,
    output_dir: str,
) -> list[SequenceDesign]:
    """
    Run ProteinMPNN on multiple backbones in batch mode.

    Args:
        backbones: List of backbone designs
        config: ProteinMPNN configuration
        output_dir: Output directory

    Returns:
        List of all sequence designs across all backbones
    """
    all_sequences: list[SequenceDesign] = []

    for backbone in backbones:
        backbone_output_dir = f"{output_dir}/{backbone.design_id}"
        seqs = run_proteinmpnn.local(backbone, config, backbone_output_dir)
        all_sequences.extend(seqs)

    return all_sequences


def _parse_proteinmpnn_output(
    fasta_path: str,
    backbone_id: str,
    output_dir: str,
) -> list[SequenceDesign]:
    """
    Parse ProteinMPNN output FASTA file into SequenceDesign objects.

    ProteinMPNN output format:
    >design_name, score=X.XXX, global_score=X.XXX, ...
    SEQUENCE...
    """
    from Bio import SeqIO

    sequences: list[SequenceDesign] = []

    try:
        for i, record in enumerate(SeqIO.parse(fasta_path, "fasta")):
            sequence_id = generate_design_id("seq")

            # Parse score from header
            score = 0.0
            header = record.description
            if "score=" in header:
                try:
                    score_str = header.split("score=")[1].split(",")[0].strip()
                    score = float(score_str)
                except (IndexError, ValueError):
                    pass

            # Extract just the binder sequence (chain B)
            # ProteinMPNN may output full sequence with chain separator
            seq_str = str(record.seq)
            if "/" in seq_str:
                # Take the designed chain (usually second after separator)
                parts = seq_str.split("/")
                seq_str = parts[-1] if len(parts) > 1 else parts[0]

            # Write individual FASTA file
            individual_fasta = f"{output_dir}/{sequence_id}.fasta"
            write_fasta(seq_str, f"{sequence_id}|backbone={backbone_id}", individual_fasta)

            sequences.append(
                SequenceDesign(
                    sequence_id=sequence_id,
                    backbone_id=backbone_id,
                    sequence=seq_str,
                    fasta_path=individual_fasta,
                    score=score,
                    recovery=None,
                )
            )

    except Exception as e:
        print(f"Error parsing ProteinMPNN output: {e}")

    return sequences


# =============================================================================
# Parallel Generation Utilities
# =============================================================================


@app.function(
    image=proteinmpnn_image,
    timeout=60,
)
def generate_sequences_parallel(
    backbones: list[BackboneDesign],
    config: ProteinMPNNConfig,
    base_output_dir: str,
) -> list[SequenceDesign]:
    """
    Generate sequences for multiple backbones in parallel using starmap.

    This leverages Modal's starmap for efficient parallel execution.

    Args:
        backbones: List of backbone designs
        config: ProteinMPNN configuration
        base_output_dir: Base directory for outputs

    Returns:
        Flattened list of all sequence designs
    """
    # Prepare arguments for starmap
    args = [
        (backbone, config, f"{base_output_dir}/{backbone.design_id}")
        for backbone in backbones
    ]

    # Use starmap for parallel execution
    all_results = list(run_proteinmpnn.starmap(args))

    # Flatten results
    all_sequences: list[SequenceDesign] = []
    for seq_list in all_results:
        all_sequences.extend(seq_list)

    return all_sequences


# =============================================================================
# Mock Implementations (for testing without GPU)
# =============================================================================


def mock_rfdiffusion(
    target: TargetProtein,
    config: RFDiffusionConfig,
    output_dir: str,
) -> list[BackboneDesign]:
    """
    Mock RFDiffusion for testing without GPU resources.
    Generates placeholder backbone designs.
    """
    import random

    os.makedirs(output_dir, exist_ok=True)
    designs: list[BackboneDesign] = []

    for i in range(config.num_designs):
        design_id = generate_design_id("mock_backbone")
        pdb_path = f"{output_dir}/{design_id}.pdb"

        # Create a minimal placeholder PDB
        binder_length = random.randint(
            config.binder_length_min, config.binder_length_max
        )
        _create_mock_pdb(pdb_path, binder_length)

        designs.append(
            BackboneDesign(
                design_id=design_id,
                pdb_path=pdb_path,
                target_pdb_path=target.pdb_path,
                hotspot_residues=target.hotspot_residues,
                binder_length=binder_length,
                rfdiffusion_score=random.uniform(0.5, 0.9),
            )
        )

    return designs


def mock_proteinmpnn(
    backbone: BackboneDesign,
    config: ProteinMPNNConfig,
    output_dir: str,
) -> list[SequenceDesign]:
    """
    Mock ProteinMPNN for testing without GPU resources.
    Generates random amino acid sequences.
    """
    import random

    os.makedirs(output_dir, exist_ok=True)
    sequences: list[SequenceDesign] = []

    amino_acids = "ACDEFGHIKLMNPQRSTVWY"

    for i in range(config.num_sequences):
        sequence_id = generate_design_id("mock_seq")

        # Generate random sequence of appropriate length
        seq = "".join(random.choices(amino_acids, k=backbone.binder_length))
        fasta_path = f"{output_dir}/{sequence_id}.fasta"
        write_fasta(seq, f"{sequence_id}|backbone={backbone.design_id}", fasta_path)

        sequences.append(
            SequenceDesign(
                sequence_id=sequence_id,
                backbone_id=backbone.design_id,
                sequence=seq,
                fasta_path=fasta_path,
                score=random.uniform(-3.0, -1.0),
                recovery=None,
            )
        )

    return sequences


def _create_mock_pdb(pdb_path: str, num_residues: int) -> None:
    """Create a minimal mock PDB file for testing."""
    with open(pdb_path, "w") as f:
        f.write("HEADER    MOCK STRUCTURE\n")
        atom_num = 1
        for i in range(num_residues):
            # Write a single CA atom per residue
            x, y, z = i * 3.8, 0.0, 0.0  # Linear chain
            f.write(
                f"ATOM  {atom_num:5d}  CA  ALA B{i+1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
            )
            atom_num += 1
        f.write("END\n")
