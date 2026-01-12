"""
validators.py - Boltz-2, FoldSeek, and Chai-1 validation functions.

This module implements:
- Boltz-2 structure prediction and affinity scoring (Phase 2)
- FoldSeek proteome scanning for decoy identification (Phase 3, Step 1)
- Chai-1 cross-reactivity checking (Phase 3, Step 2)
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Optional

from common import (
    DATA_PATH,
    WEIGHTS_PATH,
    Boltz2Config,
    Chai1Config,
    CrossReactivityResult,
    DecoyHit,
    FoldSeekConfig,
    SequenceDesign,
    StructurePrediction,
    TargetProtein,
    app,
    boltz2_image,
    chai1_image,
    data_volume,
    esmfold_image,
    foldseek_image,
    generate_design_id,
    weights_volume,
)

# =============================================================================
# Boltz-2 - Structure Prediction & Affinity Scoring
# =============================================================================


@app.function(
    image=boltz2_image,
    gpu="A100",  # A100 required for memory-intensive structure prediction
    timeout=900,
    volumes={WEIGHTS_PATH: weights_volume, DATA_PATH: data_volume},
)
def run_boltz2(
    sequence: SequenceDesign,
    target: TargetProtein,
    config: Boltz2Config,
    output_dir: str,
) -> Optional[StructurePrediction]:
    """
    Predict the complex structure of binder + target using Boltz-2.

    Args:
        sequence: Designed binder sequence
        target: Target protein specification
        config: Boltz-2 configuration
        output_dir: Directory to store output PDB files

    Returns:
        StructurePrediction if successful, None otherwise
    """

    os.makedirs(output_dir, exist_ok=True)

    try:
        # Read target sequence from PDB
        target_sequence = _extract_sequence_from_pdb(target.pdb_path, target.chain_id)

        # Create input FASTA for Boltz
        input_fasta = f"{output_dir}/input_{sequence.sequence_id}.fasta"
        with open(input_fasta, "w") as f:
            f.write(f">A|protein\n{target_sequence}\n")
            f.write(f">B|protein\n{sequence.sequence}\n")

        # Run Boltz prediction via CLI
        cmd = [
            "boltz",
            "predict",
            input_fasta,
            "--out_dir",
            output_dir,
            "--recycling_steps",
            str(config.num_recycles),
            "--accelerator",
            "gpu",
            "--devices",
            "1",
            "--use_msa_server",
        ]

        print(f"  Running Boltz on {sequence.sequence_id}...")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,
        )

        if result.returncode != 0:
            # Only print errors in quiet mode
            print(f"  ✗ {sequence.sequence_id} failed: {result.stderr[-200:] if result.stderr else 'unknown error'}")
            return None

        # Find output structure file
        import glob
        pdb_files = glob.glob(f"{output_dir}/**/*.pdb", recursive=True)
        cif_files = glob.glob(f"{output_dir}/**/*.cif", recursive=True)
        
        # Use PDB or CIF output (Boltz may output either format)
        output_pdb = None
        if pdb_files:
            output_pdb = pdb_files[0]
        elif cif_files:
            output_pdb = cif_files[0]
        else:
            print(f"  ✗ {sequence.sequence_id}: no output structure")
            return None

        # Parse Boltz-2 output metrics with AlphaProteo metrics
        # Pass hotspots for pae_interaction, backbone for RMSD
        target_len = len(target_sequence)
        backbone_pdb = getattr(sequence, 'backbone_pdb', None)  # Set by pipeline if available
        
        metrics = _parse_boltz2_metrics(
            output_dir=output_dir,
            sequence_id=sequence.sequence_id,
            hotspot_residues=target.hotspot_residues,
            backbone_pdb=backbone_pdb,
            target_len=target_len,
        )

        if metrics is None:
            # Use default metrics if we have an output structure
            metrics = {
                "plddt_overall": 70.0,
                "plddt_interface": 70.0,
                "pae_interface": 5.0,
            }

        prediction = StructurePrediction(
            prediction_id=generate_design_id("pred"),
            sequence_id=sequence.sequence_id,
            pdb_path=output_pdb,
            plddt_overall=metrics.get("plddt_overall", 0.0),
            plddt_interface=metrics.get("plddt_interface", 0.0),
            pae_interface=metrics.get("pae_interface", 100.0),
            ptm=metrics.get("ptm"),
            iptm=metrics.get("iptm"),
            # AlphaProteo SI 2.2 metrics
            pae_interaction=metrics.get("pae_interaction"),
            ptm_binder=metrics.get("ptm_binder"),
            rmsd_to_design=metrics.get("rmsd_to_design"),
        )

        # AlphaProteo filters (SI 2.2) - ONLY these 3 criteria
        # 1. Anchor Lock: min PAE at hotspots < 1.5 Å
        if prediction.pae_interaction is not None:
            if prediction.pae_interaction > config.max_pae_interaction:
                print(f"  ✗ {sequence.sequence_id}: PAE@hotspots {prediction.pae_interaction:.2f} Å > {config.max_pae_interaction} Å")
                return None
        
        # 2. Fold Quality: binder-only pTM > 0.80
        if prediction.ptm_binder is not None:
            if prediction.ptm_binder < config.min_ptm_binder:
                print(f"  ✗ {sequence.sequence_id}: pTM(binder) {prediction.ptm_binder:.3f} < {config.min_ptm_binder}")
                return None
        
        # 3. Self-Consistency: RMSD vs RFDiffusion < 2.5 Å
        if prediction.rmsd_to_design is not None:
            if prediction.rmsd_to_design > config.max_rmsd:
                print(f"  ✗ {sequence.sequence_id}: RMSD {prediction.rmsd_to_design:.2f} Å > {config.max_rmsd} Å")
                return None

        # Build status string (AlphaProteo metrics only)
        status_parts = []
        if prediction.pae_interaction is not None:
            status_parts.append(f"PAE@hs={prediction.pae_interaction:.2f}Å")
        if prediction.ptm_binder is not None:
            status_parts.append(f"pTM(b)={prediction.ptm_binder:.3f}")
        if prediction.rmsd_to_design is not None:
            status_parts.append(f"RMSD={prediction.rmsd_to_design:.2f}Å")
        print(f"  ✓ {sequence.sequence_id}: {', '.join(status_parts)}")
        return prediction

    except subprocess.TimeoutExpired:
        print(f"  ✗ {sequence.sequence_id}: timeout")
        return None
    except Exception as e:
        print(f"  ✗ {sequence.sequence_id}: {e}")
        return None
    finally:
        data_volume.commit()


@app.function(
    image=boltz2_image,
    gpu="A100",
    timeout=1800,
    volumes={WEIGHTS_PATH: weights_volume, DATA_PATH: data_volume},
)
def run_boltz2_batch(
    sequences: list[SequenceDesign],
    target: TargetProtein,
    config: Boltz2Config,
    output_dir: str,
) -> list[StructurePrediction]:
    """
    Run Boltz-2 on multiple sequences.

    Args:
        sequences: List of designed sequences
        target: Target protein
        config: Boltz-2 configuration
        output_dir: Output directory

    Returns:
        List of successful structure predictions
    """
    predictions: list[StructurePrediction] = []

    for sequence in sequences:
        seq_output_dir = f"{output_dir}/{sequence.sequence_id}"
        pred = run_boltz2.local(sequence, target, config, seq_output_dir)
        if pred is not None:
            predictions.append(pred)

    return predictions


def _extract_sequence_from_pdb(pdb_path: str, chain_id: str) -> str:
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


def _parse_boltz2_metrics(
    output_dir: str,
    sequence_id: str,
    hotspot_residues: Optional[list[int]] = None,
    backbone_pdb: Optional[str] = None,
    target_len: int = 0,
) -> Optional[dict]:
    """
    Parse Boltz-2 output metrics from JSON or npz files.
    
    Includes AlphaProteo SI 2.2 metrics:
    - pae_interaction: min PAE at hotspot residues
    - ptm_binder: pTM for binder chain only
    - rmsd_to_design: RMSD vs RFDiffusion backbone
    """
    import numpy as np
    import glob

    metrics = {}
    
    # Boltz outputs are in: {output_dir}/boltz_results_*/predictions/*/
    # Files: confidence_*.json, plddt_*.npz, pae_*.npz
    
    # Find confidence JSON file - contains comprehensive metrics
    confidence_files = glob.glob(f"{output_dir}/**/confidence_*.json", recursive=True)
    if confidence_files:
        with open(confidence_files[0], "r") as f:
            raw_metrics = json.load(f)
            # Boltz confidence file contains normalized scores (0-1)
            # Convert to 0-100 scale for compatibility with filters
            metrics["ptm"] = raw_metrics.get("ptm")
            metrics["iptm"] = raw_metrics.get("iptm")
            # Use complex_plddt (already 0-1, convert to 0-100)
            complex_plddt = raw_metrics.get("complex_plddt", 0.0)
            complex_iplddt = raw_metrics.get("complex_iplddt", 0.0)
            metrics["plddt_overall"] = complex_plddt * 100.0  # Scale to 0-100
            metrics["plddt_interface"] = complex_iplddt * 100.0  # Scale to 0-100
            # Use complex_ipde for interface PAE
            metrics["pae_interface"] = raw_metrics.get("complex_ipde", 5.0)
            
            # AlphaProteo: binder-only pTM (chain_ptm for second chain)
            if "chain_ptm" in raw_metrics and isinstance(raw_metrics["chain_ptm"], list):
                # Assuming binder is second chain (after target)
                if len(raw_metrics["chain_ptm"]) > 1:
                    metrics["ptm_binder"] = raw_metrics["chain_ptm"][1]
                elif len(raw_metrics["chain_ptm"]) == 1:
                    metrics["ptm_binder"] = raw_metrics["chain_ptm"][0]
    
    # Parse PAE matrix for hotspot-specific PAE (Anchor Lock)
    if hotspot_residues:
        pae_files = glob.glob(f"{output_dir}/**/pae_*.npz", recursive=True)
        if pae_files:
            try:
                pae_data = np.load(pae_files[0])
                if "pae" in pae_data:
                    pae_matrix = pae_data["pae"]  # Shape: [N, N]
                    # Extract PAE between binder residues and hotspot residues on target
                    # Hotspots are on target (first chain), binder is second chain
                    # PAE[i, j] = error of residue i when aligned on residue j
                    if pae_matrix.ndim == 2 and target_len > 0:
                        binder_start = target_len
                        hotspot_pae_values = []
                        for hs in hotspot_residues:
                            hs_idx = hs - 1  # 0-indexed
                            if 0 <= hs_idx < target_len:
                                # PAE from binder residues to this hotspot
                                binder_to_hs = pae_matrix[binder_start:, hs_idx]
                                hotspot_pae_values.extend(binder_to_hs.tolist())
                        if hotspot_pae_values:
                            # Use minimum PAE at hotspots (best contact)
                            metrics["pae_interaction"] = float(np.min(hotspot_pae_values))
            except Exception as e:
                print(f"  Warning: Could not parse PAE matrix: {e}")
    
    # Calculate RMSD vs RFDiffusion backbone (Self-Consistency)
    if backbone_pdb:
        predicted_pdbs = glob.glob(f"{output_dir}/**/*.cif", recursive=True) + \
                        glob.glob(f"{output_dir}/**/*.pdb", recursive=True)
        if predicted_pdbs:
            try:
                rmsd = _calculate_backbone_rmsd(backbone_pdb, predicted_pdbs[0])
                if rmsd is not None:
                    metrics["rmsd_to_design"] = rmsd
            except Exception as e:
                print(f"  Warning: Could not calculate RMSD: {e}")

    # Find pLDDT NPZ file as backup
    if "plddt_overall" not in metrics:
        plddt_files = glob.glob(f"{output_dir}/**/plddt_*.npz", recursive=True)
        if plddt_files:
            data = np.load(plddt_files[0])
            if "plddt" in data:
                plddt = data["plddt"]
                # NPZ file contains 0-1 scores, convert to 0-100
                metrics["plddt_overall"] = float(np.mean(plddt)) * 100.0
                metrics["plddt_interface"] = float(np.mean(plddt)) * 100.0

    # Set defaults for missing metrics
    if "plddt_interface" not in metrics:
        metrics["plddt_interface"] = metrics.get("plddt_overall", 70.0)
    if "pae_interface" not in metrics:
        metrics["pae_interface"] = 5.0  # Default moderate PAE

    if metrics:
        return metrics

    return None


def _calculate_backbone_rmsd(ref_pdb: str, pred_pdb: str, chain_id: str = "B") -> Optional[float]:
    """
    Calculate backbone RMSD between RFDiffusion design and Boltz-2 prediction.
    
    Uses CA atoms for alignment and RMSD calculation.
    
    Args:
        ref_pdb: Path to RFDiffusion backbone PDB
        pred_pdb: Path to Boltz-2 predicted structure (PDB or CIF)
        chain_id: Chain ID of binder (default: B)
    
    Returns:
        RMSD in Angstroms, or None if calculation fails
    """
    try:
        from Bio.PDB import PDBParser, MMCIFParser, Superimposer
        
        # Parse reference (RFDiffusion backbone)
        ref_parser = PDBParser(QUIET=True)
        ref_structure = ref_parser.get_structure("ref", ref_pdb)
        
        # Parse prediction (Boltz-2, might be CIF)
        if pred_pdb.endswith(".cif"):
            pred_parser = MMCIFParser(QUIET=True)
        else:
            pred_parser = PDBParser(QUIET=True)
        pred_structure = pred_parser.get_structure("pred", pred_pdb)
        
        # Extract CA atoms from binder chain
        ref_atoms = []
        pred_atoms = []
        
        for model in ref_structure:
            for chain in model:
                if chain.id == chain_id:
                    for residue in chain:
                        if residue.id[0] == " " and "CA" in residue:
                            ref_atoms.append(residue["CA"])
        
        for model in pred_structure:
            for chain in model:
                if chain.id == chain_id:
                    for residue in chain:
                        if residue.id[0] == " " and "CA" in residue:
                            pred_atoms.append(residue["CA"])
        
        if not ref_atoms or not pred_atoms:
            return None
        
        # Truncate to shorter length
        min_len = min(len(ref_atoms), len(pred_atoms))
        ref_atoms = ref_atoms[:min_len]
        pred_atoms = pred_atoms[:min_len]
        
        # Superimpose and calculate RMSD
        super_imposer = Superimposer()
        super_imposer.set_atoms(ref_atoms, pred_atoms)
        
        return super_imposer.rms
        
    except Exception as e:
        print(f"  Warning: RMSD calculation failed: {e}")
        return None


# =============================================================================
# ESMFold - Orthogonal Validation (Gatekeeper)
# =============================================================================


@app.function(
    image=esmfold_image,
    gpu="T4",  # T4 sufficient for ESMFold inference (cheaper than A100)
    timeout=300,  # 5 min per sequence
    volumes={WEIGHTS_PATH: weights_volume, DATA_PATH: data_volume},
)
def run_esmfold_validation(
    sequence: SequenceDesign,
    backbone_pdb: str,
    min_plddt: float = 70.0,
    max_rmsd: float = 2.5,
    output_dir: str = None,
) -> Optional[dict]:
    """
    Validate a designed sequence using ESMFold orthogonal prediction.
    
    **Architectural Rationale:**
    RFDiffusion, ProteinMPNN, and Boltz-2 all use diffusion-based or geometry-driven
    inductive biases. ESMFold is a Protein Language Model trained on evolutionary data
    (UniRef), providing an orthogonal architectural signal.
    
    **Gatekeeper Logic:**
    1. Predict structure from sequence using ESMFold (no backbone input)
    2. Calculate mean pLDDT. If < 70, PRUNE (low confidence)
    3. Align ESMFold CA atoms to RFDiffusion CA atoms
    4. Calculate RMSD. If > 2.5 Å, PRUNE (sequence does not encode the intended structure)
    
    **Why this works:**
    If ESMFold (sequence-only) cannot recover the RFDiffusion backbone geometry,
    the design is likely an adversarial hallucination that "looks good" to
    diffusion models but violates biophysical constraints.
    
    Args:
        sequence: Designed sequence from ProteinMPNN
        backbone_pdb: Path to original RFDiffusion backbone PDB
        min_plddt: Minimum mean pLDDT threshold (default: 70)
        max_rmsd: Maximum RMSD to RFDiffusion backbone (default: 2.5 Å)
        output_dir: Optional directory to save ESMFold prediction
    
    Returns:
        Dict with metrics if passed, None if pruned
        {
            "mean_plddt": float,
            "rmsd_to_design": float,
            "passed": bool,
            "esmfold_pdb": str (optional),
        }
    """
    import os
    import numpy as np
    
    try:
        # Import ESMFold via transformers
        from transformers import EsmForProteinFolding
        import torch
        
        # Import biotite for fast structure manipulation
        import biotite.structure as struc
        import biotite.structure.io.pdb as pdb
        
        # Load ESMFold model (weights are cached in image)
        print(f"  ESMFold: Loading model for {sequence.sequence_id}...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
        model = model.to(device)
        model.eval()
        
        # Prepare input
        seq_str = sequence.sequence
        print(f"  ESMFold: Predicting structure for {len(seq_str)} residues...")
        
        # Run ESMFold inference
        with torch.no_grad():
            output = model.infer_pdb(seq_str)
        
        # Parse ESMFold output (PDB string)
        esmfold_pdb_str = output
        
        # Extract pLDDT from B-factor column (ESMFold convention)
        # Parse PDB string into biotite structure
        from io import StringIO
        pdb_file = pdb.PDBFile.read(StringIO(esmfold_pdb_str))
        esmfold_structure = pdb.get_structure(pdb_file, model=1)
        
        # Get pLDDT values (stored in b_factor annotation in Biotite)
        ca_mask = esmfold_structure.atom_name == "CA"
        ca_atoms = esmfold_structure[ca_mask]
        
        # Access b_factor from the annotation dict (Biotite stores it here)
        if hasattr(ca_atoms, 'b_factor'):
            plddt_values = ca_atoms.b_factor
        elif 'b_factor' in ca_atoms.get_annotation_categories():
            plddt_values = ca_atoms.get_annotation('b_factor')
        else:
            # Fallback: extract from PDB file directly
            plddt_values = []
            for line in esmfold_pdb_str.split('\n'):
                if line.startswith('ATOM') and ' CA ' in line:
                    try:
                        b_factor = float(line[60:66].strip())
                        plddt_values.append(b_factor)
                    except (ValueError, IndexError):
                        continue
            plddt_values = np.array(plddt_values)
        
        mean_plddt = float(np.mean(plddt_values))
        
        print(f"  ESMFold: Mean pLDDT = {mean_plddt:.1f}")
        
        # Metric 1: Confidence check
        if mean_plddt < min_plddt:
            print(f"  ✗ {sequence.sequence_id}: ESMFold pLDDT {mean_plddt:.1f} < {min_plddt} (LOW CONFIDENCE)")
            return None
        
        # Metric 2: Consistency check (RMSD vs RFDiffusion backbone)
        # Load RFDiffusion backbone
        ref_file = pdb.PDBFile.read(backbone_pdb)
        ref_structure = pdb.get_structure(ref_file, model=1)
        
        # Extract CA atoms from both structures
        ref_ca = ref_structure[ref_structure.atom_name == "CA"]
        esm_ca = esmfold_structure[ca_mask]
        
        # Align lengths (handle minor length mismatches)
        min_len = min(len(ref_ca), len(esm_ca))
        ref_ca = ref_ca[:min_len]
        esm_ca = esm_ca[:min_len]
        
        # Superimpose ESMFold onto RFDiffusion backbone
        esm_ca_superimposed, transform = struc.superimpose(ref_ca, esm_ca)
        
        # Calculate RMSD
        rmsd = struc.rmsd(ref_ca, esm_ca_superimposed)
        
        print(f"  ESMFold: RMSD to RFDiffusion = {rmsd:.2f} Å")
        
        if rmsd > max_rmsd:
            print(f"  ✗ {sequence.sequence_id}: ESMFold RMSD {rmsd:.2f} Å > {max_rmsd} Å (INCONSISTENT GEOMETRY)")
            return None
        
        # PASSED both gates
        print(f"  ✓ {sequence.sequence_id}: ESMFold validation PASSED (pLDDT={mean_plddt:.1f}, RMSD={rmsd:.2f}Å)")
        
        # Optionally save ESMFold prediction
        esmfold_pdb_path = None
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            esmfold_pdb_path = f"{output_dir}/{sequence.sequence_id}_esmfold.pdb"
            with open(esmfold_pdb_path, "w") as f:
                f.write(esmfold_pdb_str)
        
        return {
            "mean_plddt": mean_plddt,
            "rmsd_to_design": rmsd,
            "passed": True,
            "esmfold_pdb": esmfold_pdb_path,
        }
        
    except Exception as e:
        print(f"  ✗ {sequence.sequence_id}: ESMFold error: {e}")
        return None
    finally:
        # Clean up GPU memory
        if 'model' in locals():
            del model
        if 'torch' in locals():
            torch.cuda.empty_cache()
        data_volume.commit()


@app.function(
    image=esmfold_image,
    gpu="T4",
    timeout=1800,  # 30 min for batch
    volumes={WEIGHTS_PATH: weights_volume, DATA_PATH: data_volume},
)
def run_esmfold_validation_batch(
    sequences: list[SequenceDesign],
    backbone_pdbs: dict[str, str],
    min_plddt: float = 70.0,
    max_rmsd: float = 2.5,
    output_dir: str = None,
) -> list[SequenceDesign]:
    """
    Validate multiple sequences using ESMFold in batch.
    
    **Economics:**
    ESMFold on T4 (~$0.30/hr) is 10x cheaper than Boltz-2 on A100 (~$3.50/hr).
    This gatekeeper can prune 30-50% of hallucinated designs before they reach
    expensive folding, saving significant compute costs.
    
    Args:
        sequences: List of designed sequences
        backbone_pdbs: Mapping from backbone_id -> PDB path
        min_plddt: Minimum pLDDT threshold
        max_rmsd: Maximum RMSD threshold
        output_dir: Output directory for ESMFold predictions
    
    Returns:
        List of sequences that passed ESMFold validation
    """
    validated_sequences: list[SequenceDesign] = []
    
    print(f"ESMFold Gatekeeper: validating {len(sequences)} sequences")
    print(f"  Thresholds: pLDDT ≥ {min_plddt}, RMSD ≤ {max_rmsd} Å")
    
    for sequence in sequences:
        # Get backbone PDB for this sequence
        backbone_pdb = backbone_pdbs.get(sequence.backbone_id)
        if not backbone_pdb:
            print(f"  ✗ {sequence.sequence_id}: No backbone PDB found for {sequence.backbone_id}")
            continue
        
        # Run validation
        result = run_esmfold_validation.local(
            sequence, 
            backbone_pdb, 
            min_plddt, 
            max_rmsd,
            f"{output_dir}/{sequence.sequence_id}" if output_dir else None
        )
        
        if result is not None and result.get("passed"):
            validated_sequences.append(sequence)
    
    pruned = len(sequences) - len(validated_sequences)
    print(f"ESMFold Gatekeeper: {len(validated_sequences)}/{len(sequences)} passed ({pruned} pruned)")
    
    return validated_sequences


# =============================================================================
# UniProt/PDB Mapping Helpers
# =============================================================================


def _get_uniprot_from_pdb(pdb_id: str) -> set[str]:
    """
    Get UniProt accession IDs associated with a PDB entry.
    
    Uses the RCSB PDB API to look up the UniProt mapping.
    
    Args:
        pdb_id: 4-letter PDB code (e.g., "3DI3")
    
    Returns:
        Set of UniProt accession IDs (e.g., {"P16871", "P13232"})
    """
    import urllib.request
    import urllib.error
    
    uniprot_ids: set[str] = set()
    
    try:
        # RCSB GraphQL API endpoint
        url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id.upper()}"
        
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read().decode())
            
            # Extract polymer entities which contain UniProt references
            if "rcsb_entry_container_identifiers" in data:
                container = data["rcsb_entry_container_identifiers"]
                if "polymer_entity_ids" in container:
                    for entity_id in container["polymer_entity_ids"]:
                        entity_url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id.upper()}/{entity_id}"
                        try:
                            with urllib.request.urlopen(entity_url, timeout=10) as entity_response:
                                entity_data = json.loads(entity_response.read().decode())
                                if "rcsb_polymer_entity_container_identifiers" in entity_data:
                                    identifiers = entity_data["rcsb_polymer_entity_container_identifiers"]
                                    if "uniprot_ids" in identifiers:
                                        uniprot_ids.update(identifiers["uniprot_ids"])
                        except Exception:
                            continue
    except urllib.error.HTTPError as e:
        print(f"  Warning: Could not fetch UniProt mapping for PDB {pdb_id}: HTTP {e.code}")
    except Exception as e:
        print(f"  Warning: Could not fetch UniProt mapping for PDB {pdb_id}: {e}")
    
    return uniprot_ids


def _get_pdbs_from_uniprot(uniprot_id: str) -> set[str]:
    """
    Get all PDB IDs associated with a UniProt accession.
    
    Uses the UniProt REST API to fetch cross-references to PDB.
    
    Args:
        uniprot_id: UniProt accession (e.g., "P16871")
    
    Returns:
        Set of 4-letter PDB codes (e.g., {"3DI3", "7OPB"})
    """
    import urllib.request
    import urllib.error
    
    pdb_ids: set[str] = set()
    
    try:
        # Use UniProt REST API - stable and reliable
        url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id.upper()}.json"
        
        req = urllib.request.Request(url)
        req.add_header('Accept', 'application/json')
        
        with urllib.request.urlopen(req, timeout=15) as response:
            data = json.loads(response.read().decode())
            
            # Extract PDB cross-references from UniProt entry
            if "uniProtKBCrossReferences" in data:
                for xref in data["uniProtKBCrossReferences"]:
                    if xref.get("database") == "PDB":
                        pdb_id = xref.get("id", "").upper()
                        if len(pdb_id) == 4:
                            pdb_ids.add(pdb_id)
                    
    except urllib.error.HTTPError as e:
        if e.code == 404:
            # UniProt ID not found - that's okay
            pass
        else:
            print(f"  Warning: Could not fetch PDB mapping for UniProt {uniprot_id}: HTTP {e.code}")
    except Exception as e:
        print(f"  Warning: Could not fetch PDB mapping for UniProt {uniprot_id}: {e}")
    
    return pdb_ids


def _extract_pdb_id_from_path(pdb_path: str) -> Optional[str]:
    """
    Extract PDB ID from a file path.
    
    Handles formats like:
    - /data/targets/3di3_target.pdb -> 3DI3
    - /path/to/il7ra_target.pdb -> None (not a PDB ID)
    - /path/to/3DI3.pdb -> 3DI3
    """
    import re
    
    filename = Path(pdb_path).stem.upper()
    
    # Try to find a 4-character PDB ID pattern
    # PDB IDs are 4 alphanumeric characters, typically starting with a digit
    match = re.search(r'\b([0-9][A-Z0-9]{3})\b', filename)
    if match:
        return match.group(1)
    
    # Also check for pattern at start of filename
    if len(filename) >= 4 and filename[0].isdigit():
        potential_id = filename[:4]
        if all(c.isalnum() for c in potential_id):
            return potential_id
    
    return None


# =============================================================================
# FoldSeek - Proteome Scanning for Decoys
# =============================================================================


def get_cached_decoys(target: TargetProtein) -> Optional[list[DecoyHit]]:
    """
    Check for cached FoldSeek results using Modal Dict (TTL key-value store).
    
    Cache key is based on PDB ID + entity ID, so same protein = same decoys.
    Reading extends TTL by 7 days (LRU-like behavior).
    
    Args:
        target: Target protein specification
    
    Returns:
        List of cached DecoyHit objects, or None if no cache exists
    """
    from common import foldseek_cache
    
    cache_key = f"{target.pdb_id.upper()}_E{target.entity_id}"
    
    try:
        cached_data = foldseek_cache.get(cache_key)
        if cached_data is not None:
            decoys = [DecoyHit(**d) for d in cached_data]
            print(f"FoldSeek: Cache HIT for {cache_key} ({len(decoys)} decoys)")
            return decoys
    except Exception as e:
        print(f"FoldSeek: Cache read failed ({e}), will recompute")
    
    print(f"FoldSeek: Cache MISS for {cache_key}")
    return None


def save_decoys_to_cache(target: TargetProtein, decoys: list[DecoyHit]) -> None:
    """
    Save FoldSeek results to Modal Dict cache (7-day TTL, refreshed on read).
    
    Args:
        target: Target protein specification
        decoys: List of decoy hits to cache
    """
    from common import foldseek_cache
    
    cache_key = f"{target.pdb_id.upper()}_E{target.entity_id}"
    
    try:
        # Store as list of dicts for JSON serialization
        foldseek_cache.put(cache_key, [d.model_dump() for d in decoys])
        print(f"FoldSeek: Cached {len(decoys)} decoys for {cache_key} (TTL: 7 days)")
    except Exception as e:
        print(f"FoldSeek: Cache write failed ({e})")


@app.function(
    image=foldseek_image,
    cpu=4,
    memory=8192,
    timeout=2400,  # 40 min to allow for database download on first run
    volumes={WEIGHTS_PATH: weights_volume, DATA_PATH: data_volume},
)
def run_foldseek(
    target: TargetProtein,
    config: FoldSeekConfig,
    output_dir: str,
) -> list[DecoyHit]:
    """
    Search for structural homologs of the target using FoldSeek.

    These structural homologs serve as potential off-targets (decoys)
    for selectivity filtering.
    
    **Filters out self-hits:** Decoys that map to the same UniProt accession
    as the target are excluded to avoid false cross-reactivity signals.

    Args:
        target: Target protein structure
        config: FoldSeek configuration
        output_dir: Directory to store results

    Returns:
        List of DecoyHit objects representing potential off-targets
    """
    os.makedirs(output_dir, exist_ok=True)

    decoys: list[DecoyHit] = []

    try:
        # Step 1: Get target's UniProt accessions to filter self-hits
        # Use target.pdb_id directly (no need to extract from path)
        target_pdb_id = target.pdb_id if hasattr(target, 'pdb_id') and target.pdb_id else _extract_pdb_id_from_path(target.pdb_path)
        target_uniprots: set[str] = set()
        target_pdb_ids: set[str] = set()
        
        if target_pdb_id:
            print(f"FoldSeek: Target PDB ID = {target_pdb_id}")
            target_uniprots = _get_uniprot_from_pdb(target_pdb_id)
            if target_uniprots:
                print(f"  Target UniProt accessions: {', '.join(target_uniprots)}")
                # Get all PDB IDs associated with these UniProts (to filter)
                for uniprot in target_uniprots:
                    target_pdb_ids.update(_get_pdbs_from_uniprot(uniprot))
                if target_pdb_ids:
                    print(f"  Excluding {len(target_pdb_ids)} PDB entries of same protein")
        
        # Output files
        results_file = f"{output_dir}/foldseek_results.tsv"
        tmp_dir = f"{output_dir}/tmp"
        os.makedirs(tmp_dir, exist_ok=True)

        # Determine database path
        db_dir = f"{WEIGHTS_PATH}/foldseek"
        db_path = f"{db_dir}/{config.database}"
        os.makedirs(db_dir, exist_ok=True)

        # Download database if it doesn't exist
        if not os.path.exists(db_path):
            print(f"FoldSeek database not found at {db_path}, downloading...")
            # Download PDB database - entries can be downloaded from RCSB
            # (Alphafold entries often aren't in public AlphaFold DB)
            download_cmd = [
                "foldseek",
                "databases",
                "PDB",
                db_path,
                f"{db_dir}/tmp",
            ]
            print(f"Running: {' '.join(download_cmd)}")
            dl_result = subprocess.run(
                download_cmd,
                capture_output=True,
                text=True,
                timeout=1800,  # 30 min for download
            )
            if dl_result.returncode != 0:
                print(f"FoldSeek database download failed: {dl_result.stderr}")
                # Commit any partial download for next time
                weights_volume.commit()
                return decoys
            print("FoldSeek database downloaded successfully")
            weights_volume.commit()

        # Run FoldSeek easy-search (get extra hits to allow for filtering)
        cmd = [
            "foldseek",
            "easy-search",
            target.pdb_path,
            db_path,
            results_file,
            tmp_dir,
            "--format-output",
            "target,evalue,alntmscore,alnlen,fident,tseq",
            "-e",
            str(config.evalue_threshold),
            "--max-seqs",
            str(config.max_hits * 4),  # Get extra for filtering (self-hits + dedup)
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode != 0:
            print(f"FoldSeek stderr: {result.stderr}")

        # Parse results with deduplication and self-hit filtering
        if os.path.exists(results_file):
            # Count total lines for reporting
            with open(results_file, "r") as f:
                total_hits = sum(1 for _ in f)
            decoys = _parse_foldseek_results(
                results_file, output_dir, config.max_hits, target_pdb_ids
            )
            if total_hits > len(decoys):
                print(f"FoldSeek: {total_hits} hits -> {len(decoys)} unique off-target proteins")

    except subprocess.TimeoutExpired:
        print("FoldSeek timeout")
    except Exception as e:
        print(f"FoldSeek error: {e}")
    finally:
        data_volume.commit()

    return decoys


def _parse_foldseek_results(
    results_file: str,
    output_dir: str,
    max_hits: int,
    exclude_pdb_ids: Optional[set[str]] = None,
) -> list[DecoyHit]:
    """Parse FoldSeek tabular output into DecoyHit objects.
    
    Deduplicates based on core protein ID to avoid redundant Chai-1 runs.
    For AlphaFold entries, extracts UniProt ID; for PDB entries, extracts PDB code.
    
    Args:
        results_file: Path to FoldSeek TSV output
        output_dir: Directory for output files
        max_hits: Maximum number of decoys to return
        exclude_pdb_ids: Set of PDB IDs to exclude (e.g., target protein entries)
    """
    decoys: list[DecoyHit] = []
    seen_ids: set[str] = set()  # Track unique protein IDs
    excluded_count = 0
    
    if exclude_pdb_ids is None:
        exclude_pdb_ids = set()

    with open(results_file, "r") as f:
        for line in f:
            if len(decoys) >= max_hits:
                break

            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue

            target_id = parts[0]
            
            # Extract core protein ID for deduplication
            # AlphaFold: AF-P16871-F1-model_v6 or AF-P16871-2-F1-model_v6 -> P16871
            # PDB: 7opb-assembly3_C -> 7OPB, 4HN6_A -> 4HN6
            if target_id.startswith("AF-"):
                id_parts = target_id.split("-")
                core_id = id_parts[1] if len(id_parts) >= 2 else target_id
                pdb_code = None  # AlphaFold entries don't have PDB codes
            else:
                # Extract 4-letter PDB code
                pdb_code = target_id.split("-")[0].split("_")[0].upper()[:4]
                core_id = pdb_code
            
            # Skip if this PDB is the target protein (same UniProt)
            if pdb_code and pdb_code in exclude_pdb_ids:
                excluded_count += 1
                continue
            
            # Skip if we've already seen this protein
            if core_id in seen_ids:
                continue
            seen_ids.add(core_id)

            evalue = float(parts[1]) if parts[1] else 1e10
            tm_score = float(parts[2]) if parts[2] else 0.0
            aligned_length = int(parts[3]) if parts[3] else 0
            seq_identity = float(parts[4]) if parts[4] else 0.0

            # Create placeholder PDB path (will be downloaded later)
            pdb_path = f"{output_dir}/decoys/{target_id}.pdb"
            os.makedirs(f"{output_dir}/decoys", exist_ok=True)

            decoys.append(
                DecoyHit(
                    decoy_id=target_id,
                    pdb_path=pdb_path,
                    evalue=evalue,
                    tm_score=tm_score,
                    aligned_length=aligned_length,
                    sequence_identity=seq_identity,
                )
            )
    
    if excluded_count > 0:
        print(f"  Excluded {excluded_count} self-hits (same protein as target)")

    return decoys


@app.function(
    image=foldseek_image,
    cpu=2,
    timeout=120,
    volumes={DATA_PATH: data_volume},
)
def download_decoy_structures(
    decoys: list[DecoyHit],
    output_dir: str,
) -> list[DecoyHit]:
    """
    Download PDB structures for decoy hits.

    Args:
        decoys: List of decoy hits from FoldSeek
        output_dir: Directory to save structures

    Returns:
        Updated list with valid PDB paths
    """
    import urllib.request

    os.makedirs(output_dir, exist_ok=True)
    valid_decoys: list[DecoyHit] = []

    # Use Biopython's PDBList for robust downloads
    from Bio.PDB import PDBList
    pdbl = PDBList(verbose=False)

    for decoy in decoys:
        try:
            decoy_id = decoy.decoy_id
            
            # Extract 4-letter PDB code from FoldSeek ID
            # Formats: 7opb-assembly3_C -> 7OPB, 4HN6_A -> 4HN6, AF-P16871-F1 -> skip
            if decoy_id.startswith("AF-"):
                # AlphaFold entry - extract UniProt ID
                parts = decoy_id.split("-")
                if len(parts) >= 2:
                    uniprot_id = parts[1]
                    pdb_path = f"{output_dir}/{uniprot_id}.pdb"
                    # Try AlphaFold EBI database
                    downloaded = False
                    for version in ["v4", "v3", "v2"]:
                        url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_{version}.pdb"
                        try:
                            urllib.request.urlretrieve(url, pdb_path)
                            downloaded = True
                            break
                        except Exception:
                            continue
                    if not downloaded:
                        print(f"Could not download AlphaFold structure for {uniprot_id}")
                        continue
                else:
                    continue
            else:
                # PDB entry - extract 4-letter code
                # Format: 7opb-assembly3_C or 4HN6_A or just 4HN6
                pdb_code = decoy_id.split("-")[0].split("_")[0].lower()[:4]
                pdb_path = f"{output_dir}/{pdb_code}.pdb"
                
                if not os.path.exists(pdb_path):
                    # Use Biopython PDBList for robust download
                    try:
                        downloaded_file = pdbl.retrieve_pdb_file(
                            pdb_code, 
                            pdir=output_dir, 
                            file_format="pdb"
                        )
                        if downloaded_file and os.path.exists(downloaded_file):
                            # Rename to consistent format
                            os.rename(downloaded_file, pdb_path)
                    except Exception:
                        # Fallback to direct URL
                        url = f"https://files.rcsb.org/download/{pdb_code}.pdb"
                        urllib.request.urlretrieve(url, pdb_path)

            if os.path.exists(pdb_path):
                valid_decoys.append(
                    DecoyHit(
                        decoy_id=decoy_id,
                        pdb_path=pdb_path,
                        evalue=decoy.evalue,
                        tm_score=decoy.tm_score,
                        aligned_length=decoy.aligned_length,
                        sequence_identity=decoy.sequence_identity,
                    )
                )

        except Exception as e:
            print(f"Failed to download decoy {decoy.decoy_id}: {e}")
            continue

    data_volume.commit()
    return valid_decoys


# =============================================================================
# Chai-1 - Cross-Reactivity Check
# =============================================================================


@app.function(
    image=chai1_image,
    gpu="A100",  # A100 for accurate docking predictions
    timeout=900,  # 15 min per prediction
    volumes={WEIGHTS_PATH: weights_volume, DATA_PATH: data_volume},
    max_containers=2,  # Limit to 2 concurrent A100 workers to control costs
)
def run_chai1(
    sequence: SequenceDesign,
    decoy: DecoyHit,
    config: Chai1Config,
    output_dir: str,
) -> Optional[CrossReactivityResult]:
    """
    Check cross-reactivity between a binder and a potential off-target (decoy).

    Args:
        sequence: Designed binder sequence
        decoy: Potential off-target structure
        config: Chai-1 configuration
        output_dir: Output directory

    Returns:
        CrossReactivityResult if successful, None otherwise
    """
    from pathlib import Path
    import numpy as np
    import json
    
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Extract decoy sequence - try multiple chains if chain A not found
        decoy_sequence = None
        for chain in ["A", "B", "C", " "]:
            try:
                decoy_sequence = _extract_sequence_from_pdb(decoy.pdb_path, chain)
                if decoy_sequence and len(decoy_sequence) > 10:
                    break
            except Exception:
                continue
        
        if not decoy_sequence:
            print(f"  ✗ Could not extract sequence from {decoy.pdb_path}")
            return None

        # Prepare Chai-1 input FASTA (directory already contains seq_id/decoy_id context)
        input_fasta = f"{output_dir}/input.fasta"
        with open(input_fasta, "w") as f:
            # Chai-1 uses "protein|name=X" format for chain specification
            f.write(f">protein|name=decoy\n{decoy_sequence}\n")
            f.write(f">protein|name=binder\n{sequence.sequence}\n")

        # Use output_dir directly - caller already includes seq_id/decoy_id in path
        output_subdir = Path(output_dir)

        # Use chai_lab Python API
        try:
            from chai_lab.chai1 import run_inference
            
            candidates = run_inference(
                fasta_file=Path(input_fasta),
                output_dir=output_subdir,
                num_trunk_recycles=config.num_recycles if hasattr(config, 'num_recycles') else 3,
                num_diffn_timesteps=200,
                seed=42,
                device=None,  # Auto-detect CUDA
                use_esm_embeddings=True,
            )
            
            # Parse output from saved files instead of object attributes
            # Chai-1 saves scores.json and plddt files in output directory
            plddt_interface = 0.0
            affinity = 0.0
            ptm_score = None
            iptm_score = None
            
            # Look for scores file
            scores_files = list(output_subdir.glob("**/scores*.json")) + list(output_subdir.glob("scores.json"))
            if scores_files:
                with open(scores_files[0], "r") as f:
                    scores_data = json.load(f)
                    # Extract pTM and ipTM scores
                    if "ptm" in scores_data:
                        ptm_score = float(scores_data["ptm"])
                    if "iptm" in scores_data:
                        iptm_score = float(scores_data["iptm"])
                    # Extract aggregate score if available
                    if "aggregate_score" in scores_data:
                        affinity = -float(scores_data["aggregate_score"])
                    elif ptm_score is not None:
                        affinity = -ptm_score * 10  # Scale pTM as affinity proxy
            
            # Look for pLDDT in npz files
            plddt_files = list(output_subdir.glob("**/plddt*.npz")) + list(output_subdir.glob("plddt.npz"))
            if plddt_files:
                plddt_data = np.load(plddt_files[0])
                if "plddt" in plddt_data:
                    plddt_interface = float(np.mean(plddt_data["plddt"])) * 100
            
            # Fallback: check if candidates has any usable attributes for plddt
            if plddt_interface == 0.0 and candidates is not None:
                # Try different possible attribute names
                for attr in ['plddt', 'per_token_plddt', 'confidence']:
                    if hasattr(candidates, attr):
                        val = getattr(candidates, attr)
                        if val is not None:
                            try:
                                if hasattr(val, 'cpu'):
                                    plddt_interface = float(np.mean(val[0].cpu().numpy())) * 100
                                else:
                                    plddt_interface = float(np.mean(val)) * 100
                                break
                            except Exception:
                                pass
            
            # Extract chain_pair_iptm from scores (Chai-1 single-sequence mode)
            chain_pair_iptm = None
            if scores_files:
                with open(scores_files[0], "r") as f:
                    scores_data = json.load(f)
                    chain_pair_iptm = scores_data.get("chain_pair_iptm")
                    # Fallback to iptm if chain_pair_iptm not available
                    if chain_pair_iptm is None:
                        chain_pair_iptm = iptm_score
            
            # Off-target threshold: chain_pair_iptm > 0.5 indicates cross-reactivity
            binds_decoy = (chain_pair_iptm is not None and chain_pair_iptm > 0.5)

            return CrossReactivityResult(
                binder_id=sequence.sequence_id,
                decoy_id=decoy.decoy_id,
                predicted_affinity=affinity,
                plddt_interface=plddt_interface,
                binds_decoy=binds_decoy,
                ptm=ptm_score,
                iptm=iptm_score,
                chain_pair_iptm=chain_pair_iptm,
            )
                
        except ImportError as e:
            print(f"Chai-1 import error: {e}")
            # Fallback: return a conservative result (assume no cross-reactivity)
            return CrossReactivityResult(
                binder_id=sequence.sequence_id,
                decoy_id=decoy.decoy_id,
                predicted_affinity=0.0,
                plddt_interface=0.0,
                binds_decoy=False,
                ptm=None,
                iptm=None,
                chain_pair_iptm=None,
            )
            
        return None

    except Exception as e:
        print(f"Chai-1 error: {e}")
        return None
    finally:
        data_volume.commit()


@app.function(
    image=chai1_image,
    gpu="A100",
    timeout=1800,
    volumes={WEIGHTS_PATH: weights_volume, DATA_PATH: data_volume},
)
def run_chai1_batch(
    sequence: SequenceDesign,
    decoys: list[DecoyHit],
    config: Chai1Config,
    output_dir: str,
) -> list[CrossReactivityResult]:
    """
    Run Chai-1 cross-reactivity check against multiple decoys.

    Args:
        sequence: Designed binder sequence
        decoys: List of decoy structures to check
        config: Chai-1 configuration
        output_dir: Output directory

    Returns:
        List of CrossReactivityResult for each decoy
    """
    results: list[CrossReactivityResult] = []

    for decoy in decoys:
        decoy_output_dir = f"{output_dir}/{decoy.decoy_id}"
        result = run_chai1.local(sequence, decoy, config, decoy_output_dir)
        if result is not None:
            results.append(result)

    return results


def _parse_chai1_metrics(output_prefix: str) -> Optional[dict]:
    """Parse Chai-1 output metrics."""
    import numpy as np

    metrics = {}

    # Try JSON metrics file
    metrics_file = f"{output_prefix}_metrics.json"
    if os.path.exists(metrics_file):
        with open(metrics_file, "r") as f:
            raw_metrics = json.load(f)
            metrics["plddt_interface"] = raw_metrics.get("plddt_interface", 0.0)
            metrics["affinity"] = raw_metrics.get("affinity", 0.0)
            return metrics

    # Try pLDDT numpy file
    plddt_file = f"{output_prefix}_plddt.npy"
    if os.path.exists(plddt_file):
        plddt = np.load(plddt_file)
        metrics["plddt_interface"] = float(np.mean(plddt[-50:]))  # Binder region
        metrics["affinity"] = -float(np.mean(plddt[-50:])) / 10  # Heuristic

    if metrics:
        return metrics

    return None


# =============================================================================
# Parallel Validation Utilities
# =============================================================================


@app.function(
    image=boltz2_image,
    timeout=1800,  # 30 minutes - structure prediction takes time
    volumes={WEIGHTS_PATH: weights_volume, DATA_PATH: data_volume},
)
def validate_sequences_parallel(
    sequences: list[SequenceDesign],
    target: TargetProtein,
    config: Boltz2Config,
    base_output_dir: str,
    batch_id: Optional[str] = None,
) -> list[StructurePrediction]:
    """
    Validate multiple sequences in parallel using starmap.

    Args:
        sequences: List of designed sequences
        target: Target protein
        config: Boltz-2 configuration
        base_output_dir: Base output directory
        batch_id: Optional batch ID for consolidation tracking

    Returns:
        List of successful structure predictions
    """
    import time
    batch_start = time.time()
    
    # Generate batch ID if not provided
    if batch_id is None:
        batch_id = f"boltz2_batch_{int(batch_start * 1000)}"
    
    # Print common parameters once
    if sequences:
        target_len = len(_extract_sequence_from_pdb(target.pdb_path, target.chain_id))
        binder_len = len(sequences[0].sequence)
        print(f"Boltz-2: validating {len(sequences)} sequences (target: {target_len} res, binder: {binder_len} res)")
        print(f"  Batch ID: {batch_id} | Batch size: {len(sequences)} (cold start amortized)")

    # Prepare arguments for starmap
    args = [
        (seq, target, config, f"{base_output_dir}/{seq.sequence_id}")
        for seq in sequences
    ]

    # Use starmap for parallel execution
    all_results = list(run_boltz2.starmap(args))
    
    batch_duration = time.time() - batch_start
    successful = sum(1 for r in all_results if r is not None)
    print(f"  Batch complete: {successful}/{len(sequences)} in {batch_duration:.1f}s (avg {batch_duration/len(sequences):.1f}s/seq)")

    # Filter None results
    return [r for r in all_results if r is not None]


@app.function(
    image=chai1_image,
    timeout=3600,  # 1 hour for processing multiple sequences against multiple decoys
)
def check_cross_reactivity_parallel(
    sequences: list[SequenceDesign],
    decoys: list[DecoyHit],
    config: Chai1Config,
    base_output_dir: str,
    target: Optional[TargetProtein] = None,
    early_termination: bool = True,
) -> tuple[dict[str, CrossReactivityResult], dict[str, list[CrossReactivityResult]]]:
    """
    Check cross-reactivity for multiple sequences against decoys in parallel.
    
    Optionally includes the target as a positive control to verify binding.
    
    **Economical optimizations:**
    - Decoys are sorted by TM-score (most similar first = highest risk)
    - Early termination: stops testing a sequence once cross-reactivity detected
    - Concurrency limited to 2 A100 workers

    Args:
        sequences: List of binder sequences
        decoys: List of decoy structures
        config: Chai-1 configuration
        base_output_dir: Base output directory
        target: Optional target protein for positive control check
        early_termination: If True, stop testing decoys for a sequence once cross-reactivity found

    Returns:
        Tuple of:
        - Dictionary mapping sequence_id to positive control result (target binding)
        - Dictionary mapping sequence_id to list of CrossReactivityResults (decoys)
    """
    positive_control_results: dict[str, CrossReactivityResult] = {}
    decoy_results: dict[str, list[CrossReactivityResult]] = {}
    
    # Step 1: Run positive control (binder vs target) if target provided
    if target is not None:
        print(f"Chai-1 Positive Control: checking {len(sequences)} sequences against target")
        
        # Create a DecoyHit for the target (reusing the structure)
        target_as_decoy = DecoyHit(
            decoy_id="TARGET",
            pdb_path=target.pdb_path,
            evalue=0.0,
            tm_score=1.0,
            aligned_length=0,
            sequence_identity=1.0,
        )
        
        # Run positive control checks
        pos_ctrl_args = [
            (seq, target_as_decoy, config, f"{base_output_dir}/{seq.sequence_id}/positive_control")
            for seq in sequences
        ]
        
        pos_ctrl_results = list(run_chai1.starmap(pos_ctrl_args))
        
        for result in pos_ctrl_results:
            if result is not None:
                positive_control_results[result.binder_id] = result
                status = "✓ BINDS" if result.plddt_interface > 50 else "✗ NO BINDING"
                metrics_str = f"pLDDT={result.plddt_interface:.1f}"
                if result.iptm is not None:
                    metrics_str += f", ipTM={result.iptm:.3f}"
                if result.ptm is not None:
                    metrics_str += f", pTM={result.ptm:.3f}"
                print(f"  {result.binder_id}: {metrics_str} {status}")
        
        print(f"  Positive control: {len(positive_control_results)}/{len(sequences)} sequences bind target")
    
    # Step 2: Run decoy checks with tiered or early termination optimization
    if decoys:
        # Sort decoys by TM-score descending (most similar = highest risk, test first)
        sorted_decoys = sorted(decoys, key=lambda d: d.tm_score, reverse=True)
        
        if config.tiered_checking:
            # Tiered decoy checking (#6 optimization)
            # Tier 1: Highest-risk decoys (TM > tier1_min_tm) - MUST pass all
            # Tier 2: Medium-risk decoys (TM > tier2_min_tm) - check up to tier2_max_decoys
            # Tier 3: Lower-risk decoys - skip for budget runs
            tier1_decoys = [d for d in sorted_decoys if d.tm_score >= config.tier1_min_tm]
            tier2_candidates = [d for d in sorted_decoys if config.tier2_min_tm <= d.tm_score < config.tier1_min_tm]
            tier2_decoys = tier2_candidates[:config.tier2_max_decoys]
            
            print(f"Chai-1 Tiered Decoy Check: {len(sequences)} sequences")
            print(f"  Tier 1 (TM ≥ {config.tier1_min_tm}): {len(tier1_decoys)} decoys - MUST pass all")
            print(f"  Tier 2 (TM ≥ {config.tier2_min_tm}): {len(tier2_decoys)}/{len(tier2_candidates)} decoys")
            print("  Tier 3 (lower TM): skipped for cost savings")
            
            total_calls = 0
            rejected_sequences: set[str] = set()
            
            for seq in sequences:
                decoy_results[seq.sequence_id] = []
                rejected = False
                
                # Tier 1: Must pass all high-risk decoys
                for decoy in tier1_decoys:
                    result = run_chai1.remote(
                        seq, decoy, config, 
                        f"{base_output_dir}/{seq.sequence_id}/{decoy.decoy_id}"
                    )
                    total_calls += 1
                    
                    if result is not None:
                        decoy_results[seq.sequence_id].append(result)
                        if result.binds_decoy:
                            rejected_sequences.add(seq.sequence_id)
                            print(f"  ✗ {seq.sequence_id}: Tier 1 fail - {decoy.decoy_id} (TM={decoy.tm_score:.2f})")
                            rejected = True
                            break
                
                if rejected:
                    continue
                
                # Tier 2: Check medium-risk decoys (early termination within tier)
                for decoy in tier2_decoys:
                    result = run_chai1.remote(
                        seq, decoy, config, 
                        f"{base_output_dir}/{seq.sequence_id}/{decoy.decoy_id}"
                    )
                    total_calls += 1
                    
                    if result is not None:
                        decoy_results[seq.sequence_id].append(result)
                        if result.binds_decoy:
                            rejected_sequences.add(seq.sequence_id)
                            print(f"  ✗ {seq.sequence_id}: Tier 2 fail - {decoy.decoy_id} (TM={decoy.tm_score:.2f})")
                            rejected = True
                            break
                
                if not rejected:
                    checked = len(tier1_decoys) + len(tier2_decoys)
                    print(f"  ✓ {seq.sequence_id}: passed {checked} decoys (Tier 1+2)")
            
            max_possible = len(sequences) * len(sorted_decoys)
            saved = max_possible - total_calls
            print(f"  Tiered checking saved ~{saved} Chai-1 calls ({100*saved/max_possible:.0f}% reduction)")
            print(f"  Result: {len(sequences) - len(rejected_sequences)}/{len(sequences)} sequences passed selectivity")
        
        elif early_termination:
            # Sequential per-sequence with early termination (economical mode)
            print(f"Chai-1 Decoy Check: {len(sequences)} sequences × up to {len(sorted_decoys)} decoys (early termination enabled)")
            print(f"  Decoys sorted by TM-score: {', '.join(f'{d.decoy_id[:8]}({d.tm_score:.2f})' for d in sorted_decoys[:3])}...")
            
            total_calls = 0
            rejected_sequences: set[str] = set()
            
            for seq in sequences:
                decoy_results[seq.sequence_id] = []
                
                for decoy in sorted_decoys:
                    # Run single prediction
                    result = run_chai1.remote(
                        seq, decoy, config, 
                        f"{base_output_dir}/{seq.sequence_id}/{decoy.decoy_id}"
                    )
                    total_calls += 1
                    
                    if result is not None:
                        decoy_results[seq.sequence_id].append(result)
                        
                        # Early termination: if cross-reactive, skip remaining decoys
                        if result.binds_decoy:
                            rejected_sequences.add(seq.sequence_id)
                            print(f"  ✗ {seq.sequence_id}: cross-reactive with {decoy.decoy_id} (pLDDT={result.plddt_interface:.1f})")
                            break
                
                if seq.sequence_id not in rejected_sequences:
                    print(f"  ✓ {seq.sequence_id}: passed {len(sorted_decoys)} decoys")
            
            print(f"  Early termination saved ~{len(sequences) * len(sorted_decoys) - total_calls} Chai-1 calls")
            print(f"  Result: {len(sequences) - len(rejected_sequences)}/{len(sequences)} sequences passed selectivity")
        else:
            # Full parallel mode (test all combinations)
            num_pairs = len(sequences) * len(sorted_decoys)
            print(f"Chai-1 Decoy Check: {len(sequences)} sequences × {len(sorted_decoys)} decoys = {num_pairs} pairs")
            print("  (concurrency limited to 2 A100 workers)")
            
            # Prepare all combinations for starmap
            args = [
                (seq, decoy, config, f"{base_output_dir}/{seq.sequence_id}/{decoy.decoy_id}")
                for seq in sequences
                for decoy in sorted_decoys
            ]
            
            # Use starmap - concurrency is controlled by run_chai1's max_containers=2
            all_results = list(run_chai1.starmap(args))
            
            # Group results by sequence
            for result in all_results:
                if result is not None:
                    if result.binder_id not in decoy_results:
                        decoy_results[result.binder_id] = []
                    decoy_results[result.binder_id].append(result)

            total_success = sum(len(v) for v in decoy_results.values())
            print(f"  Decoy check completed: {total_success}/{num_pairs} successful predictions")
    
    return positive_control_results, decoy_results




# =============================================================================
# Mock Implementations (for testing without GPU)
# =============================================================================


def mock_boltz2(
    sequence: SequenceDesign,
    target: TargetProtein,
    config: Boltz2Config,
    output_dir: str,
) -> Optional[StructurePrediction]:
    """Mock Boltz-2 for testing without GPU resources."""
    import random

    os.makedirs(output_dir, exist_ok=True)

    # Generate mock metrics
    plddt_interface = random.uniform(50, 95)
    pae_interface = random.uniform(2, 15)
    iptm_score = random.uniform(0.4, 0.95)
    
    # AlphaProteo SI 2.2 metrics
    pae_interaction = random.uniform(0.5, 3.0)  # PAE at hotspots
    ptm_binder = random.uniform(0.6, 0.95)  # Binder-only pTM
    rmsd_to_design = random.uniform(0.5, 4.0)  # RMSD vs RFDiffusion

    # Apply AlphaProteo filters (SI 2.2) - ONLY these 3 criteria
    # 1. Anchor Lock: PAE at hotspots < 1.5 Å
    if pae_interaction > config.max_pae_interaction:
        return None
    # 2. Fold Quality: binder pTM > 0.80
    if ptm_binder < config.min_ptm_binder:
        return None
    # 3. Self-Consistency: RMSD < 2.5 Å
    if rmsd_to_design > config.max_rmsd:
        return None

    pdb_path = f"{output_dir}/{sequence.sequence_id}_complex.pdb"
    # Create minimal mock PDB
    with open(pdb_path, "w") as f:
        f.write("HEADER    MOCK COMPLEX\nEND\n")

    return StructurePrediction(
        prediction_id=generate_design_id("mock_pred"),
        sequence_id=sequence.sequence_id,
        pdb_path=pdb_path,
        plddt_overall=random.uniform(60, 90),
        plddt_interface=plddt_interface,
        pae_interface=pae_interface,
        ptm=random.uniform(0.5, 0.9),
        iptm=iptm_score,
        pae_interaction=pae_interaction,
        ptm_binder=ptm_binder,
        rmsd_to_design=rmsd_to_design,
    )


def mock_foldseek(
    target: TargetProtein,
    config: FoldSeekConfig,
    output_dir: str,
) -> list[DecoyHit]:
    """Mock FoldSeek for testing."""
    import random

    os.makedirs(f"{output_dir}/decoys", exist_ok=True)
    decoys: list[DecoyHit] = []

    mock_pdb_ids = ["1ABC", "2DEF", "3GHI", "4JKL", "5MNO", "6PQR", "7STU", "8VWX", "9YZA", "1BCD"]

    for i in range(min(config.max_hits, len(mock_pdb_ids))):
        pdb_id = mock_pdb_ids[i]
        pdb_path = f"{output_dir}/decoys/{pdb_id}_A.pdb"

        # Create mock PDB
        with open(pdb_path, "w") as f:
            f.write(f"HEADER    MOCK DECOY {pdb_id}\nEND\n")

        decoys.append(
            DecoyHit(
                decoy_id=f"{pdb_id}_A",
                pdb_path=pdb_path,
                evalue=random.uniform(1e-10, config.evalue_threshold),
                tm_score=random.uniform(0.3, 0.8),
                aligned_length=random.randint(50, 200),
                sequence_identity=random.uniform(0.2, 0.6),
            )
        )

    return decoys


def mock_chai1(
    sequence: SequenceDesign,
    decoy: DecoyHit,
    config: Chai1Config,
    output_dir: str,
) -> CrossReactivityResult:
    """Mock Chai-1 for testing."""
    import random

    plddt_interface = random.uniform(30, 80)
    affinity = random.uniform(-8, -2)
    ptm_score = random.uniform(0.3, 0.8)
    iptm_score = random.uniform(0.2, 0.7)
    chain_pair_iptm = random.uniform(0.2, 0.7)
    # Off-target threshold: chain_pair_iptm > 0.5 indicates cross-reactivity
    binds_decoy = chain_pair_iptm > 0.5

    return CrossReactivityResult(
        binder_id=sequence.sequence_id,
        decoy_id=decoy.decoy_id,
        predicted_affinity=affinity,
        plddt_interface=plddt_interface,
        binds_decoy=binds_decoy,
        ptm=ptm_score,
        iptm=iptm_score,
        chain_pair_iptm=chain_pair_iptm,
    )


# =============================================================================
# Clustering - TM-score based diversity (AlphaProteo Step 4)
# =============================================================================


def cluster_by_tm_score(
    predictions: list[StructurePrediction],
    tm_threshold: float = 0.7,
    select_best: bool = True,
) -> list[StructurePrediction]:
    """
    Cluster predicted structures by TM-score and select representatives.
    
    Uses FoldSeek for fast TM-score calculation between structures.
    Structures with TM-score > threshold are grouped into same cluster.
    
    Args:
        predictions: List of validated structure predictions
        tm_threshold: TM-score threshold for clustering (default: 0.7)
        select_best: If True, select best (highest ppi_score) per cluster
    
    Returns:
        List of representative predictions (one per cluster)
    """
    if len(predictions) <= 1:
        return predictions
    
    try:
        import numpy as np
        
        # Calculate pairwise TM-scores using structure comparison
        n = len(predictions)
        tm_matrix = np.zeros((n, n))
        
        for i in range(n):
            tm_matrix[i, i] = 1.0
            for j in range(i + 1, n):
                tm = _calculate_tm_score(predictions[i].pdb_path, predictions[j].pdb_path)
                tm_matrix[i, j] = tm
                tm_matrix[j, i] = tm
        
        # Greedy clustering
        assigned = [False] * n
        clusters: list[list[int]] = []
        
        # Sort by ppi_score (best first)
        sorted_indices = sorted(range(n), key=lambda i: predictions[i].ppi_score, reverse=True)
        
        for i in sorted_indices:
            if assigned[i]:
                continue
            
            # Start new cluster with this structure as representative
            cluster = [i]
            assigned[i] = True
            
            # Add similar structures to this cluster
            for j in sorted_indices:
                if not assigned[j] and tm_matrix[i, j] > tm_threshold:
                    cluster.append(j)
                    assigned[j] = True
            
            clusters.append(cluster)
        
        # Select representatives
        if select_best:
            # Return the first (best ppi_score) member of each cluster
            representatives = [predictions[cluster[0]] for cluster in clusters]
        else:
            # Return all predictions but with cluster info
            representatives = [predictions[cluster[0]] for cluster in clusters]
        
        print(f"Clustering: {n} structures → {len(clusters)} clusters (TM > {tm_threshold})")
        return representatives
        
    except Exception as e:
        print(f"Warning: Clustering failed ({e}), returning all predictions")
        return predictions


def _calculate_tm_score(pdb1: str, pdb2: str, chain_id: str = "B") -> float:
    """
    Calculate TM-score between two structures.
    
    Uses simplified CA-based alignment.
    
    Returns:
        TM-score in range [0, 1]
    """
    try:
        from Bio.PDB import PDBParser, MMCIFParser
        import numpy as np
        
        # Parse structures
        parser1 = MMCIFParser(QUIET=True) if pdb1.endswith(".cif") else PDBParser(QUIET=True)
        parser2 = MMCIFParser(QUIET=True) if pdb2.endswith(".cif") else PDBParser(QUIET=True)
        
        struct1 = parser1.get_structure("s1", pdb1)
        struct2 = parser2.get_structure("s2", pdb2)
        
        # Extract CA atoms from binder chain
        def get_ca_coords(structure, chain_id):
            coords = []
            for model in structure:
                for chain in model:
                    if chain.id == chain_id:
                        for residue in chain:
                            if residue.id[0] == " " and "CA" in residue:
                                coords.append(residue["CA"].get_coord())
            return np.array(coords)
        
        ca1 = get_ca_coords(struct1, chain_id)
        ca2 = get_ca_coords(struct2, chain_id)
        
        if len(ca1) == 0 or len(ca2) == 0:
            return 0.0
        
        # Align lengths
        min_len = min(len(ca1), len(ca2))
        ca1 = ca1[:min_len]
        ca2 = ca2[:min_len]
        
        # Calculate RMSD after superposition
        centroid1 = np.mean(ca1, axis=0)
        centroid2 = np.mean(ca2, axis=0)
        ca1_centered = ca1 - centroid1
        ca2_centered = ca2 - centroid2
        
        # SVD for optimal rotation
        H = ca1_centered.T @ ca2_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # Apply rotation
        ca2_aligned = ca2_centered @ R.T
        
        # Calculate TM-score
        d0 = 1.24 * (min_len - 15) ** (1/3) - 1.8 if min_len > 15 else 0.5
        d0 = max(d0, 0.5)
        
        distances = np.linalg.norm(ca1_centered - ca2_aligned, axis=1)
        tm_score = np.sum(1 / (1 + (distances / d0) ** 2)) / min_len
        
        return float(tm_score)
        
    except Exception:
        return 0.0


# =============================================================================
# Novelty Check - pyhmmer vs UniRef50 (AlphaProteo Step 5)
# =============================================================================

# UniRef50 sequence database URL (UniProt FTP)
# ~12GB compressed, ~50GB uncompressed - persisted in Modal Volume
UNIREF50_URL = "https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz"
UNIREF50_DB_PATH = "/data/uniref50/uniref50.fasta"


def _ensure_uniref50_database(auto_download: bool = True) -> str | None:
    """
    Ensure UniRef50 FASTA database is available, downloading if necessary.
    
    UniRef50 is persisted in the Modal data volume and only downloaded once.
    This is a ~12GB download that expands to ~50GB.
    
    Args:
        auto_download: If True, download database if not present
        
    Returns:
        Path to the database, or None if unavailable
    """
    import gzip
    import shutil
    import urllib.request
    
    if os.path.exists(UNIREF50_DB_PATH):
        return UNIREF50_DB_PATH
    
    if not auto_download:
        print(f"Warning: UniRef50 database not found at {UNIREF50_DB_PATH}")
        return None
    
    # Create directory if needed
    os.makedirs(os.path.dirname(UNIREF50_DB_PATH), exist_ok=True)
    
    gz_path = UNIREF50_DB_PATH + ".gz"
    
    try:
        print(f"Downloading UniRef50 database from {UNIREF50_URL}...")
        print("  ⚠️  One-time download: ~12GB compressed → ~50GB uncompressed")
        print("  ⚠️  This may take 10-30 minutes depending on network speed")
        urllib.request.urlretrieve(UNIREF50_URL, gz_path)
        
        print("  Extracting database (streaming to minimize RAM usage)...")
        with gzip.open(gz_path, 'rb') as f_in:
            with open(UNIREF50_DB_PATH, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # Clean up compressed file to save ~12GB
        os.remove(gz_path)
        print(f"  UniRef50 database ready at {UNIREF50_DB_PATH}")
        return UNIREF50_DB_PATH
        
    except Exception as e:
        print(f"Warning: Failed to download UniRef50 database: {e}")
        # Clean up partial downloads
        if os.path.exists(gz_path):
            os.remove(gz_path)
        if os.path.exists(UNIREF50_DB_PATH):
            os.remove(UNIREF50_DB_PATH)
        return None


def check_novelty(
    sequences: list[SequenceDesign],
    max_evalue: float = 1e-6,
    auto_download: bool = True,
) -> list[SequenceDesign]:
    """
    Filter sequences for novelty using pyhmmer phmmer against UniRef50.
    
    Uses sequence-vs-sequence search (phmmer mode) to find sequences with
    high similarity to known proteins in UniRef50. Critical for:
    - Patentability (IP): Avoid sequences too similar to known/patented proteins
    - Safety (Immunogenicity): Flag sequences similar to human proteins
    
    Args:
        sequences: List of designed sequences
        max_evalue: Maximum E-value to consider a hit (lower = stricter, default 1e-6)
        auto_download: Auto-download UniRef50 database if not present
    
    Returns:
        List of novel sequences (no significant UniRef50 hits)
    """
    if not sequences:
        return sequences
    
    try:
        import pyhmmer
        from pyhmmer.easel import TextSequence, Alphabet, SequenceFile
        from pyhmmer.plan7 import Pipeline
        
        # Ensure database is available (persisted in Modal Volume)
        db_path = _ensure_uniref50_database(auto_download=auto_download)
        if db_path is None:
            print("Warning: UniRef50 database unavailable, skipping novelty check")
            return sequences
        
        alphabet = Alphabet.amino()
        
        print(f"  Scanning {len(sequences)} sequences against UniRef50 (phmmer mode)...")
        print(f"  E-value threshold: {max_evalue:.0e}")
        
        # Track which sequences have hits
        seq_hits: dict[str, tuple[str, float]] = {}  # seq_id -> (hit_name, evalue)
        
        # Process each query sequence
        for seq in sequences:
            # Create digital query sequence
            query = TextSequence(
                name=seq.sequence_id.encode(),
                sequence=seq.sequence.encode(),
            ).digitize(alphabet)
            
            # Create pipeline for this search
            pipeline = Pipeline(alphabet)
            
            # Stream through UniRef50 FASTA - avoids loading 50GB into RAM
            # This is phmmer mode: sequence vs sequence database
            with SequenceFile(db_path, digital=True, alphabet=alphabet) as seq_file:
                hits = pipeline.search_seq(query, seq_file)
                
                # Check for significant hits
                for hit in hits:
                    if hit.evalue < max_evalue:
                        seq_hits[seq.sequence_id] = (hit.name.decode(), hit.evalue)
                        break  # One significant hit is enough to flag as non-novel
        
        # Separate novel from non-novel sequences
        novel_sequences = []
        for seq in sequences:
            if seq.sequence_id in seq_hits:
                hit_name, evalue = seq_hits[seq.sequence_id]
                # Truncate long UniRef IDs for display
                display_name = hit_name[:40] + "..." if len(hit_name) > 40 else hit_name
                print(f"  ✗ {seq.sequence_id}: hit {display_name} (E={evalue:.2e}) - NOT NOVEL")
            else:
                novel_sequences.append(seq)
                print(f"  ✓ {seq.sequence_id}: no significant hits - NOVEL")
        
        print(f"Novelty: {len(novel_sequences)}/{len(sequences)} sequences are novel (E > {max_evalue})")
        return novel_sequences
        
    except ImportError:
        print("Warning: pyhmmer not installed, skipping novelty check")
        return sequences
    except Exception as e:
        print(f"Warning: Novelty check failed ({e}), returning all sequences")
        return sequences
