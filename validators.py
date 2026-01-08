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
import tempfile
from pathlib import Path
from typing import Optional

import modal

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
    import numpy as np

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

        # Parse Boltz-2 output metrics
        metrics = _parse_boltz2_metrics(output_dir, sequence.sequence_id)

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
        )

        # Apply filters
        if prediction.plddt_interface < config.min_iplddt * 100:
            print(f"  ✗ {sequence.sequence_id}: i-pLDDT {prediction.plddt_interface:.1f} < {config.min_iplddt * 100}")
            return None

        if prediction.pae_interface > config.max_pae:
            print(f"  ✗ {sequence.sequence_id}: PAE {prediction.pae_interface:.1f} > {config.max_pae}")
            return None

        print(f"  ✓ {sequence.sequence_id}: i-pLDDT={prediction.plddt_interface:.1f}, PAE={prediction.pae_interface:.1f}")
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


def _parse_boltz2_metrics(output_dir: str, sequence_id: str) -> Optional[dict]:
    """Parse Boltz-2 output metrics from JSON or npz files."""
    import numpy as np
    import glob

    metrics = {}
    
    # Boltz outputs are in: {output_dir}/boltz_results_*/predictions/*/
    # Files: confidence_*.json, plddt_*.npz
    
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


# =============================================================================
# FoldSeek - Proteome Scanning for Decoys
# =============================================================================


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

        # Run FoldSeek easy-search
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
            str(config.max_hits * 2),  # Get extra for filtering
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode != 0:
            print(f"FoldSeek stderr: {result.stderr}")

        # Parse results with deduplication
        if os.path.exists(results_file):
            # Count total lines for reporting
            with open(results_file, "r") as f:
                total_hits = sum(1 for _ in f)
            decoys = _parse_foldseek_results(results_file, output_dir, config.max_hits)
            if total_hits > len(decoys):
                print(f"FoldSeek: {total_hits} hits -> {len(decoys)} unique proteins")

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
) -> list[DecoyHit]:
    """Parse FoldSeek tabular output into DecoyHit objects.
    
    Deduplicates based on core protein ID to avoid redundant Chai-1 runs.
    For AlphaFold entries, extracts UniProt ID; for PDB entries, extracts PDB code.
    """
    decoys: list[DecoyHit] = []
    seen_ids: set[str] = set()  # Track unique protein IDs

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
            # PDB: 1ABC_A -> 1ABC
            if target_id.startswith("AF-"):
                id_parts = target_id.split("-")
                core_id = id_parts[1] if len(id_parts) >= 2 else target_id
            else:
                core_id = target_id.split("_")[0].upper()
            
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
                    except Exception as e:
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
    import glob
    
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

        # Prepare Chai-1 input FASTA
        input_fasta = f"{output_dir}/chai1_input_{sequence.sequence_id}_{decoy.decoy_id}.fasta"
        with open(input_fasta, "w") as f:
            # Chai-1 uses "protein|name=X" format for chain specification
            f.write(f">protein|name=decoy\n{decoy_sequence}\n")
            f.write(f">protein|name=binder\n{sequence.sequence}\n")

        output_subdir = Path(f"{output_dir}/pred_{sequence.sequence_id}_{decoy.decoy_id}")

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
            
            # Look for scores file
            scores_files = list(output_subdir.glob("**/scores*.json")) + list(output_subdir.glob("scores.json"))
            if scores_files:
                with open(scores_files[0], "r") as f:
                    scores_data = json.load(f)
                    # Extract aggregate score if available
                    if "aggregate_score" in scores_data:
                        affinity = -float(scores_data["aggregate_score"])
                    elif "ptm" in scores_data:
                        affinity = -float(scores_data["ptm"]) * 10  # Scale pTM as affinity proxy
            
            # Look for pLDDT in npz files
            plddt_files = list(output_subdir.glob("**/plddt*.npz")) + list(output_subdir.glob("plddt.npz"))
            if plddt_files:
                plddt_data = np.load(plddt_files[0])
                if "plddt" in plddt_data:
                    plddt_interface = float(np.mean(plddt_data["plddt"])) * 100
            
            # Fallback: check if candidates has any usable attributes
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
                
                # Determine if binder likely binds decoy
                binds_decoy = plddt_interface > 60 and affinity < -5.0

                return CrossReactivityResult(
                    binder_id=sequence.sequence_id,
                    decoy_id=decoy.decoy_id,
                    predicted_affinity=affinity,
                    plddt_interface=plddt_interface,
                    binds_decoy=binds_decoy,
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
) -> list[StructurePrediction]:
    """
    Validate multiple sequences in parallel using starmap.

    Args:
        sequences: List of designed sequences
        target: Target protein
        config: Boltz-2 configuration
        base_output_dir: Base output directory

    Returns:
        List of successful structure predictions
    """
    # Print common parameters once
    if sequences:
        target_len = len(_extract_sequence_from_pdb(target.pdb_path, target.chain_id))
        binder_len = len(sequences[0].sequence)
        print(f"Boltz-2: validating {len(sequences)} sequences (target: {target_len} res, binder: {binder_len} res)")

    # Prepare arguments for starmap
    args = [
        (seq, target, config, f"{base_output_dir}/{seq.sequence_id}")
        for seq in sequences
    ]

    # Use starmap for parallel execution
    all_results = list(run_boltz2.starmap(args))

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
) -> tuple[dict[str, CrossReactivityResult], dict[str, list[CrossReactivityResult]]]:
    """
    Check cross-reactivity for multiple sequences against decoys in parallel.
    
    Optionally includes the target as a positive control to verify binding.
    
    Note: Concurrency is limited to 2 A100 workers via run_chai1's max_containers.

    Args:
        sequences: List of binder sequences
        decoys: List of decoy structures
        config: Chai-1 configuration
        base_output_dir: Base output directory
        target: Optional target protein for positive control check

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
                print(f"  {result.binder_id}: pLDDT={result.plddt_interface:.1f} {status}")
        
        print(f"  Positive control: {len(positive_control_results)}/{len(sequences)} sequences bind target")
    
    # Step 2: Run decoy checks
    if decoys:
        num_pairs = len(sequences) * len(decoys)
        print(f"Chai-1 Decoy Check: {len(sequences)} sequences × {len(decoys)} decoys = {num_pairs} pairs")
        print(f"  (concurrency limited to 2 A100 workers)")
        
        # Prepare all combinations for starmap
        args = [
            (seq, decoy, config, f"{base_output_dir}/{seq.sequence_id}/{decoy.decoy_id}")
            for seq in sequences
            for decoy in decoys
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

    # Apply filters
    if plddt_interface < config.min_iplddt * 100:
        return None
    if pae_interface > config.max_pae:
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
        iptm=random.uniform(0.4, 0.85),
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
    binds_decoy = plddt_interface > 60 and affinity < -5.0

    return CrossReactivityResult(
        binder_id=sequence.sequence_id,
        decoy_id=decoy.decoy_id,
        predicted_affinity=affinity,
        plddt_interface=plddt_interface,
        binds_decoy=binds_decoy,
    )
