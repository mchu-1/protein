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
    APP_NAME,
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
    boltz2_image,
    chai1_image,
    data_volume,
    foldseek_image,
    generate_design_id,
    weights_volume,
)

# =============================================================================
# Modal App Definition
# =============================================================================

app = modal.App(APP_NAME)

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

        # Create input FASTA with both chains
        input_fasta = f"{output_dir}/input_{sequence.sequence_id}.fasta"
        with open(input_fasta, "w") as f:
            f.write(f">target|chain=A\n{target_sequence}\n")
            f.write(f">binder|chain=B\n{sequence.sequence}\n")

        # Prepare Boltz-2 input configuration
        boltz_config = {
            "sequences": [
                {"protein": target_sequence, "chain": "A"},
                {"protein": sequence.sequence, "chain": "B"},
            ],
            "num_recycles": config.num_recycles,
        }

        config_path = f"{output_dir}/boltz_config_{sequence.sequence_id}.json"
        with open(config_path, "w") as f:
            json.dump(boltz_config, f)

        output_pdb = f"{output_dir}/{sequence.sequence_id}_complex.pdb"

        # Run Boltz-2 inference
        cmd = [
            "python",
            "-m",
            "boltz.predict",
            "--config",
            config_path,
            "--output",
            output_pdb,
            "--model_path",
            f"{WEIGHTS_PATH}/boltz2",
            "--num_recycles",
            str(config.num_recycles),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )

        if result.returncode != 0:
            print(f"Boltz-2 stderr: {result.stderr}")
            # Try to continue and check for output

        # Parse Boltz-2 output metrics
        metrics = _parse_boltz2_metrics(output_dir, sequence.sequence_id)

        if metrics is None:
            return None

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
            print(
                f"Sequence {sequence.sequence_id} failed i-pLDDT filter: "
                f"{prediction.plddt_interface:.1f} < {config.min_iplddt * 100}"
            )
            return None

        if prediction.pae_interface > config.max_pae:
            print(
                f"Sequence {sequence.sequence_id} failed PAE filter: "
                f"{prediction.pae_interface:.1f} > {config.max_pae}"
            )
            return None

        return prediction

    except subprocess.TimeoutExpired:
        print(f"Boltz-2 timeout for sequence {sequence.sequence_id}")
        return None
    except Exception as e:
        print(f"Boltz-2 error for sequence {sequence.sequence_id}: {e}")
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
    """Parse Boltz-2 output metrics from JSON or pickle files."""
    import numpy as np

    metrics = {}

    # Try to find metrics file
    metrics_file = f"{output_dir}/{sequence_id}_metrics.json"
    if os.path.exists(metrics_file):
        with open(metrics_file, "r") as f:
            raw_metrics = json.load(f)
            metrics["plddt_overall"] = raw_metrics.get("plddt", 0.0)
            metrics["ptm"] = raw_metrics.get("ptm")
            metrics["iptm"] = raw_metrics.get("iptm")
            metrics["plddt_interface"] = raw_metrics.get("plddt_interface", 0.0)
            metrics["pae_interface"] = raw_metrics.get("pae_interface", 100.0)
            return metrics

    # Try numpy file
    plddt_file = f"{output_dir}/{sequence_id}_plddt.npy"
    pae_file = f"{output_dir}/{sequence_id}_pae.npy"

    if os.path.exists(plddt_file):
        plddt = np.load(plddt_file)
        metrics["plddt_overall"] = float(np.mean(plddt))
        # Estimate interface pLDDT (last N residues = binder)
        metrics["plddt_interface"] = float(np.mean(plddt[-50:]))  # Approximate

    if os.path.exists(pae_file):
        pae = np.load(pae_file)
        # Interface PAE: cross-chain PAE values
        metrics["pae_interface"] = float(np.mean(pae))  # Simplified

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
    timeout=600,
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
        db_path = f"{WEIGHTS_PATH}/foldseek/{config.database}"

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

        # Parse results
        if os.path.exists(results_file):
            decoys = _parse_foldseek_results(results_file, output_dir, config.max_hits)

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
    """Parse FoldSeek tabular output into DecoyHit objects."""
    decoys: list[DecoyHit] = []

    with open(results_file, "r") as f:
        for i, line in enumerate(f):
            if i >= max_hits:
                break

            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue

            target_id = parts[0]
            evalue = float(parts[1]) if parts[1] else 1e10
            tm_score = float(parts[2]) if parts[2] else 0.0
            aligned_length = int(parts[3]) if parts[3] else 0
            seq_identity = float(parts[4]) if parts[4] else 0.0

            # Create placeholder PDB path (would need to download actual structure)
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

    for decoy in decoys:
        try:
            # Extract PDB ID (assuming format like "1ABC_A")
            pdb_id = decoy.decoy_id.split("_")[0].lower()
            chain_id = decoy.decoy_id.split("_")[1] if "_" in decoy.decoy_id else "A"

            pdb_path = f"{output_dir}/{decoy.decoy_id}.pdb"

            if not os.path.exists(pdb_path):
                # Download from RCSB PDB
                url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
                urllib.request.urlretrieve(url, pdb_path)

            if os.path.exists(pdb_path):
                valid_decoys.append(
                    DecoyHit(
                        decoy_id=decoy.decoy_id,
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
    timeout=600,
    volumes={WEIGHTS_PATH: weights_volume, DATA_PATH: data_volume},
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
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Extract decoy sequence
        decoy_sequence = _extract_sequence_from_pdb(decoy.pdb_path, "A")

        # Prepare Chai-1 input
        input_fasta = f"{output_dir}/chai1_input_{sequence.sequence_id}_{decoy.decoy_id}.fasta"
        with open(input_fasta, "w") as f:
            f.write(f">decoy\n{decoy_sequence}\n")
            f.write(f">binder\n{sequence.sequence}\n")

        output_prefix = f"{output_dir}/{sequence.sequence_id}_{decoy.decoy_id}"

        # Run Chai-1 docking
        cmd = [
            "python",
            "-m",
            "chai.predict",
            "--fasta",
            input_fasta,
            "--output_prefix",
            output_prefix,
            "--model_path",
            f"{WEIGHTS_PATH}/chai1",
            "--num_samples",
            str(config.num_samples),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode != 0:
            print(f"Chai-1 stderr: {result.stderr}")

        # Parse Chai-1 output
        metrics = _parse_chai1_metrics(output_prefix)

        if metrics is None:
            return None

        # Determine if binder likely binds decoy
        # Use a threshold based on interface pLDDT and predicted affinity
        binds_decoy = (
            metrics.get("plddt_interface", 0) > 60
            and metrics.get("affinity", 0) < -5.0
        )

        return CrossReactivityResult(
            binder_id=sequence.sequence_id,
            decoy_id=decoy.decoy_id,
            predicted_affinity=metrics.get("affinity", 0.0),
            plddt_interface=metrics.get("plddt_interface", 0.0),
            binds_decoy=binds_decoy,
        )

    except subprocess.TimeoutExpired:
        print(f"Chai-1 timeout for {sequence.sequence_id} vs {decoy.decoy_id}")
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
    timeout=60,
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
    timeout=60,
)
def check_cross_reactivity_parallel(
    sequences: list[SequenceDesign],
    decoys: list[DecoyHit],
    config: Chai1Config,
    base_output_dir: str,
) -> dict[str, list[CrossReactivityResult]]:
    """
    Check cross-reactivity for multiple sequences against decoys in parallel.

    Args:
        sequences: List of binder sequences
        decoys: List of decoy structures
        config: Chai-1 configuration
        base_output_dir: Base output directory

    Returns:
        Dictionary mapping sequence_id to list of CrossReactivityResults
    """
    # Prepare all combinations for starmap
    args = [
        (seq, decoy, config, f"{base_output_dir}/{seq.sequence_id}/{decoy.decoy_id}")
        for seq in sequences
        for decoy in decoys
    ]

    # Use starmap for parallel execution
    all_results = list(run_chai1.starmap(args))

    # Group results by sequence
    results_by_sequence: dict[str, list[CrossReactivityResult]] = {}
    for result in all_results:
        if result is not None:
            if result.binder_id not in results_by_sequence:
                results_by_sequence[result.binder_id] = []
            results_by_sequence[result.binder_id].append(result)

    return results_by_sequence


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
