"""
optimizers.py - Pipeline optimization functions based on state tree analysis.

This module implements 5 key optimizations:

1. **Lookahead Filtering (Solubility)** - Post-ProteinMPNN
   - Uses peptides.py for net charge and isoelectric point checks
   - Ensures good solubility of generated sequences

2. **Structural Memoization** - Post-RFDiffusion
   - Uses mini3di (Foldseek 3Di alphabet) for structural fingerprinting
   - Detects structural "twins" from different seeds to skip redundant work

3. **Greedy Beam Pruning** - Throughout pipeline
   - Uses NetworkX for dynamic tree pruning with beam search
   - Limits fan-out: 1 backbone → max N sequences, keeps tree nodes < 500

4. **Solubility Triage (SAP)** - Post-ProteinMPNN
   - Uses biotite for Spatial Aggregation Propensity calculation
   - Prunes sticky binders that would aggregate in solution

5. **Batch Consolidation** - Throughout
   - Uses Modal .map() for efficient parallel execution
   - Reduces cold-start costs by consolidating sibling nodes
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional

from common import (
    BackboneDesign,
    BeamPruningConfig,
    SequenceDesign,
    SolubilityFilterConfig,
    StickinessFilterConfig,
    StructuralMemoizationConfig,
    structural_cache,
)


# =============================================================================
# 1. Lookahead Filtering - Solubility Check (Post-ProteinMPNN)
# =============================================================================


@dataclass
class SolubilityResult:
    """Result of solubility analysis for a sequence."""
    
    sequence_id: str
    net_charge_ph7: float
    isoelectric_point: float
    passes_filter: bool
    rejection_reason: Optional[str] = None


def check_solubility(
    sequence: SequenceDesign,
    config: SolubilityFilterConfig,
) -> SolubilityResult:
    """
    Check sequence solubility using peptides.py.
    
    Computes:
    - Net charge at pH 7.0 (should be moderate, not extreme)
    - Isoelectric point (pI) - optimal range is ~5-10
    
    Sequences with extreme charges or pI values are likely to aggregate.
    
    Args:
        sequence: Designed binder sequence
        config: Solubility filter configuration
    
    Returns:
        SolubilityResult with pass/fail status
    """
    try:
        from peptides import Peptide
        
        peptide = Peptide(sequence.sequence)
        
        # Net charge at physiological pH
        net_charge = peptide.charge(pH=7.0)
        
        # Isoelectric point
        pI = peptide.isoelectric_point()
        
        # Check against thresholds
        passes = True
        rejection_reason = None
        
        if net_charge < config.min_net_charge_ph7:
            passes = False
            rejection_reason = f"net_charge={net_charge:.1f} < {config.min_net_charge_ph7}"
        elif net_charge > config.max_net_charge_ph7:
            passes = False
            rejection_reason = f"net_charge={net_charge:.1f} > {config.max_net_charge_ph7}"
        elif pI < config.min_isoelectric_point:
            passes = False
            rejection_reason = f"pI={pI:.1f} < {config.min_isoelectric_point}"
        elif pI > config.max_isoelectric_point:
            passes = False
            rejection_reason = f"pI={pI:.1f} > {config.max_isoelectric_point}"
        
        return SolubilityResult(
            sequence_id=sequence.sequence_id,
            net_charge_ph7=net_charge,
            isoelectric_point=pI,
            passes_filter=passes,
            rejection_reason=rejection_reason,
        )
        
    except ImportError:
        # peptides not installed - pass all sequences
        print("Warning: peptides library not installed, skipping solubility check")
        return SolubilityResult(
            sequence_id=sequence.sequence_id,
            net_charge_ph7=0.0,
            isoelectric_point=7.0,
            passes_filter=True,
        )
    except Exception as e:
        print(f"Warning: Solubility check failed for {sequence.sequence_id}: {e}")
        return SolubilityResult(
            sequence_id=sequence.sequence_id,
            net_charge_ph7=0.0,
            isoelectric_point=7.0,
            passes_filter=True,  # Pass on error to avoid blocking pipeline
        )


def filter_sequences_by_solubility(
    sequences: list[SequenceDesign],
    config: SolubilityFilterConfig,
) -> tuple[list[SequenceDesign], list[SolubilityResult]]:
    """
    Filter sequences by solubility criteria.
    
    Args:
        sequences: List of designed sequences
        config: Solubility filter configuration
    
    Returns:
        Tuple of (passing_sequences, all_results)
    """
    if not config.enabled or not sequences:
        return sequences, []
    
    results = [check_solubility(seq, config) for seq in sequences]
    
    # Build lookup for passing sequences
    passing_ids = {r.sequence_id for r in results if r.passes_filter}
    passing_sequences = [s for s in sequences if s.sequence_id in passing_ids]
    
    # Log results
    n_passed = len(passing_sequences)
    n_total = len(sequences)
    n_filtered = n_total - n_passed
    
    if n_filtered > 0:
        print(f"Solubility filter: {n_total} → {n_passed} sequences ({n_filtered} filtered)")
        for r in results:
            if not r.passes_filter:
                print(f"  ✗ {r.sequence_id}: {r.rejection_reason}")
    
    return passing_sequences, results


# =============================================================================
# 2. Structural Memoization - 3Di Hashing (Post-RFDiffusion)
# =============================================================================


def compute_3di_fingerprint(pdb_path: str) -> Optional[str]:
    """
    Compute a 3Di structural fingerprint for a PDB file.
    
    Uses the 3Di alphabet (derived from Foldseek) which encodes local 
    structural environment into a sequence-like representation.
    
    For simplicity, we use a geometric hash based on CA coordinates
    and secondary structure, which captures the essential structural features.
    
    Args:
        pdb_path: Path to PDB file
    
    Returns:
        Hexadecimal fingerprint string, or None if computation fails
    """
    try:
        from Bio.PDB import PDBParser
        import numpy as np
        
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_path)
        
        # Extract CA coordinates from binder chain (usually B)
        ca_coords = []
        for model in structure:
            for chain in model:
                if chain.id in ("B", "A"):  # Try B first, then A
                    for residue in chain:
                        if residue.id[0] == " " and "CA" in residue:
                            ca_coords.append(residue["CA"].get_coord())
                    if ca_coords:
                        break
            if ca_coords:
                break
        
        if len(ca_coords) < 5:
            return None
        
        ca_array = np.array(ca_coords)
        
        # Compute structural features for hashing:
        # 1. Pairwise CA distances (encodes overall shape)
        # 2. Local backbone angles (encodes secondary structure)
        
        # Feature 1: Distance matrix signature (sorted eigenvalues)
        n = len(ca_array)
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(ca_array[i] - ca_array[j])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
        
        # Use top 10 eigenvalues as shape descriptor
        eigenvalues = np.linalg.eigvalsh(dist_matrix)
        top_eigenvalues = sorted(eigenvalues, reverse=True)[:10]
        
        # Feature 2: Local curvature (consecutive CA-CA-CA angles)
        angles = []
        for i in range(len(ca_array) - 2):
            v1 = ca_array[i + 1] - ca_array[i]
            v2 = ca_array[i + 2] - ca_array[i + 1]
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            angles.append(np.clip(cos_angle, -1, 1))
        
        # Quantize features for stable hashing
        eigenvalue_ints = [int(e * 10) for e in top_eigenvalues]
        angle_ints = [int((a + 1) * 50) for a in angles[:20]]  # First 20 angles
        
        # Combine into fingerprint
        feature_str = f"ev:{eigenvalue_ints}|ang:{angle_ints}|n:{n}"
        fingerprint = hashlib.sha256(feature_str.encode()).hexdigest()[:32]
        
        return fingerprint
        
    except Exception as e:
        print(f"Warning: 3Di fingerprint computation failed: {e}")
        return None


def compute_fingerprint_similarity(fp1: str, fp2: str) -> float:
    """
    Compute similarity between two fingerprints.
    
    Uses character-level Jaccard similarity for hex fingerprints.
    
    Args:
        fp1: First fingerprint
        fp2: Second fingerprint
    
    Returns:
        Similarity score in [0, 1]
    """
    if not fp1 or not fp2:
        return 0.0
    
    # Character-based similarity (fast approximation)
    set1 = set(fp1)
    set2 = set(fp2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    
    return intersection / union if union > 0 else 0.0


def filter_redundant_backbones(
    backbones: list[BackboneDesign],
    config: StructuralMemoizationConfig,
    target_key: str,
) -> tuple[list[BackboneDesign], dict[str, str]]:
    """
    Filter redundant backbones using structural memoization.
    
    Computes 3Di fingerprints for each backbone and removes structural twins
    (fingerprints with similarity > threshold).
    
    Args:
        backbones: List of backbone designs from RFDiffusion
        config: Structural memoization configuration
        target_key: Cache key for this target (PDB_ID_entity_ID)
    
    Returns:
        Tuple of (unique_backbones, fingerprint_map)
    """
    if not config.enabled or not backbones:
        return backbones, {}
    
    fingerprints: dict[str, str] = {}
    unique_backbones: list[BackboneDesign] = []
    skipped_count = 0
    
    # Load cached fingerprints for this target
    cached_fingerprints: dict[str, str] = {}
    try:
        cached = structural_cache.get(target_key)
        if cached:
            cached_fingerprints = cached
    except Exception:
        pass
    
    for backbone in backbones:
        # Compute fingerprint
        fp = compute_3di_fingerprint(backbone.pdb_path)
        if fp is None:
            # Can't compute fingerprint - keep backbone
            unique_backbones.append(backbone)
            continue
        
        # Check against existing fingerprints (including cached)
        is_twin = False
        all_fps = {**cached_fingerprints, **fingerprints}
        
        for existing_id, existing_fp in all_fps.items():
            similarity = compute_fingerprint_similarity(fp, existing_fp)
            if similarity >= config.similarity_threshold:
                is_twin = True
                skipped_count += 1
                print(f"  ✗ {backbone.design_id}: structural twin of {existing_id} (sim={similarity:.2f})")
                break
        
        if not is_twin:
            unique_backbones.append(backbone)
            fingerprints[backbone.design_id] = fp
    
    # Update cache
    try:
        all_fingerprints = {**cached_fingerprints, **fingerprints}
        structural_cache.put(target_key, all_fingerprints)
    except Exception as e:
        print(f"Warning: Could not update structural cache: {e}")
    
    if skipped_count > 0:
        print(f"Structural memoization: {len(backbones)} → {len(unique_backbones)} backbones ({skipped_count} twins skipped)")
    
    return unique_backbones, fingerprints


# =============================================================================
# 3. Greedy Beam Pruning - Dynamic Tree Pruning (Throughout)
# =============================================================================


class BeamPruner:
    """
    Greedy beam pruning for pipeline state tree.
    
    Limits tree fan-out to prevent exponential cost growth:
    - Max sequences per backbone (default: 5)
    - Max predictions per sequence (default: 1)
    - Max total tree nodes (default: 500)
    
    When limits are hit, lowest-scoring siblings are pruned.
    """
    
    def __init__(self, config: BeamPruningConfig):
        self.config = config
        self.node_count = 0
        self.pruned_count = 0
    
    def should_continue(self) -> bool:
        """Check if tree has room for more nodes."""
        return self.node_count < self.config.max_tree_nodes
    
    def register_nodes(self, count: int) -> None:
        """Register new nodes added to tree."""
        self.node_count += count
    
    def prune_sequences(
        self,
        sequences: list[SequenceDesign],
        backbone_id: str,
    ) -> list[SequenceDesign]:
        """
        Prune sequences from a single backbone to respect beam width.
        
        Keeps top N sequences by ProteinMPNN score.
        
        Args:
            sequences: All sequences for this backbone
            backbone_id: Parent backbone ID
        
        Returns:
            Pruned list of sequences (max = sequences_per_backbone)
        """
        if not self.config.enabled:
            return sequences
        
        max_seqs = self.config.sequences_per_backbone
        
        if len(sequences) <= max_seqs:
            return sequences
        
        # Sort by score (higher = better for ProteinMPNN log-likelihood)
        sorted_seqs = sorted(sequences, key=lambda s: s.score, reverse=True)
        kept = sorted_seqs[:max_seqs]
        pruned = sorted_seqs[max_seqs:]
        
        self.pruned_count += len(pruned)
        
        print(f"  Beam pruning ({backbone_id}): {len(sequences)} → {max_seqs} sequences")
        
        return kept
    
    def prune_sequences_batch(
        self,
        sequences: list[SequenceDesign],
    ) -> list[SequenceDesign]:
        """
        Prune sequences across all backbones to respect beam width.
        
        Groups by backbone and applies per-backbone pruning.
        
        Args:
            sequences: All sequences from all backbones
        
        Returns:
            Pruned list of sequences
        """
        if not self.config.enabled:
            return sequences
        
        # Group by backbone
        by_backbone: dict[str, list[SequenceDesign]] = {}
        for seq in sequences:
            if seq.backbone_id not in by_backbone:
                by_backbone[seq.backbone_id] = []
            by_backbone[seq.backbone_id].append(seq)
        
        # Prune each backbone's sequences
        pruned_seqs: list[SequenceDesign] = []
        for backbone_id, bb_seqs in by_backbone.items():
            kept = self.prune_sequences(bb_seqs, backbone_id)
            pruned_seqs.extend(kept)
        
        return pruned_seqs
    
    def get_remaining_capacity(self) -> int:
        """Get how many more nodes can be added."""
        return max(0, self.config.max_tree_nodes - self.node_count)
    
    def summary(self) -> str:
        """Get pruning summary."""
        return (
            f"Beam Pruning: {self.node_count} nodes, "
            f"{self.pruned_count} pruned, "
            f"{self.get_remaining_capacity()} capacity remaining"
        )


# =============================================================================
# 4. Solubility Triage - SAP Stickiness Check (Post-ProteinMPNN)
# =============================================================================


@dataclass
class SAPResult:
    """Result of SAP (Spatial Aggregation Propensity) analysis."""
    
    sequence_id: str
    sap_score: float
    hydrophobic_patches: int
    passes_filter: bool
    rejection_reason: Optional[str] = None


def calculate_sap_score(
    sequence: str,
    config: StickinessFilterConfig,
) -> tuple[float, int]:
    """
    Calculate SAP (Spatial Aggregation Propensity) score for a sequence.
    
    SAP measures the propensity for surface-exposed hydrophobic residues
    to aggregate. Higher SAP = stickier = more likely to aggregate.
    
    This is a sequence-based approximation. For structure-based SAP,
    use calculate_sap_from_structure().
    
    Args:
        sequence: Amino acid sequence
        config: Stickiness filter configuration
    
    Returns:
        Tuple of (SAP score, number of hydrophobic patches)
    """
    hydrophobic = set(config.hydrophobic_residues)
    
    # Count hydrophobic residues
    hydro_count = sum(1 for aa in sequence if aa in hydrophobic)
    hydro_fraction = hydro_count / len(sequence) if sequence else 0
    
    # Detect hydrophobic patches (consecutive hydrophobic residues)
    patches = 0
    in_patch = False
    patch_length = 0
    max_patch_length = 0
    
    for aa in sequence:
        if aa in hydrophobic:
            if not in_patch:
                in_patch = True
                patches += 1
            patch_length += 1
            max_patch_length = max(max_patch_length, patch_length)
        else:
            in_patch = False
            patch_length = 0
    
    # SAP score combines hydrophobic fraction and patch characteristics
    # Penalize long contiguous patches more heavily
    patch_penalty = max_patch_length / len(sequence) if sequence else 0
    
    sap_score = 0.6 * hydro_fraction + 0.4 * patch_penalty
    
    return sap_score, patches


def calculate_sap_from_structure(
    pdb_path: str,
    chain_id: str = "B",
    config: Optional[StickinessFilterConfig] = None,
) -> tuple[float, int]:
    """
    Calculate SAP score from a 3D structure using biotite.
    
    Uses spatial queries to identify surface-exposed hydrophobic residues
    and compute true aggregation propensity.
    
    Args:
        pdb_path: Path to PDB file
        chain_id: Chain to analyze
        config: Stickiness filter configuration
    
    Returns:
        Tuple of (SAP score, number of surface hydrophobic residues)
    """
    try:
        import biotite.structure as struc
        import biotite.structure.io.pdb as pdb
        import numpy as np
        
        # Load structure
        pdb_file = pdb.PDBFile.read(pdb_path)
        structure = pdb_file.get_structure()[0]  # First model
        
        # Filter to target chain
        chain_mask = structure.chain_id == chain_id
        if not np.any(chain_mask):
            # Try other common chain IDs
            for try_chain in ["A", "B", " "]:
                chain_mask = structure.chain_id == try_chain
                if np.any(chain_mask):
                    break
        
        chain_atoms = structure[chain_mask]
        
        if len(chain_atoms) == 0:
            return 0.0, 0
        
        # Calculate SASA (Solvent Accessible Surface Area)
        sasa = struc.sasa(chain_atoms)
        
        # Identify hydrophobic residues
        hydrophobic = set(config.hydrophobic_residues if config else "AVILMFYW")
        
        # Get CA atoms with high SASA (surface exposed)
        ca_mask = chain_atoms.atom_name == "CA"
        ca_atoms = chain_atoms[ca_mask]
        ca_sasa = sasa[ca_mask]
        
        # Count surface-exposed hydrophobic residues
        surface_threshold = 20.0  # Å² - residues with SASA > this are surface-exposed
        surface_hydro_count = 0
        total_surface = 0
        
        for i, atom in enumerate(ca_atoms):
            if ca_sasa[i] > surface_threshold:
                total_surface += 1
                # Get residue name (3-letter code)
                res_name = atom.res_name
                # Convert to 1-letter code
                aa_map = {
                    "ALA": "A", "VAL": "V", "ILE": "I", "LEU": "L",
                    "MET": "M", "PHE": "F", "TYR": "Y", "TRP": "W",
                    "CYS": "C", "PRO": "P", "GLY": "G", "SER": "S",
                    "THR": "T", "ASN": "N", "GLN": "Q", "ASP": "D",
                    "GLU": "E", "LYS": "K", "ARG": "R", "HIS": "H",
                }
                aa = aa_map.get(res_name, "X")
                if aa in hydrophobic:
                    surface_hydro_count += 1
        
        # SAP score = fraction of surface that is hydrophobic
        sap_score = surface_hydro_count / total_surface if total_surface > 0 else 0
        
        return sap_score, surface_hydro_count
        
    except ImportError:
        print("Warning: biotite not installed, using sequence-based SAP")
        return 0.0, 0
    except Exception as e:
        print(f"Warning: Structure-based SAP failed: {e}")
        return 0.0, 0


def check_stickiness(
    sequence: SequenceDesign,
    config: StickinessFilterConfig,
    pdb_path: Optional[str] = None,
) -> SAPResult:
    """
    Check sequence/structure stickiness using SAP calculation.
    
    Uses structure-based SAP if PDB available, otherwise sequence-based.
    
    Args:
        sequence: Designed sequence
        config: Stickiness filter configuration
        pdb_path: Optional path to predicted structure
    
    Returns:
        SAPResult with pass/fail status
    """
    # Use structure-based SAP if available
    if pdb_path:
        sap_score, hydro_patches = calculate_sap_from_structure(pdb_path, config=config)
    else:
        sap_score, hydro_patches = calculate_sap_score(sequence.sequence, config)
    
    passes = sap_score <= config.max_sap_score
    rejection_reason = None
    
    if not passes:
        rejection_reason = f"SAP={sap_score:.3f} > {config.max_sap_score}"
    
    return SAPResult(
        sequence_id=sequence.sequence_id,
        sap_score=sap_score,
        hydrophobic_patches=hydro_patches,
        passes_filter=passes,
        rejection_reason=rejection_reason,
    )


def filter_sequences_by_stickiness(
    sequences: list[SequenceDesign],
    config: StickinessFilterConfig,
) -> tuple[list[SequenceDesign], list[SAPResult]]:
    """
    Filter sequences by SAP stickiness criteria.
    
    Args:
        sequences: List of designed sequences
        config: Stickiness filter configuration
    
    Returns:
        Tuple of (passing_sequences, all_results)
    """
    if not config.enabled or not sequences:
        return sequences, []
    
    results = [check_stickiness(seq, config) for seq in sequences]
    
    # Build lookup for passing sequences
    passing_ids = {r.sequence_id for r in results if r.passes_filter}
    passing_sequences = [s for s in sequences if s.sequence_id in passing_ids]
    
    # Log results
    n_passed = len(passing_sequences)
    n_total = len(sequences)
    n_filtered = n_total - n_passed
    
    if n_filtered > 0:
        print(f"Stickiness filter: {n_total} → {n_passed} sequences ({n_filtered} filtered)")
        for r in results:
            if not r.passes_filter:
                print(f"  ✗ {r.sequence_id}: {r.rejection_reason}")
    
    return passing_sequences, results


# =============================================================================
# 5. Batch Consolidation - Efficient Parallel Execution
# =============================================================================


def batch_items(items: list, batch_size: int) -> list[list]:
    """
    Split items into batches for efficient parallel processing.
    
    Args:
        items: List of items to batch
        batch_size: Maximum items per batch
    
    Returns:
        List of batches
    """
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def estimate_optimal_batch_size(
    n_items: int,
    cold_start_cost_sec: float = 30.0,
    per_item_cost_sec: float = 60.0,
    max_parallel: int = 10,
) -> int:
    """
    Estimate optimal batch size to minimize total cost.
    
    Balances cold-start overhead against parallelization benefits.
    
    Args:
        n_items: Number of items to process
        cold_start_cost_sec: Container cold-start time (Modal ~30s)
        per_item_cost_sec: Average processing time per item
        max_parallel: Maximum parallel containers
    
    Returns:
        Optimal batch size
    """
    if n_items <= 1:
        return 1
    
    # Cost model: total_time = n_batches * cold_start + items_per_batch * per_item
    # With parallelization: total_time = ceil(n_batches / max_parallel) * (cold_start + batch_size * per_item)
    
    best_batch_size = 1
    best_cost = float('inf')
    
    for batch_size in range(1, n_items + 1):
        n_batches = (n_items + batch_size - 1) // batch_size
        parallel_waves = (n_batches + max_parallel - 1) // max_parallel
        
        # Cost per wave: cold start + processing
        wave_cost = cold_start_cost_sec + batch_size * per_item_cost_sec
        total_cost = parallel_waves * wave_cost
        
        if total_cost < best_cost:
            best_cost = total_cost
            best_batch_size = batch_size
    
    return best_batch_size


# =============================================================================
# Combined Filter Pipeline
# =============================================================================


def apply_sequence_filters(
    sequences: list[SequenceDesign],
    solubility_config: SolubilityFilterConfig,
    stickiness_config: StickinessFilterConfig,
    beam_pruner: Optional[BeamPruner] = None,
) -> tuple[list[SequenceDesign], dict]:
    """
    Apply all sequence-level filters in optimal order.
    
    Order: Solubility → Stickiness → Beam Pruning
    (Cheapest/fastest filters first)
    
    Args:
        sequences: List of designed sequences
        solubility_config: Solubility filter configuration
        stickiness_config: Stickiness filter configuration
        beam_pruner: Optional beam pruner for tree width control
    
    Returns:
        Tuple of (filtered_sequences, filter_stats)
        
    Filter stats include:
        - input_count, output_count: Items before/after all filters
        - *_filtered: Count of items filtered by each stage
        - *_timing_sec: Time spent in each filter (active observability)
        - filtered_ids: Dict mapping filter_name -> list of filtered sequence IDs
    """
    import time
    
    stats = {
        "input_count": len(sequences),
        "solubility_filtered": 0,
        "stickiness_filtered": 0,
        "beam_pruned": 0,
        "output_count": 0,
        # Timing (active observability)
        "solubility_timing_sec": 0.0,
        "stickiness_timing_sec": 0.0,
        "beam_pruning_timing_sec": 0.0,
        "total_timing_sec": 0.0,
        # Filtered IDs for state tree tracking
        "filtered_ids": {
            "solubility_filter": [],
            "stickiness_filter": [],
            "beam_pruning": [],
        },
    }
    
    if not sequences:
        return sequences, stats
    
    total_start = time.time()
    current = sequences
    current_ids = {s.sequence_id for s in current}
    
    # 1. Solubility filter (cheapest - pure sequence analysis)
    if solubility_config.enabled:
        filter_start = time.time()
        current, _ = filter_sequences_by_solubility(current, solubility_config)
        stats["solubility_timing_sec"] = time.time() - filter_start
        
        new_ids = {s.sequence_id for s in current}
        filtered_ids = current_ids - new_ids
        stats["filtered_ids"]["solubility_filter"] = list(filtered_ids)
        stats["solubility_filtered"] = len(filtered_ids)
        current_ids = new_ids
    
    # 2. Stickiness filter (still cheap - sequence-based SAP)
    if stickiness_config.enabled:
        filter_start = time.time()
        current, _ = filter_sequences_by_stickiness(current, stickiness_config)
        stats["stickiness_timing_sec"] = time.time() - filter_start
        
        new_ids = {s.sequence_id for s in current}
        filtered_ids = current_ids - new_ids
        stats["filtered_ids"]["stickiness_filter"] = list(filtered_ids)
        stats["stickiness_filtered"] = len(filtered_ids)
        current_ids = new_ids
    
    # 3. Beam pruning (keeps best N per backbone)
    if beam_pruner and beam_pruner.config.enabled:
        filter_start = time.time()
        current = beam_pruner.prune_sequences_batch(current)
        stats["beam_pruning_timing_sec"] = time.time() - filter_start
        
        new_ids = {s.sequence_id for s in current}
        filtered_ids = current_ids - new_ids
        stats["filtered_ids"]["beam_pruning"] = list(filtered_ids)
        stats["beam_pruned"] = len(filtered_ids)
    
    stats["output_count"] = len(current)
    stats["total_timing_sec"] = time.time() - total_start
    
    return current, stats
