#!/usr/bin/env python3
"""
Prepare IL7RA target from PDB 3DI3 for binder design pipeline.

Target: Human Interleukin-7 Receptor Alpha (IL7RA)
Source: https://www.rcsb.org/structure/3DI3
        Crystal structure of human IL-7 with glycosylated IL-7 receptor alpha ectodomain

Configuration from experimental data:
- PDB: 3DI3 (glycosylated variant)
- Hotspots: B58, B80, B139
- Interchain PAE threshold: <8 Å
- Designs tested: 95
- Hit rate: 0.35 (experimentally validated via BLI)
- Hit criteria: Bio-layer interferometry [0.5, 1] (positive control)
"""

import urllib.request
import os
from pathlib import Path


def download_pdb(pdb_id: str, output_path: str) -> str:
    """Download PDB file from RCSB."""
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    print(f"Downloading {pdb_id} from RCSB PDB...")
    urllib.request.urlretrieve(url, output_path)
    print(f"  → Saved to {output_path}")
    return output_path


def extract_chain(input_pdb: str, chain_id: str, output_pdb: str) -> str:
    """
    Extract a single chain from a PDB file.
    
    Args:
        input_pdb: Path to input PDB file
        chain_id: Chain ID to extract (e.g., 'B')
        output_pdb: Path for output PDB file
    
    Returns:
        Path to the output PDB file
    """
    print(f"Extracting chain {chain_id} from {input_pdb}...")
    
    extracted_lines = []
    residue_count = 0
    seen_residues = set()
    
    with open(input_pdb, 'r') as f:
        for line in f:
            # Keep HEADER, TITLE, etc. but modify them
            if line.startswith("HEADER"):
                extracted_lines.append(f"HEADER    IL7RA RECEPTOR (CHAIN {chain_id} FROM 3DI3)\n")
                continue
            
            # Extract ATOM and HETATM records for the specified chain
            if line.startswith(("ATOM", "HETATM")):
                if len(line) > 21 and line[21] == chain_id:
                    # Track unique residues
                    res_id = line[22:27].strip()
                    if res_id not in seen_residues:
                        seen_residues.add(res_id)
                        residue_count += 1
                    extracted_lines.append(line)
            
            # Keep TER records for the chain
            elif line.startswith("TER"):
                if len(line) > 21 and line[21] == chain_id:
                    extracted_lines.append(line)
    
    # Add END record
    extracted_lines.append("END\n")
    
    # Write output
    with open(output_pdb, 'w') as f:
        f.writelines(extracted_lines)
    
    print(f"  → Extracted {residue_count} residues")
    print(f"  → Saved to {output_pdb}")
    
    return output_pdb


def renumber_chain(input_pdb: str, output_pdb: str, new_chain_id: str = "A") -> str:
    """
    Renumber residues starting from 1 and optionally rename chain.
    
    Args:
        input_pdb: Path to input PDB file
        output_pdb: Path for output PDB file  
        new_chain_id: New chain ID (default: 'A')
    
    Returns:
        Path to the output PDB file
    """
    print(f"Renumbering residues and setting chain to {new_chain_id}...")
    
    output_lines = []
    residue_mapping = {}  # old_res_id -> new_res_num
    current_new_resnum = 0
    last_old_resid = None
    
    with open(input_pdb, 'r') as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                # Parse old residue ID
                old_resid = line[22:27]  # includes insertion code
                
                # Assign new residue number if this is a new residue
                if old_resid != last_old_resid:
                    current_new_resnum += 1
                    residue_mapping[old_resid] = current_new_resnum
                    last_old_resid = old_resid
                
                # Rebuild the line with new chain and residue number
                new_resnum = residue_mapping[old_resid]
                new_line = (
                    line[:21] +                    # up to chain ID
                    new_chain_id +                 # new chain ID
                    f"{new_resnum:4d}" +           # new residue number
                    " " +                          # insertion code (blank)
                    line[27:]                      # rest of line
                )
                output_lines.append(new_line)
            
            elif line.startswith("TER"):
                # Update TER record
                if last_old_resid and last_old_resid in residue_mapping:
                    new_resnum = residue_mapping[last_old_resid]
                    # Simplified TER record
                    output_lines.append(f"TER\n")
            
            elif line.startswith("END"):
                output_lines.append(line)
            
            elif line.startswith("HEADER"):
                output_lines.append(line)
    
    with open(output_pdb, 'w') as f:
        f.writelines(output_lines)
    
    print(f"  → Renumbered {current_new_resnum} residues")
    print(f"  → Saved to {output_pdb}")
    
    return output_pdb, residue_mapping


def map_hotspots(original_hotspots: list[tuple[str, int]], residue_mapping: dict) -> list[int]:
    """
    Map original hotspot residues to new numbering.
    
    Args:
        original_hotspots: List of (chain, resnum) tuples
        residue_mapping: Mapping from old residue IDs to new numbers
    
    Returns:
        List of new residue numbers
    """
    new_hotspots = []
    
    for chain, resnum in original_hotspots:
        # Try to find the residue in the mapping
        for old_resid, new_resnum in residue_mapping.items():
            if str(resnum) in old_resid.strip():
                new_hotspots.append(new_resnum)
                break
    
    return new_hotspots


def main():
    """Prepare IL7RA target structure."""
    
    # Configuration
    PDB_ID = "3DI3"
    RECEPTOR_CHAIN = "B"  # IL7RA receptor chain
    ORIGINAL_HOTSPOTS = [("B", 58), ("B", 80), ("B", 139)]
    
    # Paths
    script_dir = Path(__file__).parent
    raw_pdb = script_dir / f"{PDB_ID.lower()}_raw.pdb"
    chain_pdb = script_dir / f"{PDB_ID.lower()}_chain{RECEPTOR_CHAIN}.pdb"
    final_pdb = script_dir / "il7ra_target.pdb"
    
    print("=" * 60)
    print("IL7RA TARGET PREPARATION")
    print("=" * 60)
    print(f"Source: PDB {PDB_ID}")
    print(f"Chain: {RECEPTOR_CHAIN} (IL-7 Receptor Alpha Ectodomain)")
    print(f"Original hotspots: {ORIGINAL_HOTSPOTS}")
    print("=" * 60)
    print()
    
    # Step 1: Download PDB
    download_pdb(PDB_ID, str(raw_pdb))
    print()
    
    # Step 2: Extract receptor chain
    extract_chain(str(raw_pdb), RECEPTOR_CHAIN, str(chain_pdb))
    print()
    
    # Step 3: Renumber and clean
    _, residue_mapping = renumber_chain(str(chain_pdb), str(final_pdb), new_chain_id="A")
    print()
    
    # Step 4: Map hotspots to new numbering
    # For chain B extracted and renumbered, the hotspots should map directly
    # since we're starting from residue 1
    print("Hotspot mapping:")
    print(f"  Original (chain B): {[h[1] for h in ORIGINAL_HOTSPOTS]}")
    
    # The residues in the original PDB chain B should map to the same positions
    # if the chain starts from residue 1 in the original
    # Let's verify by checking the mapping
    new_hotspots = []
    for chain, resnum in ORIGINAL_HOTSPOTS:
        # Look for the residue number in the mapping keys
        for old_resid, new_resnum in residue_mapping.items():
            old_num = ''.join(filter(str.isdigit, old_resid))
            if old_num and int(old_num) == resnum:
                new_hotspots.append(new_resnum)
                print(f"    B{resnum} → A{new_resnum}")
                break
    
    print()
    print("=" * 60)
    print("TARGET READY")
    print("=" * 60)
    print(f"Output file: {final_pdb}")
    print(f"Chain ID: A")
    print(f"Hotspot residues: {new_hotspots}")
    print()
    print("Pipeline command:")
    hotspot_str = ",".join(str(h) for h in new_hotspots) if new_hotspots else "58,80,139"
    print(f"  uv run modal run pipeline.py \\")
    print(f"      --target-pdb {final_pdb} \\")
    print(f"      --hotspot-residues \"{hotspot_str}\" \\")
    print(f"      --chain-id A \\")
    print(f"      --dry-run")
    print()
    
    # Write a config summary file
    config_file = script_dir / "il7ra_config.txt"
    with open(config_file, 'w') as f:
        f.write("# IL7RA Binder Design Target Configuration\n")
        f.write("# =========================================\n")
        f.write("#\n")
        f.write("# Source: https://www.rcsb.org/structure/3DI3\n")
        f.write("# Reference: Crystal structure of human IL-7 with\n")
        f.write("#            glycosylated IL-7 receptor alpha ectodomain\n")
        f.write("#\n")
        f.write("[target]\n")
        f.write(f"pdb_file = {final_pdb.name}\n")
        f.write(f"pdb_source = 3DI3 (chain B, glycosylated variant)\n")
        f.write(f"chain_id = A\n")
        f.write(f"hotspots = {new_hotspots if new_hotspots else [58, 80, 139]}\n")
        f.write("\n")
        f.write("[validation_reference]\n")
        f.write("# From experimental study (SI)\n")
        f.write("interchain_pae_threshold = 8  # Angstrom\n")
        f.write("num_designs_tested = 95\n")
        f.write("experimental_hit_rate = 0.35\n")
        f.write("hit_criteria = BLI  # Bio-layer interferometry\n")
        f.write("hit_range = [0.5, 1]  # positive control normalized\n")
    
    print(f"Config saved to: {config_file}")
    
    return str(final_pdb), new_hotspots if new_hotspots else [58, 80, 139]


if __name__ == "__main__":
    main()
