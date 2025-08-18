#!/usr/bin/env python3
"""
Example usage of the RNA Template Matching and Assembly program
"""

import numpy as np
from Bio import PDB
import sys
import os

# Import the main program functions (assuming they're in rna_template_matcher.py)
try:
    from rna_template_matcher import (
        calculate_center_of_mass,
        get_centers_of_mass_from_structure,
        assemble_structure,
        save_assembled_structure,
        validate_assembled_structure
    )
except ImportError:
    print("Please ensure rna_template_matcher.py is in the same directory or in PYTHONPATH")
    sys.exit(1)

def load_input_structure(pdb_file):
    """Load input RNA structure and extract sequence and coordinates."""
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('input', pdb_file)
    
    sequence = ""
    coords = []
    
    for model in structure:
        for chain in model:
            for residue in chain:
                resname = residue.get_resname().strip()
                if resname in ['A', 'U', 'G', 'C']:
                    sequence += resname
                    # Calculate center of mass for this residue
                    atom_coords = np.array([atom.coord for atom in residue.get_atoms()])
                    mass = np.ones(len(atom_coords))
                    center_of_mass = np.average(atom_coords, axis=0, weights=mass)
                    coords.append(center_of_mass)
    
    return sequence, np.array(coords)


def run_assembly_pipeline(database_path, input_pdb_file, output_pdb_file):
    """
    Complete pipeline for RNA structure assembly.
    
    Args:
        database_path: Path to the template database folder
        input_pdb_file: Path to input PDB file with RNA structure
        output_pdb_file: Path for output assembled structure
    """
    
    print("RNA Template Matching and Assembly Pipeline")
    print("=" * 50)
    
    # Step 1: Load input data
    print(f"\n1. Loading input structure from: {input_pdb_file}")
    try:
        input_sequence, input_coords = load_input_structure(input_pdb_file)
        print(f"   Sequence: {input_sequence}")
        print(f"   Length: {len(input_sequence)} nucleotides")
        print(f"   Coordinates shape: {input_coords.shape}")
    except Exception as e:
        print(f"   Error loading input: {e}")
        return
    
    # Step 2: Validate database
    print(f"\n2. Checking template database: {database_path}")
    if not os.path.exists(database_path):
        print(f"   Error: Database path does not exist")
        return
    
    # Check for required subfolders
    required_subsequences = set()
    for i in range(len(input_sequence) - 3):
        required_subsequences.add(input_sequence[i:i+4])
    
    print(f"   Required template folders: {len(required_subsequences)}")
    missing_folders = []
    for subseq in required_subsequences:
        if not os.path.exists(os.path.join(database_path, subseq)):
            missing_folders.append(subseq)
    
    if missing_folders:
        print(f"   Warning: Missing folders for: {', '.join(missing_folders)}")
    
    # Step 3: Run assembly
    print(f"\n3. Running template matching and assembly...")
    try:
        assembled_residues, residue_weights = assemble_structure(
            database_path, input_sequence, input_coords, window_size=4
        )
        
        print(f"\n   Assembly completed successfully!")
        print(f"   Residues assembled: {len([r for r in assembled_residues if r is not None])}/{len(input_sequence)}")
        
    except Exception as e:
        print(f"   Error during assembly: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 4: Validate assembly
    print(f"\n4. Validating assembled structure...")
    issues = validate_assembled_structure(assembled_residues)
    if issues:
        print("   Issues found:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("   No issues found - all positions have complete structures")
    
    # Step 5: Save output
    print(f"\n5. Saving assembled structure to: {output_pdb_file}")
    try:
        save_assembled_structure(output_pdb_file, assembled_residues, input_sequence)
        print(f"   Saved successfully!")
        
        # Count atoms in output
        parser = PDB.PDBParser(QUIET=True)
        output_structure = parser.get_structure('output', output_pdb_file)
        atom_count = sum(1 for _ in output_structure.get_atoms())
        print(f"   Total atoms in output: {atom_count}")
        
    except Exception as e:
        print(f"   Error saving output: {e}")
        return
    
    # Step 6: Calculate statistics
    print(f"\n6. Assembly Statistics:")
    # Calculate RMSD based on centers of mass
    assembled_com = []
    for residue in assembled_residues:
        if residue is not None:
            assembled_com.append(calculate_center_of_mass(residue))
        else:
            assembled_com.append(np.array([0, 0, 0]))  # Placeholder for missing residues
    
    assembled_com = np.array(assembled_com)
    valid_positions = [i for i, r in enumerate(assembled_residues) if r is not None]
    
    if valid_positions:
        rmsd = calculate_total_rmsd(
            input_coords[valid_positions], 
            assembled_com[valid_positions]
        )
        print(f"   Total RMSD (centers of mass): {rmsd:.3f} Å")
    
    # Print coverage summary
    print(f"\n   Coverage Summary:")
    n = len(input_sequence)
    boundary_positions = list(range(3)) + list(range(n-3, n))
    
    for i, weight in enumerate(residue_weights):
        if weight > 0:
            if i in boundary_positions:
                print(f"   Position {i+1} ({input_sequence[i]}): boundary position (template used directly)")
            else:
                print(f"   Position {i+1} ({input_sequence[i]}): internal position (averaged from {weight} templates)")
        else:
            print(f"   Position {i+1} ({input_sequence[i]}): NOT COVERED")


# Note: calculate_total_rmsd is defined in this file since it's not in the main module

def calculate_total_rmsd(coords1, coords2):
    """Calculate overall RMSD between two coordinate sets."""
    diff = coords1 - coords2
    return np.sqrt(np.mean(np.sum(diff**2, axis=1)))


def compare_structures_detailed(input_pdb, output_pdb):
    """
    Detailed comparison of input and output structures.
    Shows atom-by-atom differences and structural integrity.
    """
    parser = PDB.PDBParser(QUIET=True)
    input_structure = parser.get_structure('input', input_pdb)
    output_structure = parser.get_structure('output', output_pdb)
    
    print("\nDetailed Structure Comparison")
    print("=" * 50)
    
    # Get residues from both structures
    input_residues = list(input_structure.get_residues())
    output_residues = list(output_structure.get_residues())
    
    print(f"Input residues: {len(input_residues)}")
    print(f"Output residues: {len(output_residues)}")
    
    # Compare residue by residue
    for i, (in_res, out_res) in enumerate(zip(input_residues, output_residues)):
        in_atoms = list(in_res.get_atoms())
        out_atoms = list(out_res.get_atoms())
        
        print(f"\nResidue {i+1} ({in_res.get_resname()}):")
        print(f"  Input atoms: {len(in_atoms)}")
        print(f"  Output atoms: {len(out_atoms)}")
        
        # Compare common atoms
        in_atom_dict = {atom.get_name(): atom for atom in in_atoms}
        out_atom_dict = {atom.get_name(): atom for atom in out_atoms}
        
        common_atoms = set(in_atom_dict.keys()) & set(out_atom_dict.keys())
        missing_atoms = set(in_atom_dict.keys()) - set(out_atom_dict.keys())
        extra_atoms = set(out_atom_dict.keys()) - set(in_atom_dict.keys())
        
        if missing_atoms:
            print(f"  Missing atoms: {', '.join(sorted(missing_atoms))}")
        if extra_atoms:
            print(f"  Extra atoms: {', '.join(sorted(extra_atoms))}")
        
        # Calculate RMSD for common atoms
        if common_atoms:
            coords1 = np.array([in_atom_dict[name].coord for name in sorted(common_atoms)])
            coords2 = np.array([out_atom_dict[name].coord for name in sorted(common_atoms)])
            rmsd = np.sqrt(np.mean(np.sum((coords1 - coords2)**2, axis=1)))
            print(f"  RMSD for common atoms: {rmsd:.3f} Å")


def visualize_assembly_quality(input_coords, assembled_coords, sequence):
    """Create a simple visualization of assembly quality."""
    import matplotlib.pyplot as plt
    
    # Calculate per-residue RMSD
    per_residue_rmsd = np.sqrt(np.sum((input_coords - assembled_coords)**2, axis=1))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot per-residue RMSD
    ax1.bar(range(len(sequence)), per_residue_rmsd)
    ax1.set_xlabel('Residue Position')
    ax1.set_ylabel('RMSD (Å)')
    ax1.set_title('Per-Residue RMSD After Assembly')
    ax1.set_xticks(range(len(sequence)))
    ax1.set_xticklabels(list(sequence), rotation=0)
    
    # Plot coordinate comparison
    for i in range(3):
        ax2.scatter(input_coords[:, i], assembled_coords[:, i], 
                   alpha=0.6, label=f'Coordinate {["X", "Y", "Z"][i]}')
    
    # Add diagonal line
    min_val = min(input_coords.min(), assembled_coords.min())
    max_val = max(input_coords.max(), assembled_coords.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    
    ax2.set_xlabel('Input Coordinates')
    ax2.set_ylabel('Assembled Coordinates')
    ax2.set_title('Coordinate Correlation')
    ax2.legend()
    ax2.axis('equal')
    
    plt.tight_layout()
    plt.savefig('assembly_quality.png', dpi=300)
    print(f"\n   Quality plot saved to: assembly_quality.png")


# Example usage
if __name__ == "__main__":
    # Configuration
    DATABASE_PATH = sys.argv[2]  # Update this
    INPUT_PDB = sys.argv[1]  # Your input PDB file
    OUTPUT_PDB = "assembled_structure.pdb"  # Output file name
    
    # Run the pipeline
    run_assembly_pipeline(DATABASE_PATH, INPUT_PDB, OUTPUT_PDB)
    
    # Optional: Detailed structure comparison
    if os.path.exists(OUTPUT_PDB):
        compare_structures_detailed(INPUT_PDB, OUTPUT_PDB)
    
    # Optional: Create quality visualization
    # Uncomment the following lines if you have matplotlib installed
    # sequence, input_coords = load_input_structure(INPUT_PDB)
    # parser = PDB.PDBParser(QUIET=True)
    # output_structure = parser.get_structure('output', OUTPUT_PDB)
    # output_coords = get_centers_of_mass_from_structure(output_structure)
    # visualize_assembly_quality(input_coords, output_coords, sequence)