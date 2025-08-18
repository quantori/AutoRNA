#!/usr/bin/env python3
"""
RNA Template Matching and Assembly Program

This program finds the best matching RNA structures from a template database
and assembles them into a complete structure using progressive alignment and averaging.
"""

import os
import numpy as np
from Bio import PDB
from Bio.PDB import Superimposer, Residue, Atom
from typing import List, Tuple, Dict
import glob
from scipy.spatial.distance import cdist
import copy


def calculate_center_of_mass(residue):
    """Calculate the center of mass for a nucleotide (residue)."""
    atom_coords = np.array([atom.coord for atom in residue.get_atoms()])
    mass = np.ones(len(atom_coords))  # Placeholder: Assuming equal mass for simplicity
    center_of_mass = np.average(atom_coords, axis=0, weights=mass)
    return center_of_mass


def get_centers_of_mass_from_structure(structure):
    """Extract centers of mass for all residues in a structure."""
    centers = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_resname().strip() in ['A', 'U', 'G', 'C']:
                    centers.append(calculate_center_of_mass(residue))
    return np.array(centers)


def get_residues_from_structure(structure):
    """Extract all RNA residues from a structure."""
    residues = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_resname().strip() in ['A', 'U', 'G', 'C']:
                    residues.append(residue)
    return residues


def load_pdb_file(filepath):
    """Load a PDB file and return the structure."""
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('RNA', filepath)
    return structure


def calculate_rmsd(coords1, coords2):
    """Calculate RMSD between two sets of coordinates."""
    if len(coords1) != len(coords2):
        raise ValueError("Coordinate sets must have the same length")
    diff = coords1 - coords2
    return np.sqrt(np.mean(np.sum(diff**2, axis=1)))


def find_best_alignment(template_coords, target_coords):
    """
    Find the best alignment between template and target coordinates.
    Returns the transformation matrix (rotation and translation).
    """
    if len(template_coords) != len(target_coords):
        raise ValueError("Coordinate sets must have the same length")
    
    # Use Kabsch algorithm for optimal superposition
    # Center both sets
    template_centered = template_coords - np.mean(template_coords, axis=0)
    target_centered = target_coords - np.mean(target_coords, axis=0)
    
    # Calculate rotation matrix
    H = template_centered.T @ target_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Ensure proper rotation (det(R) = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Calculate translation
    t = np.mean(target_coords, axis=0) - R @ np.mean(template_coords, axis=0)
    
    return R, t


def apply_transformation_to_atoms(atoms, rotation, translation):
    """Apply rotation and translation to a list of atoms."""
    transformed_atoms = []
    for atom in atoms:
        # Create a copy of the atom
        new_atom = copy.deepcopy(atom)
        # Apply transformation
        new_atom.coord = rotation @ atom.coord + translation
        transformed_atoms.append(new_atom)
    return transformed_atoms


def apply_transformation_to_residue(residue, rotation, translation):
    """Apply rotation and translation to all atoms in a residue."""
    for atom in residue:
        atom.coord = rotation @ atom.coord + translation


def validate_rna_backbone(structure):
    """
    Validate that all RNA residues in the structure have complete backbone atoms.
    Returns True if all backbone atoms are present, False otherwise.
    """
    # Essential RNA backbone atoms (without P for first residue)
    essential_backbone = {"O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'"}
    phosphate_atoms = {"P", "OP1", "OP2"}
    
    residues = get_residues_from_structure(structure)
    if not residues:
        return False
    
    for i, residue in enumerate(residues):
        atom_names = set(atom.get_name() for atom in residue.get_atoms())
        
        # Check essential backbone atoms
        missing_backbone = essential_backbone - atom_names
        if missing_backbone:
            return False
        
        # Check phosphate atoms (except for first residue)
        if i > 0:
            missing_phosphate = {"P"} - atom_names  # At minimum need P
            if missing_phosphate:
                return False
    
    return True


def find_best_template(database_path, subsequence, target_coords):
    """
    Find the best matching template for a given subsequence.
    Returns the best structure and transformation parameters.
    Only considers templates with complete backbone atoms.
    """
    subfolder = os.path.join(database_path, subsequence)
    if not os.path.exists(subfolder):
        raise ValueError(f"Subfolder {subfolder} not found")
    
    pdb_files = glob.glob(os.path.join(subfolder, "*.pdb"))
    if not pdb_files:
        raise ValueError(f"No PDB files found in {subfolder}")
    
    best_rmsd = float('inf')
    best_structure = None
    best_transformation = None
    best_residues = None
    
    valid_templates = 0
    rejected_templates = 0
    
    for pdb_file in pdb_files:
        try:
            structure = load_pdb_file(pdb_file)
            
            # Validate backbone completeness
            if not validate_rna_backbone(structure):
                print(f"    Skipping {os.path.basename(pdb_file)}: incomplete backbone")
                rejected_templates += 1
                continue
            
            valid_templates += 1
            template_coords = get_centers_of_mass_from_structure(structure)
            
            if len(template_coords) != len(target_coords):
                continue
            
            # Find best alignment
            R, t = find_best_alignment(template_coords, target_coords)
            
            # Calculate RMSD after alignment
            aligned_coords = (R @ template_coords.T).T + t
            rmsd = calculate_rmsd(aligned_coords, target_coords)
            
            if rmsd < best_rmsd:
                best_rmsd = rmsd
                best_structure = pdb_file
                best_transformation = (R, t)
                # Get residues for later use
                best_residues = get_residues_from_structure(structure)
                
        except Exception as e:
            print(f"    Error processing {pdb_file}: {e}")
            continue
    
    if valid_templates == 0:
        raise ValueError(f"No valid templates with complete backbone found in {subfolder}")
    
    print(f"    Evaluated {valid_templates} valid templates ({rejected_templates} rejected)")
    
    return best_structure, best_transformation, best_residues, best_rmsd


def average_atom_positions(atoms_list1, atoms_list2):
    """
    Average the positions of corresponding atoms from two lists.
    Assumes atoms are in the same order and have the same names.
    Returns new atom objects with averaged positions.
    """
    averaged_atoms = []
    
    # Create a mapping of atom names to atoms for each list
    atoms_dict1 = {atom.get_name(): atom for atom in atoms_list1}
    atoms_dict2 = {atom.get_name(): atom for atom in atoms_list2}
    
    # Get all unique atom names
    all_atom_names = set(atoms_dict1.keys()) | set(atoms_dict2.keys())
    
    # Element mapping for RNA atoms
    element_map = {
        'P': 'P', 'O': 'O', 'N': 'N', 'C': 'C', 'H': 'H',
        # Handle primed atoms
        "O5'": 'O', "C5'": 'C', "C4'": 'C', "O4'": 'O',
        "C3'": 'C', "O3'": 'O', "C2'": 'C', "O2'": 'O', "C1'": 'C',
        # Handle base atoms
        'N1': 'N', 'N2': 'N', 'N3': 'N', 'N4': 'N', 'N6': 'N', 'N7': 'N', 'N9': 'N',
        'C2': 'C', 'C4': 'C', 'C5': 'C', 'C6': 'C', 'C8': 'C',
        'O2': 'O', 'O4': 'O', 'O6': 'O',
        # Phosphate oxygens
        'OP1': 'O', 'OP2': 'O', 'OP3': 'O'
    }
    
    def get_element(atom_name):
        """Get element from atom name."""
        if atom_name in element_map:
            return element_map[atom_name]
        # Default: first character of atom name
        return atom_name[0] if atom_name else 'X'
    
    for atom_name in sorted(all_atom_names):
        if atom_name in atoms_dict1 and atom_name in atoms_dict2:
            # Both structures have this atom - average the positions
            atom1 = atoms_dict1[atom_name]
            atom2 = atoms_dict2[atom_name]
            
            # Create a new atom with averaged position
            avg_coord = (atom1.coord + atom2.coord) / 2.0
            
            # Create new atom with averaged properties
            new_atom = Atom.Atom(
                atom_name,
                avg_coord,
                (atom1.get_bfactor() + atom2.get_bfactor()) / 2.0,
                (atom1.get_occupancy() + atom2.get_occupancy()) / 2.0,
                atom1.get_altloc(),  # Use first atom's altloc
                atom_name.ljust(4),  # Full name
                0,  # Serial number
                element=get_element(atom_name)
            )
            averaged_atoms.append(new_atom)
        elif atom_name in atoms_dict1:
            # Only first structure has this atom - create a copy
            atom1 = atoms_dict1[atom_name]
            new_atom = Atom.Atom(
                atom_name,
                atom1.coord.copy(),
                atom1.get_bfactor(),
                atom1.get_occupancy(),
                atom1.get_altloc(),
                atom_name.ljust(4),
                0,  # Serial number
                element=get_element(atom_name)
            )
            averaged_atoms.append(new_atom)
        else:
            # Only second structure has this atom - create a copy
            atom2 = atoms_dict2[atom_name]
            new_atom = Atom.Atom(
                atom_name,
                atom2.coord.copy(),
                atom2.get_bfactor(),
                atom2.get_occupancy(),
                atom2.get_altloc(),
                atom_name.ljust(4),
                0,  # Serial number
                element=get_element(atom_name)
            )
            averaged_atoms.append(new_atom)
    
    return averaged_atoms


def assemble_structure(database_path, input_sequence, input_coords, window_size=4):
    """
    - Left boundary (0-2): Use template from window 0-3 directly
    - Internal (3-6): Average overlapping templates  
    - Right boundary (7-9): Use template from window 6-9 directly
    """
    n = len(input_sequence)
    if n < window_size:
        raise ValueError(f"Input sequence too short (minimum {window_size} nucleotides required)")
    
    # Define boundary regions (first 3 and last 3 nucleotides)
    left_boundary = 3
    right_boundary = n - 3
    
    # Store assembled residues with full atomic detail
    assembled_residues = [None] * n
    residue_weights = [0] * n  # Track how many times each position has been averaged
    best_templates_info = {}  # Store best template info for each window
    
    # First pass: Find best templates for all windows and store info
    print("First pass: Finding best templates for all windows...")
    for i in range(n - window_size + 1):
        subsequence = input_sequence[i:i+window_size]
        target_coords = input_coords[i:i+window_size]
        
        print(f"Finding best template for window {i}-{i+window_size-1}: {subsequence}")
        
        try:
            # Find best template for this window
            template_file, transformation, template_residues, rmsd = find_best_template(
                database_path, subsequence, target_coords
            )
            
            if transformation is None:
                print(f"  No suitable template found for {subsequence}")
                continue
                
            print(f"  Best template: {template_file} (RMSD: {rmsd:.3f})")
            
            best_templates_info[i] = {
                'file': template_file,
                'transformation': transformation,
                'residues': template_residues,
                'rmsd': rmsd,
                'window_start': i,
                'window_end': i + window_size
            }
            
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    # Second pass: Assemble structure with special handling for boundaries
    print("\nSecond pass: Assembling structure...")
    
    # Handle left boundary (first 3 nucleotides)
    if 0 in best_templates_info:
        print(f"Setting left boundary nucleotides (0-{left_boundary-1}) from first template")
        info = best_templates_info[0]
        R, t = info['transformation']
        
        for j in range(min(window_size, left_boundary)):
            residue = info['residues'][j]
            new_residue = copy.deepcopy(residue)
            apply_transformation_to_residue(new_residue, R, t)
            assembled_residues[j] = new_residue
            residue_weights[j] = 1
    
    # Handle right boundary (last 3 nucleotides)
    last_window_start = n - window_size
    if last_window_start in best_templates_info and last_window_start >= 0:
        print(f"Setting right boundary nucleotides ({right_boundary}-{n-1}) from last template")
        info = best_templates_info[last_window_start]
        R, t = info['transformation']
        
        for j in range(max(0, right_boundary - last_window_start), window_size):
            global_pos = last_window_start + j
            if global_pos >= right_boundary:
                residue = info['residues'][j]
                new_residue = copy.deepcopy(residue)
                apply_transformation_to_residue(new_residue, R, t)
                assembled_residues[global_pos] = new_residue
                residue_weights[global_pos] = 1
    
    # Handle internal positions with averaging
    for i in sorted(best_templates_info.keys()):
        info = best_templates_info[i]
        R, t = info['transformation']
        
        # Apply transformation to all atoms in template residues
        transformed_residues = []
        for residue in info['residues']:
            new_residue = copy.deepcopy(residue)
            apply_transformation_to_residue(new_residue, R, t)
            transformed_residues.append(new_residue)
        
        # Add/average residues into the assembled structure
        for j, residue in enumerate(transformed_residues):
            global_pos = i + j
            
            # Skip boundary positions (they're already set)
            if global_pos < left_boundary or global_pos >= right_boundary:
                continue
            
            if assembled_residues[global_pos] is None:
                # First time seeing this position
                assembled_residues[global_pos] = copy.deepcopy(residue)
                residue_weights[global_pos] = 1
            else:
                # Average with existing residue (only for internal positions)
                existing_atoms = list(assembled_residues[global_pos].get_atoms())
                new_atoms = list(residue.get_atoms())
                
                # Average atom positions
                averaged_atoms = average_atom_positions(existing_atoms, new_atoms)
                
                # Create a new residue with averaged atoms
                # We need to create a fresh residue to avoid atom conflicts
                res_name = assembled_residues[global_pos].get_resname()
                res_id = assembled_residues[global_pos].get_id()
                
                # Remove the old residue from its parent (if it has one)
                parent = assembled_residues[global_pos].get_parent()
                if parent:
                    parent.detach_child(res_id)
                
                # Create new residue with averaged atoms
                new_residue = Residue.Residue(res_id, res_name, ' ')
                
                # Add all averaged atoms to the new residue
                for atom in averaged_atoms:
                    new_residue.add(atom)
                
                # Replace the residue in our list
                assembled_residues[global_pos] = new_residue
                residue_weights[global_pos] += 1
    
    # Final check: Fill any remaining gaps in boundaries with best available templates
    for pos in range(n):
        if assembled_residues[pos] is None:
            print(f"Warning: Position {pos} still empty, searching for coverage...")
            # Find a window that covers this position
            for window_start, info in best_templates_info.items():
                if window_start <= pos < window_start + window_size:
                    R, t = info['transformation']
                    local_pos = pos - window_start
                    residue = info['residues'][local_pos]
                    new_residue = copy.deepcopy(residue)
                    apply_transformation_to_residue(new_residue, R, t)
                    assembled_residues[pos] = new_residue
                    residue_weights[pos] = 1
                    print(f"  Filled position {pos} from window {window_start}-{window_start+window_size-1}")
                    break
    
    return assembled_residues, residue_weights


def save_assembled_structure(output_file, assembled_residues, sequence):
    """Save the assembled structure with full atomic detail as a PDB file."""
    # Create a new structure
    builder = PDB.StructureBuilder.StructureBuilder()
    builder.init_structure("assembled")
    builder.init_model(0)
    builder.init_chain("A")
    
    # Add residues with all atoms
    for i, (residue, base) in enumerate(zip(assembled_residues, sequence)):
        if residue is not None:
            # Create new residue
            builder.init_residue(base, ' ', i+1, ' ')
            
            # Add all atoms from the assembled residue
            for atom in residue.get_atoms():
                builder.init_atom(
                    atom.get_name(),
                    atom.coord,
                    atom.get_bfactor(),
                    atom.get_occupancy(),
                    atom.get_altloc(),
                    atom.get_fullname(),
                    element=atom.element
                )
        else:
            print(f"Warning: No structure for position {i+1} ({base})")
    
    structure = builder.get_structure()
    
    # Save to PDB file
    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(output_file)


def validate_assembled_structure(assembled_residues):
    """Validate the assembled structure and report any issues."""
    issues = []
    
    for i, residue in enumerate(assembled_residues):
        if residue is None:
            issues.append(f"Position {i+1}: No structure assembled")
        else:
            atom_count = len(list(residue.get_atoms()))
            if atom_count < 10:  # RNA nucleotides typically have >10 atoms
                issues.append(f"Position {i+1}: Only {atom_count} atoms (possibly incomplete)")
    
    return issues


def main():
    """Main execution function."""
    # Example usage
    database_path = "/home/leo/Research/NewRNAProject/4th_letters/output"  # Update this path
    
    # Example input data
    input_sequence = "AAAUUGCCGA"  # Your RNA sequence
    
    # Generate example input coordinates (replace with your actual data)
    # In real usage, these would come from your input PDB file
    n = len(input_sequence)
    input_coords = np.random.randn(n, 3) * 10  # Example coordinates
    
    # Run assembly
    print(f"Assembling structure for sequence: {input_sequence}")
    print(f"Number of nucleotides: {n}")
    print(f"Boundary handling: First 3 and last 3 nucleotides will use best templates directly")
    
    try:
        assembled_residues, residue_weights = assemble_structure(
            database_path, input_sequence, input_coords
        )
        
        # Validate assembly
        issues = validate_assembled_structure(assembled_residues)
        if issues:
            print("\nAssembly issues found:")
            for issue in issues:
                print(f"  - {issue}")
        
        # Save result
        output_file = "assembled_structure.pdb"
        save_assembled_structure(output_file, assembled_residues, input_sequence)
        
        print(f"\nAssembly complete!")
        print(f"Output saved to: {output_file}")
        
        # Print summary
        print(f"\nAssembly summary:")
        boundary_positions = list(range(3)) + list(range(n-3, n))
        for i, weight in enumerate(residue_weights):
            if i in boundary_positions:
                print(f"  Position {i+1} ({input_sequence[i]}): boundary position (no averaging)")
            else:
                print(f"  Position {i+1} ({input_sequence[i]}): averaged from {weight} templates")
            
    except Exception as e:
        print(f"Error during assembly: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()