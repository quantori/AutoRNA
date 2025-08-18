#!/usr/bin/env python3
"""
Fix RNA structures with disconnected terminal phosphate groups
"""

import sys
import re
from openmm.app import *
from openmm import *
from openmm.unit import *

def diagnose_pdb(filename):
    """Diagnose issues with RNA PDB file"""
    print(f"\nDiagnosing {filename}...")
    print("=" * 60)
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Parse atoms by residue
    residues = {}
    atom_count = 0
    
    for line in lines:
        if line.startswith('ATOM'):
            atom_count += 1
            res_num = int(line[22:26].strip())
            res_name = line[17:20].strip()
            atom_name = line[12:16].strip()
            chain = line[21]
            
            if res_num not in residues:
                residues[res_num] = {
                    'name': res_name,
                    'chain': chain,
                    'atoms': []
                }
            residues[res_num]['atoms'].append(atom_name)
    
    print(f"Total atoms: {atom_count}")
    print(f"Residues found: {len(residues)}")
    
    # Check each residue
    for res_num in sorted(residues.keys()):
        res = residues[res_num]
        atoms = res['atoms']
        print(f"\nResidue {res_num} ({res['name']}):")
        print(f"  Atoms: {len(atoms)}")
        
        # Check for phosphate atoms
        has_P = 'P' in atoms
        has_OP1 = 'OP1' in atoms
        has_OP2 = 'OP2' in atoms
        has_HO5 = "HO5'" in atoms
        has_HO3 = "HO3'" in atoms
        
        if has_P:
            print(f"  Has phosphate: P={has_P}, OP1={has_OP1}, OP2={has_OP2}")
        if has_HO5:
            print(f"  Has 5'-OH (terminal marker)")
        if has_HO3:
            print(f"  Has 3'-OH (terminal marker)")
        
        # Identify terminal type
        if res_num == min(residues.keys()):
            if has_HO5 and has_P:
                print("  ⚠️  WARNING: 5'-terminal has both HO5' and phosphate!")
            elif has_HO5:
                print("  ✓ Proper 5'-terminal")
        
        if res_num == max(residues.keys()):
            if has_HO3:
                print("  ✓ Proper 3'-terminal")
            else:
                print("  ⚠️  Missing HO3' for 3'-terminal")

def fix_disconnected_phosphate(input_file, output_file):
    """Fix RNA with disconnected terminal phosphate"""
    print(f"\nFixing {input_file}...")
    
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # First pass: identify problematic phosphate atoms
    first_residue_atoms = []
    phosphate_atoms = []
    other_lines = []
    
    for line in lines:
        if line.startswith('ATOM'):
            res_num = int(line[22:26].strip())
            atom_name = line[12:16].strip()
            
            if res_num == 1:
                # Check if this is a phosphate atom that should be removed
                if atom_name in ['P', 'OP1', 'OP2'] and "HO5'" in [l[12:16].strip() for l in first_residue_atoms]:
                    # This residue already has HO5', so phosphate should be removed
                    phosphate_atoms.append(line)
                    print(f"  Removing disconnected phosphate atom: {atom_name}")
                else:
                    first_residue_atoms.append(line)
            else:
                other_lines.append(line)
        else:
            other_lines.append(line)
    
    # Check if we found the issue
    if phosphate_atoms and any("HO5'" in line for line in first_residue_atoms):
        print("  Found 5'-terminal with both HO5' and phosphate - removing phosphate")
        
        # Rebuild the file without the disconnected phosphate
        new_lines = []
        
        # Add header
        for line in lines:
            if line.startswith('REMARK') or line.startswith('CRYST'):
                new_lines.append(line)
            elif line.startswith('ATOM'):
                break
        
        # Add first residue atoms (without phosphate)
        new_lines.extend(first_residue_atoms)
        
        # Add remaining atoms
        new_lines.extend(other_lines)
        
        # Renumber atoms
        atom_num = 1
        final_lines = []
        for line in new_lines:
            if line.startswith('ATOM'):
                new_line = line[:6] + f"{atom_num:5d}" + line[11:]
                final_lines.append(new_line)
                atom_num += 1
            else:
                final_lines.append(line)
        
        # Write output
        with open(output_file, 'w') as f:
            f.writelines(final_lines)
        
        print(f"\n✓ Fixed structure written to {output_file}")
        print(f"  Removed {len(phosphate_atoms)} phosphate atoms from 5'-terminal")
        
    else:
        print("\n⚠️  No disconnected phosphate issue found")
        print("  Copying original file...")
        with open(output_file, 'w') as f:
            f.writelines(lines)

def validate_fixed_structure(filename):
    """Validate the fixed structure with OpenMM"""
    print(f"\nValidating {filename}...")
    
    try:
        pdb = PDBFile(filename)
        forcefield = ForceField('amber14-all.xml')
        
        # Create modeller
        modeller = Modeller(pdb.topology, pdb.positions)
        
        # Test system creation
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=NoCutoff
        )
        
        print("✓ Structure is valid for OpenMM!")
        print(f"  System has {system.getNumParticles()} particles")
        
        # Test adding solvent
        print("\nTesting solvent addition...")
        modeller.addSolvent(
            forcefield,
            padding=1.0*nanometer,
            ionicStrength=0.15*molar
        )
        print("✓ Solvent addition successful!")
        
        return True
        
    except Exception as e:
        print(f"✗ Validation failed: {e}")
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python fix_rna_terminal_phosphate.py input.pdb [output.pdb]")
        print("\nThis script fixes RNA structures where the 5'-terminal has")
        print("both HO5' and a disconnected phosphate group.")
        return 1
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else input_file.replace('.pdb', '_fixed.pdb')
    
    # Diagnose the issue
    diagnose_pdb(input_file)
    
    # Fix the issue
    fix_disconnected_phosphate(input_file, output_file)
    
    # Validate the fix
    if output_file != input_file:
        validate_fixed_structure(output_file)
    
    print("\nNext steps:")
    print(f"1. Check the structure: pymol {output_file}")
    print(f"2. Run MD refinement: python rna_refine_final.py {output_file} refined.pdb")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())