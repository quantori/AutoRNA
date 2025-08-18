#!/usr/bin/env python3
"""
Fix common issues in RNA structures for OpenMM simulations
"""

import sys
from openmm.app import *
from openmm import *
from openmm.unit import *

def fix_rna_for_openmm(input_pdb, output_pdb):
    """
    Fix RNA structure for OpenMM by ensuring proper atom naming and adding hydrogens
    """
    print(f"Processing {input_pdb}...")
    
    # Load structure
    pdb = PDBFile(input_pdb)
    
    # Try different force fields in order of preference
    forcefield = None
    ff_options = [
        ('amber14-all.xml', 'AMBER14-all'),
        ('amber14/RNA.OL3.xml', 'AMBER14 RNA.OL3'),
        ('amber99sb.xml', 'AMBER99SB')
    ]
    
    for ff_file, ff_name in ff_options:
        try:
            print(f"Trying {ff_name}...")
            forcefield = ForceField(ff_file)
            
            # Test if it works with the structure
            modeller = Modeller(pdb.topology, pdb.positions)
            test_system = forcefield.createSystem(
                modeller.topology,
                nonbondedMethod=NoCutoff
            )
            print(f"✓ {ff_name} is compatible")
            break
        except:
            continue
    
    if forcefield is None:
        print("Error: No compatible force field found!")
        print("\nTroubleshooting steps:")
        print("1. Ensure RNA residues are named A, G, C, U (not ADE, GUA, CYT, URI)")
        print("2. Use pdb4amber to prepare the structure:")
        print(f"   pdb4amber -i {input_pdb} -o {output_pdb} --add-missing-atoms")
        return False
    
    # Create modeller and add hydrogens
    modeller = Modeller(pdb.topology, pdb.positions)
    
    # Check if hydrogens are needed
    has_h = any(atom.element == element.hydrogen for atom in modeller.topology.atoms())
    
    if not has_h:
        print("Adding hydrogen atoms...")
        modeller.addHydrogens(forcefield, pH=7.0)
        print("✓ Hydrogens added")
    else:
        print("✓ Hydrogens already present")
    
    # Save the fixed structure
    print(f"Saving to {output_pdb}...")
    with open(output_pdb, 'w') as f:
        PDBFile.writeFile(modeller.topology, modeller.positions, f)
    
    # Summary
    n_residues = len(list(modeller.topology.residues()))
    n_atoms = len(list(modeller.topology.atoms()))
    print(f"\n✓ Structure prepared: {n_residues} residues, {n_atoms} atoms")
    
    return True

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fix_rna_for_openmm.py input.pdb output.pdb")
        sys.exit(1)
    
    success = fix_rna_for_openmm(sys.argv[1], sys.argv[2])
    sys.exit(0 if success else 1)