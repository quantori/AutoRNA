#!/usr/bin/env python3
"""
RNA MD refinement with proper water force field handling
"""

import argparse
import sys
import numpy as np
from openmm import *
from openmm.app import *
from openmm.unit import *

class RNARefinementFixed:
    def __init__(self, pdb_file):
        """Initialize RNA refinement with fixed water handling"""
        self.pdb = PDBFile(pdb_file)
        self.forcefield = None
        self.modeller = None
        self.system = None
        self.simulation = None
        
    def load_forcefield(self):
        """Load force field with proper water model"""
        print("Loading force fields...")
        
        # Try different combinations that include water
        ff_combinations = [
            # Option 1: Separate files for RNA and water
            (['amber14/RNA.OL3.xml', 'amber14/tip3pfb.xml'], 'AMBER14 RNA + TIP3P-FB'),
            (['amber14/RNA.OL3.xml', 'amber14/tip3p.xml'], 'AMBER14 RNA + TIP3P'),
            
            # Option 2: All-in-one files
            (['amber14-all.xml'], 'AMBER14-all'),
            
            # Option 3: Older versions
            (['amber99sbildn.xml', 'amber99_obc.xml', 'tip3p.xml'], 'AMBER99 + TIP3P'),
            (['amber99sb.xml', 'tip3p.xml'], 'AMBER99SB + TIP3P'),
        ]
        
        for ff_files, ff_name in ff_combinations:
            try:
                print(f"  Trying {ff_name}...")
                self.forcefield = ForceField(*ff_files)
                
                # Test with a simple system
                modeller = Modeller(self.pdb.topology, self.pdb.positions)
                test_system = self.forcefield.createSystem(
                    modeller.topology,
                    nonbondedMethod=NoCutoff
                )
                
                print(f"  ✓ Loaded {ff_name} successfully")
                return True
                
            except Exception as e:
                print(f"    Failed: {str(e)[:60]}...")
                continue
        
        print("\n✗ Could not load force field with water model!")
        return False
    
    def prepare_system(self, padding_nm=1.0, ionic_strength_molar=0.15):
        """Prepare system with explicit water model loading"""
        print("\nPreparing RNA system...")
        
        # Load force field
        if not self.load_forcefield():
            raise RuntimeError("Failed to load compatible force field")
        
        # Create modeller
        self.modeller = Modeller(self.pdb.topology, self.pdb.positions)
        
        # Check and add hydrogens
        has_h = any(atom.element == element.hydrogen 
                   for atom in self.modeller.topology.atoms())
        
        if not has_h:
            print("Adding hydrogen atoms...")
            self.modeller.addHydrogens(self.forcefield, pH=7.0)
            print("✓ Hydrogens added")
        
        # Debug: Print topology info before adding solvent
        print("\nTopology before solvent:")
        residue_counts = {}
        for residue in self.modeller.topology.residues():
            res_name = residue.name
            residue_counts[res_name] = residue_counts.get(res_name, 0) + 1
        
        for res_name, count in sorted(residue_counts.items()):
            print(f"  {res_name}: {count}")
        
        # Add solvent with explicit model
        print(f"\nAdding solvent (padding={padding_nm} nm)...")
        print("  Using water model from force field")
        
        try:
            # Get the initial number of residues
            n_residues_before = len(list(self.modeller.topology.residues()))
            
            # Add solvent
            self.modeller.addSolvent(
                self.forcefield,
                model='tip3p',  # Explicitly specify water model
                padding=padding_nm*nanometer,
                ionicStrength=ionic_strength_molar*molar,
                positiveIon='Na+',
                negativeIon='Cl-'
            )
            
            # Check what was added
            n_residues_after = len(list(self.modeller.topology.residues()))
            n_water_added = n_residues_after - n_residues_before
            
            print(f"✓ Added {n_water_added} water/ion molecules")
            
        except Exception as e:
            print(f"✗ Error adding solvent: {e}")
            print("\nDebug information:")
            print(f"  Force field files: {self.forcefield._files}")
            raise
        
        # Create system
        print("\nCreating simulation system...")
        self.system = self.forcefield.createSystem(
            self.modeller.topology,
            nonbondedMethod=PME,
            nonbondedCutoff=1.0*nanometer,
            constraints=HBonds
        )
        
        print(f"✓ System created with {self.system.getNumParticles()} particles")
    
    def add_restraints(self, k_start=100.0):
        """Add position restraints to RNA heavy atoms"""
        restraint = CustomExternalForce("k*periodicdistance(x, y, z, x0, y0, z0)^2")
        restraint.addGlobalParameter("k", k_start*kilojoule_per_mole/nanometer**2)
        restraint.addPerParticleParameter("x0")
        restraint.addPerParticleParameter("y0")
        restraint.addPerParticleParameter("z0")
        
        self.reference_positions = self.modeller.positions
        
        n_restrained = 0
        for atom in self.modeller.topology.atoms():
            if (atom.residue.name in ['A', 'G', 'C', 'U'] and 
                atom.element != element.hydrogen):
                restraint.addParticle(
                    atom.index,
                    self.reference_positions[atom.index]
                )
                n_restrained += 1
        
        self.system.addForce(restraint)
        self.restraint_force = restraint
        
        print(f"Added restraints to {n_restrained} RNA heavy atoms")
    
    def setup_simulation(self, temperature_k=300.0):
        """Set up MD simulation"""
        integrator = LangevinMiddleIntegrator(
            temperature_k*kelvin,
            1.0/picosecond,
            0.00002*picoseconds
        )
        
        self.simulation = Simulation(
            self.modeller.topology,
            self.system,
            integrator
        )
        
        self.simulation.context.setPositions(self.modeller.positions)
        
        self.simulation.reporters.append(
            StateDataReporter(
                sys.stdout, 1000,
                step=True,
                potentialEnergy=True,
                temperature=True
            )
        )
    
    def minimize(self, max_iterations=50000000):
        """Energy minimization"""
        print(f"Energy minimization ({max_iterations} steps)...")
        self.simulation.minimizeEnergy(maxIterations=max_iterations)
    
    def run_md(self, steps, k_restraint=None):
        """Run MD with optional restraints"""
        if k_restraint is not None:
            self.simulation.context.setParameter('k', k_restraint*kilojoule_per_mole/nanometer**2)
        self.simulation.step(steps)
    
    def run_refinement(self, n_iterations=10, steps_per_iter=1000000):
        """Iterative refinement protocol"""
        # Initial minimization
        self.minimize(100000000)
        
        # Restraint schedule
        k_values = np.logspace(2, -1, n_iterations)
        
        # Iterative refinement
        for i in range(n_iterations):
            print(f"\n--- Iteration {i+1}/{n_iterations} ---")
            k = k_values[i]
            print(f"Restraint constant: {k:.1f} kJ/mol/nm²")
            
            self.run_md(steps_per_iter, k_restraint=k)
            self.minimize(100000000)
        
        # Final unrestrained
        print("\nFinal unrestrained MD...")
        self.run_md(steps_per_iter, k_restraint=0.0)
        
        print("\nFinal minimization...")
        self.minimize(2000000000)
    
    def save_rna_only(self, output_file):
        """Save only RNA residues"""
        state = self.simulation.context.getState(getPositions=True)
        positions = state.getPositions()
        
        # New topology with only RNA
        new_topology = Topology()
        new_positions = []
        
        # Map chains
        chain_map = {}
        for chain in self.modeller.topology.chains():
            # Only create chain if it has RNA
            has_rna = any(r.name in ['A', 'G', 'C', 'U'] for r in chain.residues())
            if has_rna:
                new_chain = new_topology.addChain(chain.id)
                chain_map[chain] = new_chain
        
        # Add RNA residues
        for residue in self.modeller.topology.residues():
            if residue.name in ['A', 'G', 'C', 'U']:
                new_residue = new_topology.addResidue(
                    residue.name,
                    chain_map[residue.chain]
                )
                
                for atom in residue.atoms():
                    new_atom = new_topology.addAtom(
                        atom.name,
                        atom.element,
                        new_residue
                    )
                    new_positions.append(positions[atom.index])
        
        # Save
        print(f"\nSaving refined RNA to {output_file}...")
        positions_array = []
        for pos in new_positions:
            positions_array.append([
                pos[0].value_in_unit(nanometer),
                pos[1].value_in_unit(nanometer),
                pos[2].value_in_unit(nanometer)
            ])

        # Create proper Quantity object
        final_positions = Quantity(np.array(positions_array), nanometer)
        with open(output_file, 'w') as f:
            PDBFile.writeFile(new_topology, final_positions, f)
        
        print(f"✓ Saved {len(final_positions)} RNA atoms")

def test_water_loading():
    """Test if water models load correctly"""
    print("Testing water model loading...")
    
    # Create a simple test topology with just water
    topology = Topology()
    chain = topology.addChain()
    residue = topology.addResidue('HOH', chain)
    topology.addAtom('O', element.oxygen, residue)
    topology.addAtom('H1', element.hydrogen, residue)
    topology.addAtom('H2', element.hydrogen, residue)
    
    positions = [
        (0, 0, 0)*nanometer,
        (0.1, 0, 0)*nanometer,
        (-0.03, 0.1, 0)*nanometer
    ]
    
    # Test different water force fields
    water_ffs = [
        'tip3p.xml',
        'tip3pfb.xml', 
        'amber14/tip3p.xml',
        'amber14/tip3pfb.xml'
    ]
    
    for ff_file in water_ffs:
        try:
            ff = ForceField(ff_file)
            system = ff.createSystem(topology)
            print(f"{ff_file} works")
        except Exception as e:
            print(f"{ff_file} failed: {e}")

def main():
    parser = argparse.ArgumentParser(
        description='RNA MD refinement with fixed water handling'
    )
    parser.add_argument('input_pdb', help='Input RNA PDB file')
    parser.add_argument('output_pdb', help='Output refined PDB file')
    parser.add_argument('--iterations', type=int, default=50)
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--temperature', type=float, default=300.0)
    parser.add_argument('--padding', type=float, default=2.0)
    parser.add_argument('--test-water', action='store_true',
                       help='Test water model loading')
    
    args = parser.parse_args()
    
    if args.test_water:
        test_water_loading()
        return 0
    
    try:
        # Create refinement object
        refiner = RNARefinementFixed(args.input_pdb)
        
        # Prepare system
        refiner.prepare_system(padding_nm=args.padding)
        
        # Add restraints
        refiner.add_restraints()
        
        # Set up simulation
        refiner.setup_simulation(temperature_k=args.temperature)
        
        # Run refinement
        refiner.run_refinement(
            n_iterations=args.iterations,
            steps_per_iter=args.steps
        )
        
        # Save result
        refiner.save_rna_only(args.output_pdb)
        
        print("\n✓ Refinement completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())