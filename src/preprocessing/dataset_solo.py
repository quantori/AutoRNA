import os
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from tqdm import tqdm
import itertools
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
three_letter_amino_acids = {'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLU', 'GLN', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'}


class RNA_Dataset:
    RNA_NUCLEOBASE = {'A', 'U', 'G', 'C'}
    RNA_THREE_LETTER_CODES = {'ADE': 'A', 'URA': 'U', 'GUA': 'G', 'CYT': 'C'}
    RESIDUES = set()

    def __init__(self, conf):
        self.conf = conf
        self.type = conf['type']
        self.max_data = conf['max_data']
        self.pdb_folder_path = conf['pdb_folder_path']
        self.min_length = conf['min_length']
        self.max_length = conf['max_length']
        self.parser = PDBParser(QUIET=True)
        self.df = self.process_all_pdb_files()
        print("length before filtering" ,len(self.df))
#        print(self.RESIDUES)
        pd.set_option('display.max_rows', None)
#        print(self.df.head(2000))
#        print(len(self.df))

        self.df = self.drop_empty_sequences()
        print("Dropping empty sequences", len(self.df))

        self.df = self.filter_nonvalid_rna()
        print("Filtering non-valid (size) RNA", len(self.df))

        self.df = self.drop_duplications()
        print("Dropping duplications", len(self.df))

    def get(self):
        return self.df

    def drop_duplications(self):
        print('Dropping duplications in sequences ... ')
        df_no_duplicates = self.df.drop_duplicates(subset=['seq'])
        self.df = df_no_duplicates
        return self.df

    @staticmethod
    def calculate_center_of_mass(residue):
        atom_coords = np.array([atom.coord for atom in residue.get_atoms()])
        mass = np.ones(len(atom_coords))  # Placeholder: Assuming equal mass for simplicity
        center_of_mass = np.average(atom_coords, axis=0, weights=mass)
        return center_of_mass

    """
    @staticmethod
    def calculate_center_of_mass(residue):
        
        # Atomic masses in Daltons (g/mol) for common elements in RNA
        ATOMIC_MASSES = {
            'H': 1.008,    # Hydrogen
            'C': 12.011,   # Carbon
            'N': 14.007,   # Nitrogen
            'O': 15.999,   # Oxygen
            'P': 30.974,   # Phosphorus
            'S': 32.065,   # Sulfur (occasionally in modified nucleotides)
        }
        
        atom_coords = []
        masses = []
        
        for atom in residue.get_atoms():
            # Get atom coordinates
            atom_coords.append(atom.coord)
            
            # Get element from atom - BioPython stores this in the element attribute
            element = atom.element.strip()
            
            # If element is not directly available or is empty, parse from atom name
            if not element:
                # Get first character(s) of atom name that represent the element
                atom_name = atom.get_name().strip()
                if atom_name:
                    # Handle common atom naming conventions
                    if atom_name[0].isdigit():
                        # Sometimes atoms are named like "1HB" - element is second character
                        element = atom_name[1]
                    else:
                        # Usually element is first 1-2 characters
                        if len(atom_name) >= 2 and atom_name[:2] in ATOMIC_MASSES:
                            element = atom_name[:2]
                        else:
                            element = atom_name[0]
            
            # Get the mass for this element
            mass = ATOMIC_MASSES.get(element.upper(), 1.0)  # Default to 1.0 if element not found
            masses.append(mass)
        
        # Convert to numpy arrays
        atom_coords = np.array(atom_coords)
        masses = np.array(masses)
        
        # Calculate center of mass: COM = Σ(mi * ri) / Σ(mi)
        total_mass = np.sum(masses)
        center_of_mass = np.sum(atom_coords * masses[:, np.newaxis], axis=0) / total_mass
        
        return center_of_mass
    """
    def drop_empty_sequences(self):
        print('Dropping empty sequences ... ')
        self.df['seq'] = self.df['seq'].apply(lambda seq: np.nan if len(seq) == 0 else seq)
        self.df.dropna(subset=['seq'], inplace=True)
        return self.df

    def filter_nonvalid_rna(self):
        print('Filtering non-valid RNAs')
        self.df['seq'] = self.df['seq'].apply(
            lambda seq: seq if set(seq).issubset(self.RNA_NUCLEOBASE) and self.min_length < len(seq) < self.max_length else np.nan
        )
        self.df.dropna(subset=['seq'], inplace=True)
        return self.df

    def process_all_pdb_files(self):
        # List all files in the directory (assuming they are all PDB files)
        pdb_files = os.listdir(self.pdb_folder_path)[:self.max_data]  # Limit to max_data if needed
        all_dfs = []  # List to store DataFrames

        # Loop over all PDB files in the directory with a progress bar
        for pdb_file in tqdm(pdb_files, desc="Processing PDB files"):
            pdb_file_path = os.path.join(self.pdb_folder_path, pdb_file)
            if pdb_file.endswith(".ent") or pdb_file.endswith(".pdb"):
                # Check if the file is a PDB file
                pdb_df = self.from_pdb(pdb_file_path)  # Call from_pdb() on each file
                if pdb_df is not None:
                    all_dfs.append(pdb_df)  # Append the DataFrame to the list

        # Concatenate all DataFrames into one
        if all_dfs:
            df = pd.concat(all_dfs, ignore_index=True)
        else:
            df = pd.DataFrame()  # Create an empty DataFrame if no files are found

        return df

    def check_chain_protein(self, chain):
        for residue in chain.get_residues():
            residue_name = residue.get_resname()
            self.RESIDUES.add(residue_name)
            if residue_name in three_letter_amino_acids:
                return True
        return False


    def first_model_or_single(self, structure: Structure) -> tuple[Model, int]:
        # Total number of models in the structure
        n_models = len(structure)
        if n_models == 0:
            raise ValueError("The structure contains no models.")
        
        # Access the first model directly by index
        first_model = structure[0]
        return first_model, n_models

    def from_pdb(self, pdb_file) -> pd.DataFrame:
#        print("Processing PDB file:", pdb_file)
        # Initialize lists to store concatenated data
        concatenated_seq = []  # Concatenated RNA sequence
        concatenated_coords = []  # Concatenated coordinates (centers of mass)
        pdb_ids = []  # List to store a single PDB ID (for each residue)
        unique_chains = [] # Set to store unique chain IDs

        structure = self.parser.get_structure(pdb_file.split('.')[0], pdb_file)
        structure, n_struct = self.first_model_or_single(structure)
        # Skip if chain contains protein residues
        for chain in structure.get_chains():
            if self.check_chain_protein(chain):
                return None

        # Process chains for RNA residues
        #print("new molecule")
        print("pdb_file", pdb_file)
        for chain in structure.get_chains():
            rna_seq = []
            chain_coords = []
            #print("chain")
            for residue in chain.get_residues():
                residue_name = residue.get_resname()
                if residue_name in self.RNA_THREE_LETTER_CODES:
                    residue_name = self.RNA_THREE_LETTER_CODES[residue_name]
                if residue_name in self.RNA_NUCLEOBASE:
                    #print(residue)
                    com = self.calculate_center_of_mass(residue)# , residue["C5'"], residue["O4'"]] #self.calculate_center_of_mass(residue)  # Calculate center of mass
                    rna_seq.append(residue_name)  # Add the one-letter residue name to sequence
                    for elem in com:  # Append center of mass coordinates as a list
                        chain_coords.append(elem)
            if len(chain_coords)>0:
                # Concatenate all chain sequences and coordinates for the file
                concatenated_seq += rna_seq
                concatenated_coords += chain_coords
                pdb_ids += [pdb_file.split('.')[0]] * len(rna_seq)  # Repeat PDB ID for each residue
                first_nucleotide = rna_seq[0]
                last_nucleotide = rna_seq[-1]
                first_coord = chain_coords[0]
                last_coord = chain_coords[-1]
                unique_chains.append((chain.id, rna_seq, chain_coords, first_nucleotide, last_nucleotide, first_coord, last_coord))
        if len(unique_chains)>7: # 2^n*n!
            return None
        concatenated_coords = []
        concatenated_seq = []
        pdb_ids = []
        ordered_chains = self.find_minimum_distance_combination(unique_chains)
        for chain_id, rna_seq, chain_coords, _, _, _, _ in unique_chains:
            concatenated_seq += rna_seq
            concatenated_coords += chain_coords
            pdb_ids += [pdb_file.split('.')[0]] * len(rna_seq)

        if 3*len(concatenated_seq) == len(concatenated_coords) == 3*len(pdb_ids):
            rna_df = {
                'index': [pdb_file.split('/')[-1].split(".")[0]],
                'chains': [[chain_id for chain_id, _, _, _, _, _, _ in ordered_chains]],
                'seq': [''.join(concatenated_seq)],
                'backbones': [concatenated_coords]
            }
            rna_df_n = pd.DataFrame.from_dict(rna_df)
            rna_df_n.to_csv("output_df.csv", index=False, mode='a', header=False)
            return rna_df_n
        else:
            raise ValueError(
                f"Lengths mismatch: {len(concatenated_seq)} sequences, {len(concatenated_coords)} coordinates.")

    def euclidean_distance(self,p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def total_distance(self,chain):
        total = 0
        for i in range(len(chain) - 1):
            last_point = chain[i][-1]
            first_point = chain[i + 1][-2]
            total += self.euclidean_distance(last_point, first_point)
        return total

    def flip_element(self, new_combination):
        new_combination = list(new_combination)
        new_combination[2] = new_combination[2][::-1]
        new_combination[1] = new_combination[1][::-1]
        new_combination[5], new_combination[6] = new_combination[6], new_combination[5]
        new_combination[3], new_combination[4] = new_combination[4], new_combination[3]
        return tuple(new_combination)
    def generate_all_combinations(self, lst):
        all_combinations = []
        for perm in itertools.permutations(lst):
            n = len(perm)
            for flip_flags in itertools.product([False, True], repeat=n):
                new_combination = [ self.flip_element(element) if flip else element
                    for element, flip in zip(perm, flip_flags)
                ]
                all_combinations.append(tuple(new_combination))
        return all_combinations
    def find_minimum_distance_combination(self,chains):
        all_combinations = self.generate_all_combinations(chains)
        min_distance = 10e10
        for combination in all_combinations:
            distance = self.total_distance(combination)
            if distance < min_distance:
                min_distance = distance
                best_combination = combination
        return best_combination