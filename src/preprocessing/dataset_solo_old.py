
"""import os
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, is_aa
from tqdm import tqdm

from Bio.SeqUtils import seq1

class RNA_Dataset:
    RNA_NUCLEOBASE = {'A', 'U', 'G', 'C'}
    RNA_THREE_LETTER_CODES = {'ADE': 'A', 'URA': 'U', 'GUA': 'G', 'CYT': 'C'}

    def __init__(self, conf):
        self.conf = conf
        self.type = conf['type']
        self.max_data = conf['max_data']
        self.pdb_folder_path = conf['pdb_folder_path']
        self.min_length = conf['min_length']
        self.max_length = conf['max_length']
        self.parser = PDBParser()
        self.df = self.process_all_pdb_files()
        pd.set_option('display.max_rows', None)
        print(self.df.head(2000))
        print(len(self.df))
        self.df = self.drop_empty_sequences()
        print("dropping empty sequences", len(self.df))
        self.df = self.filter_nonvalid_rna()
        print("filtering non valid (size) RNA", len(self.df))
        self.df  = self.drop_duplications()
        print("dropping duplications", len(self.df))



    def get(self):
        return self.df

    def drop_duplications(self):
        print('Dropping duplications in sequences ... ')
        df = self.df.copy(deep=True)
        df_no_duplicates = df.drop_duplicates(subset=['seq'])
        self.df = df_no_duplicates
        return self.df

    def calculate_center_of_mass(residue):


        Calculate the center of mass for a nucleotide (residue).

        atom_coords = np.array([atom.coord for atom in residue.get_atoms()])
        mass = np.ones(len(atom_coords))  # Placeholder: Assuming equal mass for simplicity
        center_of_mass = np.average(atom_coords, axis=0, weights=mass)
        return center_of_mass

    def drop_empty_sequences(self):
        print('Dropping empty sequences ... ')
        empty_seq = []
        for seq in self.df['seq']:
            s = np.nan if len(seq) == 0 else seq
            empty_seq.append(s)
        self.df['seq'] = empty_seq
        self.df.dropna(subset=['seq'], inplace=True)
        return self.df

    def filter_nonvalid_rna(self):
        print('Filtering non-valid RNAs')
        self.df['seq'] = self.df['seq'].apply(
            lambda seq: seq if set(seq).issubset(self.RNA_NUCLEOBASE) and self.min_length < len(
                seq) < self.max_length else np.nan
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

            if pdb_file.endswith(".ent") or pdb_file.endswith(".pdb"):  # Check if the file is a PDB file
                pdb_df = self.from_pdb(pdb_file_path)  # Call from_pdb() on each file
                if pdb_df is not None:
                    all_dfs.append(pdb_df)  # Append the DataFrame to the list
        # Concatenate all DataFrames into one
        if all_dfs:
            df = pd.concat(all_dfs, ignore_index=True)
        else:
            df = pd.DataFrame()  # Create an empty DataFrame if no files are found

        pd.set_option('display.max_rows', None)
        return df

    def check_chain_protein(chain):
        protein_letters = set("TGEKPYVCQECGKAFNCSSYLSKHQR")  # Set of standard protein letters
        for residue in chain.get_residues():
            residue_name = seq1(residue.get_resname())
            if residue_name in protein_letters:
                return True  # Return True if any protein letter is found
        return False  # Return False if no protein letters are found

    def from_pdb(self, pdb_file) -> pd.DataFrame:
        # Initialize lists to store concatenated data
        concatenated_seq = []  # Concatenated RNA sequence
        concatenated_coords = []  # Concatenated coordinates (centers of mass)
        pdb_ids = []  # List to store a single PDB ID (for each residue)
        unique_chains = set()  # Set to store unique chain IDs

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(pdb_file.split('.')[0], pdb_file)

        for chain in structure.get_chains():
            if RNA_Dataset.check_chain_protein(chain):
                return None

        structure = parser.get_structure(pdb_file.split('.')[0], pdb_file)
        for chain in structure.get_chains():
            rna_seq = []
            chain_coords = []

            for residue in chain.get_residues():
                residue_name = residue.get_resname()
                if residue_name in self.RNA_THREE_LETTER_CODES.keys():
                    residue_name = self.RNA_THREE_LETTER_CODES[residue_name]
                if residue_name in self.RNA_NUCLEOBASE:
                    com = RNA_Dataset.calculate_center_of_mass(residue)  # Calculate center of mass
                    com_list = com.tolist()  # Convert NumPy array to Python list
                    rna_seq.append(residue_name)  # Add the one-letter residue name to sequence
                    chain_coords.append(com_list)  # Append center of mass coordinates as a list

            # Concatenate all chain sequences and coordinates for the file
            concatenated_seq += rna_seq
            concatenated_coords += chain_coords
            pdb_ids += [pdb_file.split('.')[0]] * len(rna_seq)  # Repeat PDB ID for each residue

            # Add chain ID to unique chain list
            unique_chains.add(chain.id)

        # Ensure all lists have the same length
        if len(concatenated_seq) == len(concatenated_coords) == len(pdb_ids):
            # Prepare the final DataFrame for this file
            rna_df = {
                'index': [pdb_file.split('/')[-1].split(".")[0]],  # Only one PDB ID for the entire file
                'chains': [list(unique_chains)],  # A list of all unique chain identifiers
                'seq': [''.join(concatenated_seq)],  # Concatenated sequence as a single string
                'backbones': [concatenated_coords]  # Store concatenated coordinates as a list of lists
            }
            return pd.DataFrame.from_dict(rna_df)
        else:
            raise ValueError(
                f"Lengths mismatch: {len(concatenated_seq)} sequences, {len(concatenated_coords)} coordinates.")
"""