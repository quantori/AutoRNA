import multiprocessing
import os
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from tqdm import tqdm
from torch.utils.data import Dataset
from utils.utils_rna import *

class InitialDataset:
    AMINOACIDS = {'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
                  'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TYR', 'VAL', 'TRP'}

    IONS = ['MG', 'ZN', 'NA', 'CL', 'K', 'SO4', 'SR', 'GTP', 'GOL', 'CA',
            'MN', 'CD', 'NCO', 'ACT', 'EDO', 'PO4']

    RNA_NUCLEOBASE = {'A', 'U', 'G', 'C'}
    @staticmethod
    def run_data(data: Dataset):
        data.drop_duplications()
        print("dropping duplications", len(data.df))
        data.filter_h2o()
        print("final_size h20", len(data.df))
        data.filter_proteins()
        print("final_size proteins", len(data.df))
        data.filter_ions()
        print("final_size ions", len(data.df))
        data.drop_empty_sequences()
        print("final_size drop empty", len(data.df))
        data.filter_nonvalid_rna()
        print("final_size nonvalid", len(data.df))
        data.calculate_backbones()
        print("final_size backbones ", len(data.df))
        print(data.df)
        data.calculate_dist_matrix('backbone')
        print("final_size dist matrix", len(data.df))
        # Calculate the distance matrix
        data.calculate_dist_matrix('backbone')

        print("final_size dist matrix", len(data.df))
        """
        def merge_sequences(sequences):
            return sum(sequences, [])

        def merge_backbones(backbones):
            return sum(backbones, [])
        merged_df = data.df.groupby('index').agg({
            'seq': merge_sequences,
            'backbones': merge_backbones
        }).reset_index()
        merged_df['chains'] = data.df.groupby('index')['chain'].agg(lambda x: ','.join(x)).reset_index(drop=True)
        data.df = merged_df
        print("Length of the merged", len(data.df))
        """
        return data

    def __init__(self, conf):
        self.conf = conf
        self.max_data = conf['max_data']
        self.pdb_folder_path = conf['pdb_folder_path']
        self.min_length = conf['min_length']
        self.max_length = conf['max_length']
        self.parser = PDBParser()
        self.df = self.from_pdb()
        self.type = conf['type']
        print("LENGTH DF", len(self.df))

    def save_csv(self, path):
        joined_seq = []
        df = self.df.copy(deep=True)
        for seq in df['seq']:
            joined_seq.append(''.join(seq))
        df['seq'] = joined_seq
        df.to_csv(path, index=False)

    def save_fasta(self, path):
        pass

    def filter_nonvalid_rna(self):
        print('Filtering non-valid rnas')
        valid_seqs = []
        for seq in self.df['seq']:
            if set(seq).issubset(self.RNA_NUCLEOBASE) and self.min_length < len(seq) < self.max_length:
                valid_seqs.append(seq)
                # print("good", seq)
            else:
                # print("bad", seq)
                valid_seqs.append(np.nan)
        self.df['seq'] = valid_seqs
        self.df.dropna(subset=['seq'], inplace=True)

    def filter_proteins(self):
        print("Filtering proteins")
        filtered_seq = []
        for seq in tqdm(self.df['seq'], desc='Filtering proteins'):
            filtered = [res for res in seq if res not in self.AMINOACIDS]
            filtered_seq.append(filtered)
        self.df['seq'] = filtered_seq

    def filter_ions(self):
        filtered_seq = []
        for seq in tqdm(self.df['seq'], desc='Filtering ions'):
            filtered = seq
            for aa in self.IONS:
                filtered = list(filter(lambda x: x != aa, filtered))
            filtered_seq.append(filtered)
        self.df['seq'] = filtered_seq

    def filter_h2o(self):
        print('Filtering "H2O"-molecules')
        filtered_seq = []
        for seq in self.df['seq']:
            filtered_seq.append(list(filter(lambda x: x != 'HOH', seq)))
        self.df['seq'] = filtered_seq

    def filter_dna(self):
        pass

    def get_by_id(self, pdb_id):
        return self.df[self.df['index'] == pdb_id]

    def drop_duplications(self):
        print('Dropping duplications in sequences')
        joined_seq = []
        df = self.df.copy(deep=True)
        for seq in df['seq']:
            joined_seq.append('.'.join(seq))
        df['seq'] = joined_seq
        df.drop_duplicates(subset=['seq'], inplace=True)
        self.df = self.df.iloc[df.index]


    def drop_duplications(self):
        print("Dropping duplications in sequences")

        df = self.df.copy(deep=True)
        df["seq"] = df["seq"].apply(lambda s: ".".join(s))
        df = df.drop_duplicates(subset="seq").reset_index(drop=True)

        self.df = df                      # ← no iloc/loc needed


    def from_csv(self, path) -> pd.DataFrame:
        print(f'Creating dataset from saved {path}')
        df = pd.read_csv(path)
        splitted_seq = []
        for seq in df['seq']:
            seq = seq.split('.') if seq is not np.nan else []
            splitted_seq.append(seq)
        df['seq'] = splitted_seq
        return df

    def single_pdb(self, file_name):
        seqs, pdb_ids, chain_ids = [], [], []
        name = file_name.split('.')[0]
        structure = self.parser.get_structure(file=os.path.join(self.pdb_folder_path, file_name), id=name)
        for chain in structure.get_chains():
            rna_seq = [residue.get_resname() for residue in chain.get_residues() if
                       residue.get_resname() in self.RNA_NUCLEOBASE]
            if rna_seq:
                seqs.append(rna_seq)
                pdb_ids.append(name)
                chain_ids.append(chain.id)
        return seqs, pdb_ids, chain_ids

    def from_pdb(self) -> pd.DataFrame:
        seqs, pdb_ids, chain_ids = [], [], []
        names = os.listdir(self.pdb_folder_path)
        names = names[:self.max_data]
        all = map(self.single_pdb, names)
        for s, p, c in all:
            seqs += s
            pdb_ids += p
            chain_ids += c
        rna_df = {'index': pdb_ids, 'chain': chain_ids, 'seq': seqs}
        rna_df = pd.DataFrame.from_dict(rna_df)
        rna_df_sorted = rna_df.sort_index()
        rna_df_sorted = rna_df_sorted.drop_duplicates(subset=['index', 'chain'])
        #rna_df_filtered = self.remove_more_2_chains(rna_df_sorted)
        #rna_df_filtered.reset_index()
        #pd.set_option('display.max_rows', None)
        #print(rna_df_filtered.head(500))
        return rna_df_sorted

    def remove_more_2_chains(self,df):
        repeated_indices = df['index'].value_counts()
        indices_to_keep = repeated_indices[repeated_indices <= 2].index

        # Filtering the dataframe to keep only rows where the index is repeated 2 or fewer times
        filtered_df = df[df['index'].isin(indices_to_keep)]
        return filtered_df

    def calculate_angles(self):
        ids = self.df['index'].unique()
        if self.conf.jobs > 1:
            pool = multiprocessing.Pool(processes=16)
            all = pool.map(self.chain_angles, ids)
            pool.close()
        else:
            all = map(self.chain_angles, ids)

        d = dict(all)
        coords = []
        for r in self.df.iterrows():
            pdb_id = r[1]['index']
            chain = r[1]['chain']
            coords.append(d[pdb_id][chain])
        self.df['angles'] = coords
        self.df.dropna(subset=['angles'], inplace=True)

    def chain_angles(self, pdb_id):
        path = f'{os.path.join(self.pdb_folder_path, pdb_id)}.{self.type}'
        structure = self.parser.get_structure(file=path, id=pdb_id)
        rna_chains_names = set(self.df[self.df['index'] == pdb_id]['chain'])
        pdb_angles = {}
        for chain in structure.get_chains():
            if chain.id in rna_chains_names:
                pdb_angles[chain.id] = utils_rna.single_chain_angles(chain)
        return pdb_id, pdb_angles

    def shuffle(self):
        self.df = shuffle(self.df)

    def calculate_secondary(self):
        sss = []
        for seq in tqdm(self.df['seq']):
            (ss, mfe) = RNA.fold(''.join(seq))
            sss.append(ss)
        self.df['secondary'] = sss

    def center_of_mass(self, pdb_id):
        path = f'{os.path.join(self.pdb_folder_path, pdb_id)}.{self.type}'
        structure = self.parser.get_structure(file=path, id=pdb_id)
        rna_chains_names = set(self.df[self.df['index'] == pdb_id]['chain'])
        pdb_coms = {}
        for chain in structure.get_chains():
            if chain.id in rna_chains_names:
                pdb_coms[chain.id] = center_of_mass_chain(chain)
        return pdb_id, pdb_coms

    def calculate_center_of_mass(self):
        ids = self.df['index'].unique()
        if self.conf['jobs'] > 1:
            pool = multiprocessing.Pool(processes=16)
            all = pool.map(self.center_of_mass, ids)
            pool.close()
        else:
            all = map(self.center_of_mass, ids)
        d = dict(all)
        coords = []
        for r in self.df.iterrows():
            pdb_id = r[1]['index']
            chain = r[1]['chain']
            coords.append(d[pdb_id][chain])
        self.df['center_of_mass'] = coords
        self.df.dropna(subset=['center_of_mass'], inplace=True)

    def calculate_dist_matrix(self, type):
        dist_matrices_list = []
        if type == 'center_of_mass':
            for center_of_mass in tqdm(self.df['center_of_mass']):
                matrix = calculate_distance_matrix(center_of_mass)
                dist_matrices_list.append(matrix)
            self.df['distance_matrix_com'] = dist_matrices_list
        elif type == 'backbone':
            for backbone in tqdm(self.df['backbones']):
                matrix = calculate_distance_matrix(backbone)
                dist_matrices_list.append(matrix)
            self.df['distance_matrix_backbone'] = dist_matrices_list

    def backbone_coords(self, pdb_id):
        path = f'{os.path.join(self.pdb_folder_path, pdb_id)}.{self.type}'
        structure = self.parser.get_structure(file=path, id=pdb_id)
        rna_chains_names = set(self.df[self.df['index'] == pdb_id]['chain'])
        pdb_coms = {}
        for chain in structure.get_chains():
            if chain.id in rna_chains_names:
                pdb_coms[chain.id] = extract_backbone_chain(chain)

        return pdb_id, pdb_coms

    def calculate_backbones(self):
        ids = self.df['index'].unique()
        if self.conf['jobs'] > 1:
            pool = multiprocessing.Pool(processes=16)
            all = pool.map(self.center_of_mass, ids)
            pool.close()
        else:
            all = map(self.backbone_coords, ids)

        d = dict(all)
        coords = []
        for r in self.df.iterrows():
            pdb_id = r[1]['index']
            chain = r[1]['chain']
            coords.append(d[pdb_id][chain])
        self.df['backbones'] = coords
        self.df.dropna(subset=['backbones'], inplace=True)
