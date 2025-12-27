from preprocessing.preprocessing import RNADataset, preprocess_without_splitting
from vae import Loss
from utils.visualization import ExperimentVisualizer
from preprocessing.dataset import InitialDataset
from vae import VAE_Utils, VAE, Loss
from utils.spatial_visualization import spatial_visualizer
from utils.utils import clear_and_create_folder, calculate_gdt
import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import pickle
import json
import time
from preprocessing.dataset_solo import RNA_Dataset
import random
from tmtools import tm_align
import torch
from Bio import SeqIO
from process_result import distance_matrix_to_3d_multiple, convert_distance_matrix_to_pdb_multiple



torch.serialization.add_safe_globals([VAE])
import warnings
from Bio.PDB import PDBExceptions
# Suppress specific warning types
warnings.filterwarnings("ignore", category=PDBExceptions.PDBConstructionWarning)
from Bio import BiopythonDeprecationWarning
# Suppress specific Biopython deprecation warnings
warnings.simplefilter("ignore", BiopythonDeprecationWarning)
import warnings

# Ignore specific deprecation warning
warnings.filterwarnings("ignore", category=DeprecationWarning, module='Bio.pairwise2')

def one_hot_encode(seq, size):
    """One-hot encode RNA sequences.
    Args:
        seq: A list of sequences represented as strings.

    Returns:
        A list of one-hot encoded sequences.
    """
    chain_list = []
    for s in seq:
        if s == 'A':
            chain_list.append([1.0, 0.0, 0.0, 0.0])
        elif s == 'C':
            chain_list.append([0.0, 1.0, 0.0, 0.0])
        elif s == 'G':
            chain_list.append([0.0, 0.0, 1.0, 0.0])
        else:  # Assumes 'U'
                chain_list.append([0.0, 0.0, 0.0, 1.0])
    if (len(chain_list) < size):
        for _ in range(len(chain_list), size):
            chain_list.append([0.0,0.0,0.0,0.0])
    return torch.from_numpy(np.array([chain_list]))

class InferencePipeline:
    def __init__(self, config, viz, viz_3d, desc):
        self.config = config
        self.model = self.load_model()
        self.viz = viz
        self.viz_3d = viz_3d
        self.desc = desc

    def load_model(self):
        saved_model_path = os.path.join(self.config['model_path'], 'best_model.pth')
        print(f"Loaded model {saved_model_path}")
        if os.path.exists(saved_model_path):
            model = torch.load(saved_model_path, weights_only=False)
            print("Model loaded successfully!")
            return model
        else:
            print("No saved model found at the specified path. Exiting...")
            exit()

    def run_inference(self, CONFIG):
        random.seed(CONFIG['SEED'])
        np.random.seed(CONFIG['SEED'])
        torch.manual_seed(CONFIG['SEED'])
        fasta_file = CONFIG['fasta']
        size = CONFIG['size']
        if self.model is None:
            print("Model not loaded. Please ensure the model is loaded before running inference.")
            return
        self.model.eval()
        device = torch.device(CONFIG['device'] if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        with torch.no_grad():
            pred_coords = []
            five_pred_coords = []
            seq_arr = []
            pdb_arr = []  # Array to store pdb data
            for record in SeqIO.parse(fasta_file, "fasta"):
                x_random = torch.rand(1, size*size).to(device)
                print(one_hot_encode(record.seq, size).shape)
                cond = one_hot_encode(record.seq, size).double().view(1, -1).to(device)
                print("cond")
                print(cond)
                print(cond.shape)
                y_pred, mu, log_sigma = self.model(x_random, cond)
                print(y_pred.reshape(size, size))
                print(record.id)
                convert_distance_matrix_to_pdb_multiple(y_pred.reshape(size, size)*100, record.seq, output_filename=record.id+".pdb")


    def visualize_results(self):
        visualizer = ExperimentVisualizer(self.config['output_path'])
        if self.viz:
            visualizer.visualize_structure()
        if self.viz_3d:
            visualizer.visualize_five()


if __name__ == '__main__':
    with open('config/config_prediction.json', 'r') as f:
        config_test = json.load(f)
    config_base_path = config_test['config_train']
    with open(config_base_path, 'r')as f:
        config_train = json.load(f)
    CONFIG = {**config_train,  **config_test}
    CONFIG['experiment_path'] = CONFIG['output_path']
    CONFIG['test_calcs'] = os.path.join(CONFIG['output_path'], "test_calcs")
    pipeline = InferencePipeline(CONFIG, viz=True, viz_3d=True, desc="test")
    pipeline.run_inference(CONFIG)
