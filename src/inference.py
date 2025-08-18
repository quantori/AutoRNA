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

class InferencePipeline:
    def __init__(self, config, test_loader, viz, viz_3d, desc):
        self.config = config
        self.model = self.load_model()
        self.test_loader = test_loader
        self.viz = viz
        self.viz_3d = viz_3d
        self.desc = desc

    def load_model(self):
        saved_model_path = os.path.join(self.config['model_path'], 'best_model.pth')
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
        if self.model is None:
            print("Model not loaded. Please ensure the model is loaded before running inference.")
            return
        self.model.eval()
        device = torch.device(CONFIG['device'] if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        sum_32 = 0
        sum_64 = 0
        sum_32_64 = 0
        time_arr = []
        with torch.no_grad():
            test_loss = 0
            test_mae_arr = []
            test_rmse_arr = []
            test_mae_symmetric_arr = []
            test_rmse_symmetric_arr = []
            n_images = 0
            true_coords = []
            pred_coords = []
            five_pred_coords = []
            seq_arr = []
            pdb_arr = []  # Array to store pdb data
            test_rmse_dict = {'<32': [], '32-64': [], '>64': []}
            for batch_data in self.test_loader:
                print("Batch data keys:", batch_data)
                time0 = time.time()
                x = batch_data['Image'].double().to(device)
                n_images += x.size(0)
                mask = batch_data['Mask'].double().to(device)
                mask = mask.view(mask.size(0), -1)
                x = x.view(x.size(0), -1)
                x_random = torch.randn_like(x).to(device)
                cond = batch_data['Sequence'].double().view(batch_data['Sequence'].size(0), -1).to(device)
                pdb = batch_data['Index']
                y_pred, mu, log_sigma = self.model(x_random, cond)
                loss = Loss.beta_vae_loss(x, y_pred, mu, log_sigma, self.config['beta'])
                test_loss += loss.item()
                maxi = batch_data['CoordinatesStats']['maxi']
                mini = batch_data['CoordinatesStats']['mini']
                scaling = (maxi - mini)[0].item()
                mae = Loss.mae_arr(x, y_pred, mask, scaling)
                test_mae_arr.append(mae)
                rmse = Loss.rmse_arr(x, y_pred, mask, scaling)
                for i in range(len(rmse)):
                    mask_length = batch_data['Mask'].double()[i]
                    length = np.sqrt(mask_length.sum())
                    if length < 32:
                        sum_32 += 1
                        test_rmse_dict['<32'].append(rmse[i])
                    elif 32 <= length <= 64:
                        sum_32_64 += 1
                        test_rmse_dict['32-64'].append(rmse[i])
                    else:
                        sum_64 += 1
                        test_rmse_dict['>64'].append(rmse[i])
                test_rmse_arr.append(rmse)
                true_coords.append((x * scaling).cpu().numpy())
                pred_coords.append((y_pred * scaling).cpu().numpy())
                seq_arr.append(cond.cpu().numpy())  # Collecting sequence data
                pdb_arr.append(pdb)  # Collecting pdb data
                y_pred = self.model.simmetrize(y_pred)
                mae = Loss.mae_arr(x, y_pred, mask, scaling)
                test_mae_symmetric_arr.append(mae)

                rmse = Loss.rmse_arr(x, y_pred, mask, scaling)
                test_rmse_symmetric_arr.append(rmse)

                # Collect five predictions
                batch_five_pred_coords = []
                for _ in range(5):
                    x_random = torch.randn_like(x)
                    y_pred, _, _ = self.model(x_random, cond)
                    listy = (y_pred * scaling).cpu().numpy()
                    batch_five_pred_coords.append(listy)

                batch_five_pred_coords = np.transpose(batch_five_pred_coords, (1, 0, 2))
                five_pred_coords.append(batch_five_pred_coords)
                time1 = time.time()
                time_arr.append((time1-time0)/n_images)

            test_loss /= n_images
            test_mae, test_mae_std = Loss.calculate_mean_std(test_mae_arr)
            test_rmse, test_rmse_std = Loss.calculate_mean_std(test_rmse_arr)
            test_mae_symmetric, test_mae_symmetric_std = Loss.calculate_mean_std(test_mae_symmetric_arr)
            test_rmse_symmetric, test_rmse_symmetric_std = Loss.calculate_mean_std(test_rmse_symmetric_arr)
            test_rmse_less32, test_rmse_std_less_32 = Loss.calculate_mean_std(test_rmse_dict['<32'])
            test_rmse_32_64, test_rmse_std_32_64 = Loss.calculate_mean_std(test_rmse_dict['32-64'])
            test_rmse_more64, test_rmse_std_more64 = Loss.calculate_mean_std(test_rmse_dict['>64'])

            print(f'Loss: {test_loss:.8f}, \n'
                  f'MAE: {test_mae:.8f}, \n'
                  f'MAE std: {test_mae_std:.8f}, \n'
                  f'RMSE: {test_rmse:.8f}, \n'
                  f'RMSE std: {test_rmse_std:.8f}, \n'
                  f'MAE symmetric: {test_mae_symmetric:.8f}, \n'
                  f'MAE symmetric std: {test_mae_symmetric_std:.8f}, \n'
                  f'RMSE symmetric: {test_rmse_symmetric:.8f}, \n'
                  f'RMSE symmetric std: {test_rmse_symmetric_std:.8f} \n'
                  f'RMSE <32: {test_rmse_less32:.8f}, \n',
                  f'RMSE <32 std: {test_rmse_std_less_32:.8f}, \n'
                  f'RMSE <64 >32: {test_rmse_32_64:.8f}, \n'
                  f'RMSE <64 >32 std: {test_rmse_std_32_64:.8f}, \n'
                  f'RMSE >64: {test_rmse_more64:.8f}, \n'
                  f'RMSE >64 std: {test_rmse_std_more64:.8f}, \n'
                  f'number less 32: {sum_32:.8f}, \n'
                  f'number more 32: {sum_64:.8f}, \n'
                  f'number more 32 and less 64: {sum_32_64:.8f}, \n')

        filepath = os.path.join(self.config['test_calcs'], "test_results.txt")
        with open(filepath, 'w') as file:
            file.write(f"MAE {test_mae}\n")
            file.write(f"MAE STD {test_mae_std}\n")
            file.write(f"MAE SYMMETRIC {test_mae_symmetric}\n")
            file.write(f"MAE SYMMETRIC STD {test_mae_symmetric_std}\n")
            file.write(f"RMSE {test_rmse}\n")
            file.write(f"RMSE STD {test_rmse_std}\n")
            file.write(f"RMSE SYMMETRIC {test_rmse_symmetric}\n")
            file.write(f"RMSE SYMMETRIC STD {test_rmse_symmetric_std}\n")
        print("The average time for calculation for one image (RNA sequence) :  ", np.mean(np.array(time_arr)))
        self.save_results(true_coords, pred_coords, seq_arr, five_pred_coords, pdb_arr)
        self.calculate_additional_metrics(true_coords, pred_coords, seq_arr)

    def calculate_additional_metrics(self, true_coords, pred_coords, seq_arr):
        add_metric_folder = os.path.join(self.config['exp_path'], 'additional_metrics_'+self.desc)
        clear_and_create_folder(add_metric_folder)
        gdt_score_arr = []
        tm_score_arr = []
        true_coords = np.concatenate(true_coords, axis=0)
        pred_coords = np.concatenate(pred_coords, axis=0)
        seq_arr = np.concatenate(seq_arr, axis=0)
        gdt_score_dict = {"<32": [], "32-64": [], ">64": []}
        for i in range(len(np.array(true_coords)[:, 0])):
            true_coords_item = true_coords[i]
            pred_coords_item = pred_coords[i]
            seq_arr_item = seq_arr[i]
            seq = seq_arr_item.reshape(self.config['max_length'], 4)
            array = pred_coords_item.reshape(self.config['max_length'], self.config['max_length'])
            array_true = true_coords_item.reshape(self.config['max_length'], self.config['max_length'])
            try:
                points_3d, points_3d_true = spatial_visualizer.get_coordinates(seq, array, array_true)
                gdt_score, _ = calculate_gdt(points_3d_true, points_3d)
                mapping = {(1, 0, 0, 0): "A", (0, 1, 0, 0): "C", (0, 0, 1, 0): "G", (0, 0, 0, 1): "U"}
                nuc_arr = ""
                for seq_item in seq:
                    if np.sum(np.array(seq_item)) > 0:
                        nuc_arr += mapping[tuple(seq_item)]
                if len(nuc_arr) <= 32:
                    gdt_score_dict['<32'].append(gdt_score)
                if len(nuc_arr) > 32 and len(nuc_arr) <= 64:
                    gdt_score_dict['32-64'].append(gdt_score)
                if len(nuc_arr) >= 64:
                    gdt_score_dict['>64'].append(gdt_score)
                res = tm_align(points_3d_true, points_3d, nuc_arr, nuc_arr)
                tm_score = res.tm_norm_chain1
                gdt_score_arr.append(gdt_score)
                tm_score_arr.append(tm_score)
            except:
                print("Unable to perform SVD for this particular sequence")
        gdt_score_arr_np = np.array(gdt_score_arr)
        tm_score_arr_np = np.array(tm_score_arr)
        mean_gdt_score = np.mean(gdt_score_arr_np)
        std_gdt_score = np.std(gdt_score_arr_np)
        mean_tm_score = np.mean(tm_score_arr_np)
        std_tm_score = np.std(tm_score_arr_np)
        mean_gdt_score32 = np.mean(gdt_score_dict['<32'])
        mean_gdt_score32_64 = np.mean(gdt_score_dict['32-64'])
        mean_gdt_score64 = np.mean(gdt_score_dict['>64'])
        std_gdt_score32 = np.std(gdt_score_dict['<32'])
        std_gdt_score32_64 = np.std(gdt_score_dict['32-64'])
        std_gdt_score64 = np.std(gdt_score_dict['>64'])

        print("********* Additional metrics *********")
        print("MEAN GDT, STD GDT", mean_gdt_score, std_gdt_score)
        print("MEAN TM_score, STD TM_score", mean_tm_score, std_tm_score)
        print("MEAN,STD GDT<32", mean_gdt_score32, std_gdt_score32)
        print("MEAN,STD GDT32-64", mean_gdt_score32_64, std_gdt_score32_64)
        print("MEAN,STD GDT>64", mean_gdt_score64, std_gdt_score64)
        with open(os.path.join(add_metric_folder, "metrics.txt"), 'w') as file:
            print("********* Additional metrics *********", file=file)
            print("MEAN GDT, STD GDT:", mean_gdt_score, std_gdt_score, file=file)
            print("MEAN TM_score, STD TM_score:", mean_tm_score, std_tm_score, file=file)
            print("MEAN,STD GDT<32", mean_gdt_score32, std_gdt_score32)
            print("MEAN,STD GDT32-64", mean_gdt_score32_64, std_gdt_score32_64)
            print("MEAN,STD GDT>64", mean_gdt_score64, std_gdt_score64)

        return True

    @staticmethod
    def calculate_errors(true_coords, five_pred_coords):
        avg_cv_ratios = []

        for i in range(true_coords.shape[0]):
            errors = []
            for j in range(five_pred_coords.shape[1]):
                error = np.mean(np.abs(true_coords[i] - five_pred_coords[i, j]))
                errors.append(error)

            errors = np.array(errors)
            mean_error = np.mean(errors)
            std_error = np.std(errors)
            cv_ratio = std_error / mean_error
            avg_cv_ratios.append(cv_ratio)

        avg_cv_ratios = np.array(avg_cv_ratios)
        mean_of_avg_cv_ratios = np.mean(avg_cv_ratios)
        std_of_avg_cv_ratios = np.std(avg_cv_ratios)

        return mean_of_avg_cv_ratios, std_of_avg_cv_ratios

    def save_results(self,
                     true_coords,
                     pred_coords,
                     seq_arr,
                     five_pred_coords,
                     pdb_arr):

        true_coords = np.concatenate(true_coords, axis=0)
        pred_coords = np.concatenate(pred_coords, axis=0)
        seq_arr = np.concatenate(seq_arr, axis=0)
        pdb_arr = np.concatenate(pdb_arr, axis=0)
        five_pred_coords = np.concatenate(five_pred_coords, axis=0)
        #  mean, std = InferencePipeline.calculate_errors(true_coords, five_pred_coords)

        with open(os.path.join(self.config['test_calcs'], "true.pickle"), 'wb') as f:
            pickle.dump(true_coords, f)
        with open(os.path.join(self.config['test_calcs'], "pred.pickle"), 'wb') as f:
            pickle.dump(pred_coords, f)
        with open(os.path.join(self.config['test_calcs'], "sequences.pickle"), 'wb') as f:
            pickle.dump(seq_arr, f)
        with open(os.path.join(self.config['test_calcs'], "five.pickle"), 'wb') as f:
            pickle.dump(five_pred_coords, f)
        with open(os.path.join(self.config['test_calcs'], "pdb.pickle"), 'wb') as f:
            pickle.dump(pdb_arr, f)
        self.visualize_results()

    def visualize_results(self):
        visualizer = ExperimentVisualizer(self.config['output_path'])
        if self.viz:
            visualizer.visualize_structure()
        if self.viz_3d:
            visualizer.visualize_five()


if __name__ == '__main__':
    with open('src/config/config_inference.json', 'r') as f:
        config_test = json.load(f)
    config_base_path = config_test['config_train']
    with open(config_base_path, 'r')as f:
        config_train = json.load(f)
    CONFIG = {**config_train,  **config_test}
    CONFIG['experiment_path'] = CONFIG['output_path']
    CONFIG['test_calcs'] = os.path.join(CONFIG['output_path'], "test_calcs")
    CONFIG['image_path'] = os.path.join(CONFIG['output_path'], "images")
    list_of_folders = [CONFIG['output_path'], CONFIG['test_calcs'], CONFIG['image_path']]
    for folder in list_of_folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
    raw_dataset = RNA_Dataset(CONFIG)
    test_data_dict = preprocess_without_splitting(CONFIG, raw_dataset)
    test_dataset = RNADataset(test_data_dict, CONFIG, save_images=False)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    print("len(test_loader)", len(test_loader))
    print(test_loader)

    pipeline = InferencePipeline(CONFIG, test_loader, viz=True, viz_3d=True, desc="test")
    pipeline.run_inference(CONFIG)
