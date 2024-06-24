from preprocessing.preprocessing import RNADataset, train_val_test_split
from vae import VAE, Loss
from utils.visualization import ExperimentVisualizer
import warnings
import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import pickle
import json
import time
from preprocessing.dataset import InitialDataset, run_data
import random

warnings.filterwarnings("ignore")
warnings.warn("This is a warning message", UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

class InferencePipeline:
    def __init__(self, config , test_loader):
        self.config = config
        self.model = self.load_model()
        self.test_loader = test_loader

    def load_model(self):
        saved_model_path = os.path.join(self.config['model_path'], 'best_model.pth')
        if os.path.exists(saved_model_path):
            model = torch.load(saved_model_path)
            print("Model loaded successfully!")
            return model
        else:
            print("No saved model found at the specified path. Exiting...")
            exit()

    def run_inference(self):
        if self.model is None:
            print("Model not loaded. Please ensure the model is loaded before running inference.")
            return

        self.model.eval()
        test_mae_pure, test_mae_symmetric, n_images_sum, test_rmse_pure, test_rmse_symmetric = 0.0, 0.0, 0.0, 0.0, 0.0
        true_coords, pred_coords, seq_arr, five_pred_coords, pdb_arr = [], [], [], [], []
        time_arr = []
        with torch.no_grad():
            for batch_data in self.test_loader:
                time0 = time.time()
                x = batch_data['Image'].double()
                n_images = batch_data['Image'].shape[0]  # Assuming 'Image' is a tensor
                mask = batch_data['Mask'].double()
                mask = mask.view(mask.size(0), -1)
                x = x.view(x.size(0), -1)
                x_random = torch.randn_like(x)
                cond = batch_data['Sequence'].double().view(batch_data['Sequence'].size(0), -1)
                pdb = batch_data['Index']
                y_pred, mu, log_sigma = self.model(x_random, cond)
                maxi = batch_data['CoordinatesStats']['maxi']
                mini = batch_data['CoordinatesStats']['mini']
                scaling = (maxi - mini)[0].item()
                mae = Loss.mae_sum(x, y_pred, mask, scaling)
                test_mae_pure += mae.item()

                rmse = Loss.rmse_sum(x, y_pred, mask, scaling)
                test_rmse_pure += rmse.item()

                true_coords.append((x * scaling).cpu().numpy())
                pred_coords.append((y_pred * scaling).cpu().numpy())
                y_pred = self.model.simmetrize(y_pred)  # Assuming the model has a 'simmetrize' method
                mae = Loss.mae_sum(x, y_pred, mask, scaling)
                test_mae_symmetric += mae.item()

                rmse = Loss.rmse_sum(x, y_pred, mask, scaling)
                test_rmse_symmetric += rmse.item()

                pdb_arr.append(pdb)
                seq_arr.append(cond.cpu().numpy())
                n_images_sum += n_images
                time1= time.time()
                time_arr.append(time1-time0)
                batch_five_pred_coords = []

                for _ in range(5):
                    x_random = torch.randn_like(x)
                    y_pred, _, _ = self.model(x_random, cond)
                    listy = (y_pred * scaling).cpu().numpy()
                    batch_five_pred_coords.append(listy)

                batch_five_pred_coords = np.transpose(batch_five_pred_coords, (1, 0, 2))
                five_pred_coords.append(batch_five_pred_coords)

        print("The average time for calculation for one image ", np.array(time_arr).mean() )
        self.save_results(true_coords, pred_coords, seq_arr, five_pred_coords, test_mae_pure, test_mae_symmetric,
                          test_rmse_pure, test_rmse_symmetric, n_images_sum, pdb_arr)

    def calculate_errors(self, true_coords, five_pred_coords):
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

    def save_results(self, true_coords, pred_coords, seq_arr, five_pred_coords, test_mae_pure, test_mae_symmetric,
                     test_rmse_pure, test_rmse_symmetric,n_images_sum, pdb_arr):

        true_coords = np.concatenate(true_coords, axis=0)
        pred_coords = np.concatenate(pred_coords, axis=0)
        seq_arr = np.concatenate(seq_arr, axis=0)
        pdb_arr = np.concatenate(pdb_arr, axis=0)
        five_pred_coords = np.concatenate(five_pred_coords, axis=0)

        test_mae_pure /= n_images_sum
        test_mae_symmetric /= n_images_sum

        test_rmse_pure /= n_images_sum
        test_rmse_symmetric /= n_images_sum

        mean, std = self.calculate_errors(true_coords, five_pred_coords)
        print("Mean CV", mean)
        print("Std CV", std)
        print("TEST MAE PURE", test_mae_pure)
        print("TEST MAE SYMMETRIC", test_mae_symmetric)
        print("TEST RMSE PURE", test_rmse_pure)
        print("TEST RMSE SYMMETRIC", test_rmse_symmetric)



        filepath = os.path.join(self.config['test_calcs'], "test_results.txt")
        with open(filepath, 'w') as file:
            file.write(f"TEST MAE PURE {test_mae_pure}\n")
            file.write(f"TEST MAE SYMMETRIC {test_mae_symmetric}\n")
            file.write(f"TEST RMSE PURE {test_rmse_pure}\n")
            file.write(f"TEST RMSE SYMMETRIC {test_rmse_symmetric}\n")

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
        visualizer = ExperimentVisualizer(self.config['exp_path'])
        visualizer.visualize_structure()
        visualizer.visualize_five()

if __name__ == '__main__':
    with open('src/config/config_inference.json', 'r') as f:
        CONFIG = json.load(f)
    random.seed(CONFIG['SEED'])
    np.random.seed(CONFIG['SEED'])
    torch.manual_seed(CONFIG['SEED'])
    CONFIG['experiment_path'] = CONFIG['output_path']
    CONFIG['test_calcs'] = os.path.join(CONFIG['output_path'], "test_calcs")
    CONFIG['image_path'] = os.path.join(CONFIG['output_path'], "images")
    list_of_folders = [CONFIG['output_path'], CONFIG['test_calcs']]
    for folder in list_of_folders:
        if not os.path.exists(folder):
            os.makedirs(folder)

    raw_dataset = InitialDataset(CONFIG)
    raw_dataset = run_data(raw_dataset)
    (train_dataset, val_dataset, test_dataset) = train_val_test_split(raw_dataset,
                                                                      ratio=CONFIG['ratio_arr'],
                                                                      proportion=CONFIG['proportion'])

    data_train = RNADataset(train_dataset, CONFIG)
    data_val = RNADataset(val_dataset, CONFIG)
    data_test = RNADataset(test_dataset, CONFIG)

    train_loader = DataLoader(data_train, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(data_val, batch_size=CONFIG['batch_size'], shuffle=True)
    test_loader = DataLoader(data_test, batch_size=CONFIG['batch_size'], shuffle=True)

    pipeline = InferencePipeline(CONFIG, test_loader)
    pipeline.run_inference()