from preprocessing.preprocessing import RNADataset, train_val_test_split
from utils.utils import init_experiment, draw_the_stat, collect_length_statistics
from preprocessing.dataset_solo import RNA_Dataset
from vae import VAE_Utils, VAE, Loss
import warnings
import torch
from torch.utils.data import DataLoader
from inference import InferencePipeline
import numpy as np
from tqdm import tqdm
import os
import random
import csv
import json
from preprocessing.filter_homology import cluster_rna_structures
from Bio import BiopythonDeprecationWarning
warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")
warnings.warn("This is a warning message", UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)

if __name__ == '__main__':
    with open('src/config/config.json', 'r') as f:
        CONFIG = json.load(f)
    CONFIG = init_experiment(CONFIG)
    log_file = os.path.join(CONFIG['results_path'], CONFIG['logs'])
    file = open(log_file, mode='w', newline='')
    writer = csv.writer(file)
    writer.writerow(['Epoch',
                     'Training Loss',
                     'Validation Loss',
                     'Training MAE',
                     'Training MAE Symmetric',
                     'Training MAE Symmetric STD',
                     'Training RMSE',
                     'Training RMSE Std',
                     'Training RMSE Symmetric',
                     'Training RMSE Symmetric STD',
                     'Validation MAE',
                     'Validation MAE Std',
                     'Validation MAE Symmetric',
                     'Validation MAE Symmetric STD',
                     'Validation RMSE',
                     'Validation RMSE Std',
                     'Validation RMSE Symmetric',
                     'Validation RMSE Symmetric STD'])

    RNA_Dataset = RNA_Dataset(CONFIG)
    raw_dataset = RNA_Dataset.get()
    cluster_rna_structures(CONFIG['pdb_folder_path'], CONFIG['clusters_path'], CONFIG['results_path'])
    print("length of the dataset",len(raw_dataset))
    (train_dataset, val_dataset, test_dataset) = train_val_test_split(CONFIG,
                                                                      raw_dataset,
                                                                      ratio=CONFIG['ratio_arr'],
                                                                      proportion=CONFIG['proportion'],
                                                                      cluster_dir=CONFIG['clusters_path'],
                                                                      write_partition_to_disc=True,
                                                                      partition_path=CONFIG['split_path'])
    data_train = RNADataset(train_dataset, CONFIG)
    data_val = RNADataset(val_dataset, CONFIG)
    data_test = RNADataset(test_dataset, CONFIG)

    train_loader = DataLoader(data_train, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(data_val, batch_size=CONFIG['batch_size'], shuffle=True)
    test_loader = DataLoader(data_test, batch_size=CONFIG['batch_size'], shuffle=True)

    collect_length_statistics(CONFIG, train_loader, val_loader, test_loader)
    n_x = CONFIG['max_length'] * CONFIG['max_length']
    n_z = CONFIG['n_z']
    n_c = CONFIG['max_length'] * 4
    device = torch.device(CONFIG['device'] if torch.cuda.is_available() else 'cpu')
    model = VAE(n_x, n_z, n_c, CONFIG['dropout']).double().to(device)
    print(" Total number of parameters ", VAE.count_parameters(model))
    optimizer = torch.optim.Adam(model.parameters())
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.00001, momentum=0.8)
    best_val_mae = 10e10
    best_train_mae = 10e10
    best_val_rmse = 10e10
    for epoch in range(0, CONFIG['n_epoch']):
        model.train()
        train_loader_pbar = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        train_loss = 0
        train_mae_arr = []
        train_rmse_arr = []
        train_mae_symmetric_arr = []
        train_rmse_symmetric_arr = []
        n_images = 0
        for idx, batch_data in train_loader_pbar:
            n_images += batch_data['Image'].size()[0]
            x = batch_data['Image'].double().to(device)
            mask = batch_data['Mask'].double().to(device)
            original_shape = x.shape
            x = x.view(x.size(0), -1)
            x_random = torch.randn_like(x).to(device)
            mask = mask.view(mask.size(0), -1).to(device)

            cond = batch_data['Sequence'].double().view(batch_data['Sequence'].size(0), -1).to(device)
            y_pred, mu, log_sigma = model(x_random, cond)
            loss = Loss.beta_vae_loss_symmetric(x, y_pred, mu, log_sigma, CONFIG['beta'], original_shape)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            maxi = batch_data['CoordinatesStats']['maxi']
            mini = batch_data['CoordinatesStats']['mini']
            scaling = (maxi-mini)[0].item()
            mae = Loss.mae_arr(x, y_pred, mask, scaling)
            rmse = Loss.rmse_arr(x, y_pred, mask, scaling)
            train_mae_arr.append(mae)
            train_rmse_arr.append(rmse)

            y_pred = model.simmetrize(y_pred)
            mae = Loss.mae_arr(x, y_pred, mask, scaling)
            rmse = Loss.rmse_arr(x, y_pred, mask, scaling)
            train_mae_symmetric_arr.append(mae)
            train_rmse_symmetric_arr.append(rmse)

        train_loss /= n_images

        train_mae, train_mae_std = Loss.calculate_mean_std(train_mae_arr)
        train_rmse, train_rmse_std = Loss.calculate_mean_std(train_rmse_arr)
        train_mae_symmetric, train_mae_symmetric_std = Loss.calculate_mean_std(train_mae_symmetric_arr)
        train_rmse_symmetric, train_rmse_symmetric_std = Loss.calculate_mean_std(train_rmse_symmetric_arr)


        model.eval()
        val_loss = 0
        val_mae_arr = []
        val_rmse_arr = []
        val_mae_symmetric_arr = []
        val_rmse_symmetric_arr = []
        n_images = 0
        with torch.no_grad():
            for idx, batch_data in enumerate(val_loader):
                n_images += batch_data['Image'].size()[0]
                x = batch_data['Image'].double().to(device)
                mask = batch_data['Mask'].double()
                mask = mask.view(mask.size(0), -1).to(device)
                original_shape = x.shape
                x = x.view(x.size(0), -1)
                x_random = torch.randn_like(x).to(device)
                cond = batch_data['Sequence'].double().view(batch_data['Sequence'].size(0), -1).to(device)
                y_pred, mu, log_sigma = model(x_random, cond)
                loss = Loss.beta_vae_loss_symmetric(x, y_pred, mu, log_sigma, CONFIG['beta'], original_shape)
                val_loss += loss.item()

                maxi = batch_data['CoordinatesStats']['maxi']
                mini = batch_data['CoordinatesStats']['mini']

                scaling = (maxi-mini)[0].item()
                mae = Loss.mae_arr(x, y_pred, mask, scaling)
                val_mae_arr.append(mae)

                rmse = Loss.rmse_arr(x, y_pred, mask, scaling)
                val_rmse_arr.append(rmse)

                y_pred = model.simmetrize(y_pred)
                mae = Loss.mae_arr(x, y_pred, mask, scaling)
                val_mae_symmetric_arr.append(mae)

                rmse = Loss.rmse_arr(x, y_pred, mask, scaling)
                val_rmse_symmetric_arr.append(rmse)

        val_loss /= n_images
        val_mae, val_mae_std = Loss.calculate_mean_std(val_mae_arr)
        val_rmse, val_rmse_std = Loss.calculate_mean_std(val_rmse_arr)
        val_mae_symmetric, val_mae_symmetric_std = Loss.calculate_mean_std(val_mae_symmetric_arr)
        val_rmse_symmetric, val_rmse_symmetric_std = Loss.calculate_mean_std(val_rmse_symmetric_arr)

        print(f'Epoch {epoch}, '
              f'Training Loss: {train_loss:.8f}, '
              f'Validation Loss: {val_loss:.8f}',
              f'Training MAE: {train_mae:.8f}',
              f'Training RMSE: {train_rmse:.8f}',
              f'Validation MAE: {val_mae:.8f}',
              f'Validation MAE symmetric: {val_mae_symmetric:.8f}',
              f'Validation RMSE: {val_rmse:.8f}',
              f'Validation RMSE symmetric: {val_rmse_symmetric:.8f}')

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_val_mae_sim = val_mae_symmetric
            best_train_mae = train_mae
            model_save_path = os.path.join(CONFIG['model_path'], 'best_model.pth')
            model_save_path_state =  os.path.join(CONFIG['model_path'], 'best_model_state.pth')
            torch.save(model, model_save_path)
            torch.save(model.state_dict(), model_save_path_state)
            print(f"Best model saved with Validation MAE: {best_val_mae:.8f}")
        writer.writerow([epoch, train_loss, val_loss,
                         train_mae, train_mae_std,
                         train_mae_symmetric, train_mae_symmetric_std,
                         train_rmse, train_rmse_std,
                         train_rmse_symmetric, train_rmse_symmetric_std,
                         val_mae, val_mae_std,
                         val_mae_symmetric, val_mae_symmetric_std,
                         val_rmse, val_rmse_std,
                         val_rmse_symmetric, val_rmse_symmetric_std])
    file.close()

    draw_the_stat(log_file, CONFIG)

    n_images = CONFIG['n_images_show']
    with torch.no_grad():
        sample_z = torch.randn(n_images, CONFIG['n_z']).double().to(device)
        batch_data = next(iter(val_loader))
        true_images = batch_data['Image'][:, :, :].double()
        true_images = true_images.view(true_images.size(0), -1).numpy()
        condition_vectors = batch_data['Sequence'].double().view(batch_data['Sequence'].size(0), -1).to(device)
        if len(true_images) >= n_images:
            idx = random.sample(range(len(true_images)), n_images)
        else:
            idx = random.choices(range(len(true_images)), k=n_images)
        true_images = true_images[idx]
        condition_vectors = condition_vectors[idx]
        generated_images = model.decoder(sample_z, condition_vectors).to("cpu").numpy()
    #
    # Create examples of training images
    #
    path_to_examples = os.path.join(CONFIG['results_path'], CONFIG['file_generated'])
    VAE_Utils.save_images(pred_images=generated_images,
                          true_images=true_images,
                          title_texts=np.arange(0, n_images, 1),
                          image_size=CONFIG['max_length'],
                          filesave=path_to_examples)

    #
    # Test pipeline
    #
    CONFIG['output_path'] = CONFIG['exp_path']
    pipeline = InferencePipeline(CONFIG, test_loader, viz=True, viz_3d=True, desc="test")
    pipeline.run_inference(CONFIG)
