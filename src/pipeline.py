from preprocessing.preprocessing import RNADataset, train_val_test_split
from utils.utils import init_experiment, draw_the_stat
from preprocessing.dataset import InitialDataset, run_data
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

warnings.filterwarnings("ignore")
warnings.warn("This is a warning message", UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

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
                     'Validation MAE',
                     'Validation Symmetric MAE',
                     'Training RMSE',
                     'Validation RMSE',
                     'Validation Symmetric RMSE'])

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

    n_x = CONFIG['max_length'] * CONFIG['max_length']
    n_z = CONFIG['n_z']
    n_c = CONFIG['max_length'] * 4
    model = VAE(n_x, n_z, n_c, CONFIG['dropout']).double()
    print("Total number of parameters ", VAE.count_parameters(model))
    optimizer = torch.optim.Adam(model.parameters())
    best_val_mae = 10e10
    best_train_mae = 10e10
    best_val_rmse = 10e10

    for epoch in range(0, CONFIG['n_epoch']):
        model.train()
        train_loader_pbar = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        train_loss = 0
        train_mae = 0
        train_rmse = 0
        n_images = 0
        for idx, batch_data in train_loader_pbar:
            n_images += batch_data['Image'].size()[0]
            x = batch_data['Image'].double()
            mask = batch_data['Mask'].double()
            x = x.view(x.size(0), -1)
            x_random = torch.randn_like(x)
            mask = mask.view(mask.size(0), -1)

            cond = batch_data['Sequence'].double().view(batch_data['Sequence'].size(0), -1)
            y_pred, mu, log_sigma = model(x_random, cond)
            loss = Loss.beta_vae_loss(x, y_pred, mu, log_sigma, CONFIG['beta'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            maxi = batch_data['CoordinatesStats']['maxi']
            mini = batch_data['CoordinatesStats']['mini']
            scaling = (maxi-mini)[0].item()
            mae = Loss.mae_sum(x, y_pred, mask, scaling)
            rmse = Loss.rmse_sum(x, y_pred, mask, scaling)
            train_mae += mae.item()
            train_rmse += rmse.item()
        train_loss /= n_images
        train_mae /= n_images
        train_rmse /= n_images
        model.eval()

        val_loss = 0
        val_mae = 0
        val_rmse = 0
        val_mae_symmetric = 0
        val_rmse_symmetric = 0
        n_images = 0
        with torch.no_grad():
            for idx, batch_data in enumerate(val_loader):
                n_images += batch_data['Image'].size()[0]
                x = batch_data['Image'].double()
                mask = batch_data['Mask'].double()
                mask = mask.view(mask.size(0), -1)
                x = x.view(x.size(0), -1)
                x_random = torch.randn_like(x)
                cond = batch_data['Sequence'].double().view(batch_data['Sequence'].size(0), -1)
                y_pred, mu, log_sigma = model(x_random, cond)
                loss = Loss.beta_vae_loss(x, y_pred, mu, log_sigma, CONFIG['beta'])
                val_loss += loss.item()

                maxi = batch_data['CoordinatesStats']['maxi']
                mini = batch_data['CoordinatesStats']['mini']

                scaling = (maxi-mini)[0].item()
                mae = Loss.mae_sum(x, y_pred, mask, scaling)
                val_mae += mae.item()

                rmse = Loss.rmse_sum(x, y_pred, mask, scaling)
                val_rmse += rmse.item()

                y_pred = model.simmetrize(y_pred)
                mae = Loss.mae_sum(x, y_pred, mask, scaling)
                val_mae_symmetric += mae.item()

                rmse = Loss.rmse_sum(x, y_pred, mask, scaling)
                val_rmse_symmetric += rmse.item()

        val_loss = val_loss / n_images
        val_mae = val_mae / n_images
        val_mae_symmetric = val_mae_symmetric / len(val_loader.dataset)
        val_rmse = val_rmse / n_images
        val_rmse_symmetric = val_rmse_symmetric / len(val_loader.dataset)

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
            torch.save(model, model_save_path)
            print(f"Best model saved with Validation MAE: {best_val_mae:.8f}")


        writer.writerow([epoch, train_loss, val_loss, train_mae,
                         train_rmse,val_mae, val_mae_symmetric, val_rmse, val_rmse_symmetric])
    file.close()

    draw_the_stat(log_file, CONFIG)

    n_images = CONFIG['n_images_show']
    with torch.no_grad():
        sample_z = torch.randn(n_images, CONFIG['n_z']).double()
        batch_data = next(iter(val_loader))
        true_images = batch_data['Image'][:, :, :].double()
        true_images = true_images.view(true_images.size(0), -1).numpy()
        condition_vectors = batch_data['Sequence'].double().view(batch_data['Sequence'].size(0), -1)
        if len(true_images) >= n_images:
            idx = random.sample(range(len(true_images)), n_images)
        else:
            idx = random.choices(range(len(true_images)), k=n_images)
        true_images = true_images[idx]
        condition_vectors = condition_vectors[idx]
        generated_images = model.decoder(sample_z, condition_vectors).numpy()
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
    pipeline = InferencePipeline(CONFIG, test_loader)
    pipeline.run_inference()
