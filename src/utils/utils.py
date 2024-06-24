import numpy as np
import random
import os
import json
import torch
import torch.backends
import shutil
import matplotlib.pyplot as plt
import pandas as pd


def draw_the_stat(log_file, CONFIG):
    """
    Plot and save graphs for training and validation loss, and MAE (Mean Absolute Error).

    Args:
        log_file (str): Path to the CSV file containing epoch, training loss, validation loss, training MAE,
        and validation MAE.
        CONFIG (dict): Configuration dictionary with keys including 'results_path' specifying where to save the plots.

    Returns:
        None
    """
    df = pd.read_csv(log_file)
    filename = os.path.join(CONFIG['results_path'], 'loss.jpg')
    plt.figure(figsize=(10, 5))
    plt.plot(df['Epoch'], df['Training Loss'], label='Training Loss')
    plt.plot(df['Epoch'], df['Validation Loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

    filename = os.path.join(CONFIG['results_path'], 'mae.jpg')
    plt.figure(figsize=(10, 5))
    plt.plot(df['Epoch'], df['Training MAE'], label='Training MAE')
    plt.plot(df['Epoch'], df['Validation MAE'], label='Validation MAE')
    plt.title('Training and Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


def init_experiment(CONFIG):
    """
    Initialize the experiment by setting random seeds, checking device availability, creating necessary directories,
    and copying the VAE model configuration file.

    Args:
        CONFIG (dict): A configuration dictionary that includes seeds, paths, and device information.

    Returns:
        dict: Updated CONFIG dictionary with modified or additional configurations.
    """
    random.seed(CONFIG['SEED'])
    np.random.seed(CONFIG['SEED'])
    torch.manual_seed(CONFIG['SEED'])

    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because the current PyTorch install was not "
                  "built with MPS enabled.")
        else:
            print("MPS not available because the current MacOS version is not 12.3+ "
                  "and/or you do not have an MPS-enabled device on this machine.")
    else:
        CONFIG['device'] = torch.device("mps")

    print("DEVICE NAME :", CONFIG['device'])
    CONFIG['image_path'] = os.path.join(CONFIG['exp_path'], "images")
    CONFIG['test_calcs'] = os.path.join(CONFIG['exp_path'], "test_calcs")
    CONFIG['model_path'] = os.path.join(CONFIG['exp_path'], "model")
    CONFIG['config_path'] = os.path.join(CONFIG['exp_path'], "config")
    CONFIG['results_path'] = os.path.join(CONFIG['exp_path'], "results")
    list_of_folders = [CONFIG['exp_path'],
                       CONFIG['image_path'],
                       CONFIG['test_calcs'],
                       CONFIG['model_path'],
                       CONFIG['config_path'],
                       CONFIG['results_path']]

    for folder in list_of_folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
    print(CONFIG)
    json_path = os.path.join(CONFIG['config_path'], "config.json")
    with open(json_path, 'w') as file:
        NEW_CONFIG = CONFIG
        NEW_CONFIG['device'] = str(NEW_CONFIG['device'])
        json.dump(CONFIG, file)
    source_path = 'src/vae.py'
    destination_path = os.path.join(CONFIG['config_path'], "vae.py")
    shutil.copyfile(source_path, destination_path)
    return CONFIG


def show_images(images, title=""):
    """
    Display a grid of images.

    Args:
        images (torch.Tensor or numpy.ndarray): Images to display.
        title (str): Title of the figure.

    Returns:
        None
    """
    if type(images) is torch.Tensor:
        images = images.detach().cpu().numpy()
    fig = plt.figure(figsize=(8, 8))
    rows = int(len(images) ** (1 / 2))
    cols = round(len(images) / rows)
    idx = 0
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, idx + 1)
            if idx < len(images):
                plt.imshow(images[idx][0], cmap="gray")
                idx += 1
    fig.suptitle(title, fontsize=30)
    plt.show()


def show_first_batch(loader):
    """
    Display images from the first batch of a data loader.

    Args:
        loader (torch.utils.data.DataLoader): The data loader to extract the first batch from.

    Returns:
        None
    """
    for batch in loader:
        show_images(batch[0], "Images in the first batch")
        break


def show_forward(ddpm, loader, device):
    """
    For each batch in the loader, show the original images and their corresponding images at different stages
    of the DDPM process.

    Args:
        ddpm (object): The denoising diffusion probabilistic model (DDPM) object.
        loader (torch.utils.data.DataLoader): Data loader providing batches of images.
        device (torch.device): Device to which the images should be transferred before processing.

    Returns:
        None
    """
    for batch in loader:
        imgs = batch[0]
        show_images(imgs, "Original images")
        for percent in [0.25, 0.5, 0.75, 1]:
            show_images(ddpm(imgs.to(device), [int(percent * ddpm.n_steps) - 1 for _ in range(len(imgs))]),
                f"DDPM Noisy images {int(percent * 100)}%"
            )
        break
