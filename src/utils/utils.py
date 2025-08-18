import numpy as np
import random
import os
import json
import torch
import torch.backends
import shutil
import matplotlib.pyplot as plt
import pandas as pd
import math
import seaborn as sns
from matplotlib.gridspec import GridSpec
import tmscoring
import uuid


def collect_length_statistics(CONFIG, train_loader, val_loader, test_loader):
    folder_path = CONFIG['stat_path']
    loaders = [train_loader, val_loader, test_loader]
    names = ["Train", "Validation", "Test"]
    loader_arr = []  # Initialize the list to store arrays
    for loader, name in zip(loaders, names):
        length_arr = []
        for batch in loader:
            input_masks = batch['Mask']
            for mask in input_masks:
                summy = mask.sum()
                length = math.sqrt(summy)
                length_arr.append(length)
        loader_arr.append(np.array(length_arr))  # Convert to numpy array before appending
    # Concatenate all arrays into a single array

    combined_array = np.concatenate(loader_arr)

    # Create a GridSpec layout with 2 columns:
    # The first column will have 3 rows for the individual histograms
    # The second column will have a single large plot for the combined histogram.
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 2, width_ratios=[1, 2])  # 3 rows, 2 columns

    # Train dataset histogram
    ax0 = fig.add_subplot(gs[0, 0])
    sns.histplot(np.array(loader_arr[0]), bins=32, alpha=0.25, kde=False, label='Train',
                 ax=ax0, stat="density", color='steelblue', edgecolor='steelblue')
    ax0.set_xlim(0, 64)
    ax0.set_ylim(0, 0.15)
    ax0.legend(fontsize=12)
    ax0.set_title('The distribution of the density of the length for Train/Validation/Test')
    ax0.set_xlabel('Sequence Length', fontsize=12)
    ax0.set_ylabel('Density', fontsize=12)

    # Validation dataset histogram
    ax1 = fig.add_subplot(gs[1, 0])
    sns.histplot(np.array(loader_arr[1]), bins=32, alpha=0.25, kde=False, label='Validation',
                 ax=ax1, stat="density", color='red', edgecolor='red')
    ax1.set_xlim(0, 64)
    ax1.set_ylim(0, 0.15)
    ax1.legend(fontsize=12)
    ax1.set_xlabel('Sequence Length', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)

    # Test dataset histogram
    ax2 = fig.add_subplot(gs[2, 0])
    sns.histplot(np.array(loader_arr[2]), bins=32, alpha=0.25, kde=False, label='Test',
                 ax=ax2, stat="density", color='orange', edgecolor='orange')
    ax2.set_xlim(0, 64)
    ax2.set_ylim(0, 0.15)
    ax2.legend(fontsize=12)
    ax2.set_xlabel('Sequence Length', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)

    # Combined dataset histogram
    ax3 = fig.add_subplot(gs[:, 1])  # Span all rows in the second column
    sns.histplot(combined_array, bins=50, alpha=0.5, kde=False, ax=ax3, color='steelblue', edgecolor='steelblue')
    ax3.set_xlim(0, 64)
    ax3.set_title('Absolute Number of Sequences for Specific Length for the Combined Dataset', fontsize=12)
    ax3.set_xlabel('Sequence Length', fontsize=12)
    ax3.set_ylabel('Count', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, "density_distributions.png"))


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

    CONFIG['clusters_path'] = os.path.join(CONFIG['exp_path'], "clusters")
    CONFIG['split_path'] = os.path.join(CONFIG['exp_path'], "train_val_test_split")
    CONFIG['image_path'] = os.path.join(CONFIG['exp_path'], "images")
    CONFIG['test_calcs'] = os.path.join(CONFIG['exp_path'], "test_calcs")
    CONFIG['model_path'] = os.path.join(CONFIG['exp_path'], "model")
    CONFIG['config_path'] = os.path.join(CONFIG['exp_path'], "config")
    CONFIG['results_path'] = os.path.join(CONFIG['exp_path'], "results")
    CONFIG['stat_path'] = os.path.join(CONFIG['exp_path'], "stats")
    list_of_folders = [CONFIG['exp_path'],
                       CONFIG['image_path'],
                       CONFIG['test_calcs'],
                       CONFIG['model_path'],
                       CONFIG['config_path'],
                       CONFIG['results_path'],
                       CONFIG['stat_path']]

    for folder in list_of_folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
    json_path = os.path.join(CONFIG['config_path'], "config.json")
    with open(json_path, 'w') as file:
        NEW_CONFIG = CONFIG
        NEW_CONFIG['device'] = str(NEW_CONFIG['device'])
        json.dump(CONFIG, file)
    source_path = 'src/vae.py'
    #destination_path = os.path.join(CONFIG['config_path'], "vae.py")
    #shutil.copyfile(source_path, destination_path)
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
                        f"DDPM Noisy images {int(percent * 100)}%")
        break


def create_temp_pdb(folder_true, folder_pred, true_coords, pred_coords, seq_arr):
    random_string = str(uuid.uuid4())
    true_pdb_path = os.path.join(folder_true, random_string + '.pdb')
    write_pdb(true_coords, seq_arr, true_pdb_path)
    pred_pdb_path = os.path.join(folder_pred, random_string + '.pdb')
    write_pdb(pred_coords, seq_arr, pred_pdb_path)
    return true_pdb_path, pred_pdb_path


def write_pdb(coords, seq_arr, pdb_filename):
    with open(pdb_filename, 'w') as f:
        for i, (nucleotide, coord) in enumerate(zip(seq_arr, coords), start=1):
            if isinstance(coord, np.ndarray):
                coord = coord.tolist()
            x, y, z = map(float, coord)
            mapping = {(1, 0, 0, 0): "A",  (0, 1, 0, 0): "C",  (0, 0, 1, 0): "G", (0, 0, 0, 1): "U"}
            nucleotide = mapping[tuple(nucleotide)]
            f.write(
                f"ATOM  {i:5d}  {nucleotide:>3}  NUC     A   {i:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00        "
                f"   N\n")


def clear_and_create_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)


def calculate_gdt(reference_coords, predicted_coords, thresholds=[1, 2, 4, 8]):
    distances = np.linalg.norm(reference_coords - predicted_coords, axis=1)
    gdt_results = []
    for threshold in thresholds:
        within_threshold = np.sum(distances <= threshold)
        percentage_within_threshold = (within_threshold / len(reference_coords)) * 100
        gdt_results.append(percentage_within_threshold)
    gdt_score = np.mean(gdt_results)
    return gdt_score, gdt_results


"""
def calculate_average_tm_score(true_folder, pred_folder):
    true_files = set(f for f in os.listdir(true_folder) if f.endswith(".pdb"))
    pred_files = set(f for f in os.listdir(pred_folder) if f.endswith(".pdb"))
    matched_files = true_files.intersection(pred_files)
    total_tm_score = 0
    count = 0
    for filename in matched_files:
        true_pdb_path = os.path.join(true_folder, filename)
        pred_pdb_path = os.path.join(pred_folder, filename)
        tm_score = calculate_tm_score(true_pdb_path, pred_pdb_path)
        if tm_score is not None:
            total_tm_score += tm_score
            count += 1
    if count > 0:
        average_tm_score = total_tm_score / count
    else:
        average_tm_score = 0
    return average_tm_score
"""
