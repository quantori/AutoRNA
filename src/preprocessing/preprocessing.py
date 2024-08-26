import numpy as np
from utils.utils_rna import add_padding_list, add_padding_value, create_distance_matrix, rotate_molecule
import torch
from PIL import Image
from utils.visualization import ExperimentVisualizer
import os
import pandas as pd
import shutil
def create_positions(seq_list):
    """Generate positions for non-zero sequences.

    Args:
        seq_list: A list of numpy arrays representing sequences.

    Returns:
        A list of positions for each sequence where non-zero values are replaced by their index and
        zeros remain unchanged.
    """
    full_pos_list = []
    for i in range(len(seq_list)):
        pos_seq = []
        for j in range(len(seq_list[i])):
            if seq_list[i][j].sum() != 0:
                pos_seq.append(j)
            else:
                pos_seq.append(0)
        full_pos_list.append(pos_seq)
    return full_pos_list


def manage_train_val_test_folder(main_folder):
    if os.path.exists(main_folder):
        shutil.rmtree(main_folder)
    os.makedirs(os.path.join(main_folder, 'train'))
    os.makedirs(os.path.join(main_folder, 'val'))
    os.makedirs(os.path.join(main_folder, 'test'))

def create_mask_encoder(seq_list):
    """Create a mask for the encoder.

    This function generates a mask where the sum across the third dimension of the sequence list is used to identify
    active positions.

    Args:
        seq_list: A list of sequences (3D numpy array).

    Returns:
        A 2D numpy array where 1 indicates an active position and 0 indicates a non-active position.
    """
    mask = np.array(seq_list).sum(axis=2)
    mask[mask == 1] = 10.0
    mask[mask == 0] = 1
    mask[mask == 10.0] = 0
    return mask


def average_deviation_coords(pred, true, mask):
    """Calculate the average deviation of predicted coordinates from the true coordinates.

    Args:
        pred: Predicted coordinates tensor.
        true: True coordinates tensor.
        mask: Mask tensor to filter active positions.

    Returns:
        The average deviation of coordinates scaled by a constant factor.
    """
    mask = mask.unsqueeze(1)
    mask = mask.repeat(1, 3, 1, 1)
    pred = pred * mask
    true = true * mask
    number_of_ones = torch.sum(mask)
    diff_full = torch.abs(pred - true)
    summy = torch.sum(diff_full)
    summy = summy / number_of_ones
    return summy * 255.0 / 2.0


def cluster_folder_split(folders, ratio=[0.7, 0.9, 1.0]):
    """
    Split folders containing clusters into training, validation, and test sets based on provided ratios.

    Args:
        folders: List of folder paths containing the clusters.
        ratio: A list of three numbers indicating the proportion of train, validation, and test sets.

    Returns:
        A tuple containing lists of folder paths for training, validation, and test sets.
    """
    assert len(ratio) == 3, "Ratio list must contain exactly three elements."
    assert sum(ratio) == 1.0, "The sum of the ratios must be 1.0."
    np.random.shuffle(folders)
    total_folders = len(folders)
    train_split = int(total_folders * ratio[0])
    val_split = int(total_folders * ratio[1])
    train_folders = folders[:train_split]
    val_folders = folders[train_split:val_split]
    test_folders = folders[val_split:]
    print("Number of folders in training, validation, and test sets: ", len(train_folders), len(val_folders), len(test_folders))
    return train_folders, val_folders, test_folders

def read_clusters_into_df(base_dir):
    data = []
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path) and folder.startswith('cluster_'):
            cluster_number = int(folder.replace('cluster_', ''))
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                if os.path.isfile(file_path):
                    data.append({'file_name': file_name.split(".")[0], 'cluster': cluster_number})
    df = pd.DataFrame(data)
    return df

def copy_to_partition(source_folder, target_folder, df, idx):
    for filename in df.loc[idx, 'file_name']:
        new_filename = f"{filename}.ent"
        src_path = os.path.join(source_folder, new_filename)
        dest_path = os.path.join(target_folder, new_filename)
        shutil.copy(src_path, dest_path)


def train_val_test_split(CONFIG,
                         dataset,
                         ratio=[0.7, 0.9, 1.0],
                         proportion=1.0,
                         cluster_dir='data/clusters',
                         write_partition_to_disc = True,
                         partition_path = 'data/train_val_test_split'):
    """Split the dataset into training, validation, and test sets based on provided ratios.

    Args:
        dataset: The dataset to be split.
        ratio: A list of three numbers indicating the proportion of train, validation, and test sets.
        proportion: A scaling factor for the dataset size.

    Returns:
        A tuple containing training, validation, and test datasets.
    """
    df_clusters = read_clusters_into_df(cluster_dir)
    dataset.df['file_name'] = dataset.df['index'].astype(str)
    df_clusters['file_name'] = df_clusters['file_name'].astype(str)

    dataset.df = dataset.df.merge(df_clusters, on='file_name', how='left')
    unique_clusters = dataset.df['cluster'].unique()
    np.random.shuffle(unique_clusters)

    # Calculate the number of clusters for each split
    num_clusters = len(unique_clusters)
    train_clusters_end = round(num_clusters * ratio[0])
    val_clusters_end = round(num_clusters * ratio[1])
    print(num_clusters)
    print(train_clusters_end)
    print(val_clusters_end)
    # Split clusters
    train_clusters = unique_clusters[:train_clusters_end]
    val_clusters = unique_clusters[train_clusters_end:val_clusters_end]
    test_clusters = unique_clusters[val_clusters_end:]
    print(len(train_clusters), len(val_clusters), len(test_clusters))

    # Assign indices based on clusters
    train_idx = dataset.df[dataset.df['cluster'].isin(train_clusters)].index
    val_idx = dataset.df[dataset.df['cluster'].isin(val_clusters)].index
    test_idx = dataset.df[dataset.df['cluster'].isin(test_clusters)].index

    print("LENGTH of training, validation, and test datasets: ", len(train_idx), len(val_idx), len(test_idx))

    if write_partition_to_disc:
        manage_train_val_test_folder(partition_path)
        copy_to_partition(CONFIG['pdb_folder_path'],os.path.join(partition_path, "train"), dataset.df, train_idx)
        copy_to_partition(CONFIG['pdb_folder_path'], os.path.join(partition_path, "val"), dataset.df, val_idx)
        copy_to_partition(CONFIG['pdb_folder_path'], os.path.join(partition_path, "test"), dataset.df, test_idx)

    dataset_train, dataset_val, dataset_test = {}, {}, {}
    dataset_train['coords'] = np.array(dataset.df['backbones'])[train_idx]
    dataset_train['seq'] = np.array(dataset.df['seq'])[train_idx]
    dataset_train['index'] = np.array(dataset.df['index'])[train_idx]
    dataset_val['coords'] = np.array(dataset.df['backbones'])[val_idx]
    dataset_val['seq'] = np.array(dataset.df['seq'])[val_idx]
    dataset_val['index'] = np.array(dataset.df['index'])[val_idx]
    dataset_test['coords'] = np.array(dataset.df['backbones'])[test_idx]
    dataset_test['seq'] = np.array(dataset.df['seq'])[test_idx]
    dataset_test['index'] = np.array(dataset.df['index'])[test_idx]
    return dataset_train, dataset_val, dataset_test


class RNADataset:
    """A dataset class for RNA sequences and structures.

    This class is responsible for preparing and handling a dataset consisting of RNA sequences,
    their corresponding structural data, and images representing distance matrices. It includes
    functionalities for encoding sequences, padding, and normalizing data.

    Attributes:
        transform: A callable transform to be applied on a sample.
        config: A dictionary containing configuration parameters.
        data: A dictionary containing the full dataset of sequences and coordinates.
        pad_size: An integer representing the padding size for sequences to a uniform length.
        pad_val: The value used for padding sequences.
        torch_seq: A torch.Tensor containing the one-hot encoded sequences.
        torch_pos: A torch.Tensor containing the positions of sequences.
        mask_list: A torch.Tensor containing masks for sequences.
        mask_encoder_list: A torch.Tensor containing encoder masks for sequences.
        torch_images: A torch.Tensor of distance matrices represented as images.
        coords_stats: A dictionary containing statistics of coordinates for normalization purposes.
    """

    def __init__(self, full_dataset, config, transform=None, save_images=True):
        """Initialize the dataset with the given parameters.

        Args:
            full_dataset: A dictionary containing 'seq' and 'coords' keys for sequences and coordinates.
            config: A dictionary with configuration options, including 'max_length' and 'image_path'.
            transform: An optional callable to be applied on each sample.
        """
        self.transform = transform
        self.config = config
        self.data = full_dataset
        self.pad_size = config['max_length']
        self.pad_val = 0.0

        # Process sequences: one-hot encode, pad, and create masks
        seq_list = self.data['seq']
        seq_list = self.one_hot_encode(seq_list)
        seq_list = add_padding_list(seq_list, value=[0, 0, 0, 0], max_length=config['max_length'])
        mask_encoder = create_mask_encoder(seq_list)

        # Initialize coordinate statistics
        self.coords_stats = {"maxi": 100, "mini": 0}

        # Process coordinates: create distance matrix and padding

        (distance_matrix, mask_list) = create_distance_matrix(self.data['coords'],
                                                              padding_value=0.0,
                                                              max_length=config['max_length'])

        count_greater_than_100 = np.sum(np.array(distance_matrix) > 100)
        total_elements = sum(np.size(np.array(row)) for row in distance_matrix)
        proportion = count_greater_than_100 / total_elements
        print("Proportion of elements greater than 100:", proportion)

        pos_list = create_positions(seq_list)
        pos_list = add_padding_value(pos_list, value=0, max_length=config['max_length'])

        # Convert processed lists to tensors
        self.torch_seq = torch.Tensor(seq_list)
        self.torch_pos = torch.Tensor(pos_list)
        self.mask_list = torch.Tensor(mask_list)
        self.mask_encoder_list = torch.Tensor(mask_encoder).type(torch.bool)
        self.pdb_indexes = self.data['index']
        # Log max and min values of the distance matrix
        print("Statistics, maximum value:", np.max(np.array(distance_matrix)))
        print("Statistics, minimum_value:", np.min(np.array(distance_matrix)))

        # Create and store distance matrix images
        self.torch_images = torch.Tensor(self.create_distance_images(distance_matrix,
                                                                     mask_list,
                                                                     (0.0, 100.0),
                                                                     save_images))

    def __len__(self):
        """Return the total number of items in the dataset."""
        return len(self.torch_seq)

    def __getitem__(self, idx):
        """Retrieve a sample and its corresponding data by index.

        Args:
            idx: An integer index corresponding to the sample.

        Returns:
            A dictionary containing the sequence, image, mask, and coordinate statistics for the sample.
        """
        seq = self.torch_seq[idx]
        image = self.torch_images[idx] / 255  # Normalize image
        mask = self.mask_list[idx]
        coords_stats = self.coords_stats
        pdb_indexes = self.pdb_indexes[idx]

        sample = {
            "Sequence": seq,
            "Image": image,
            "Mask": mask,
            "CoordinatesStats": coords_stats,
            "Index": pdb_indexes
        }
        return sample

    def create_distance_images(self, distance_matrix, mask_list, edges, save_images=True):
        """Create grayscale images from distance matrices.

        Args:
            distance_matrix: A list of 2D arrays representing distance matrices.
            mask_list: A list of masks corresponding to the distance matrices.
            edges: A tuple containing minimum and maximum values for normalization.
            save_images :Whether images are saved
        Returns:
            A list of 2D arrays representing the grayscale images.
        """

        mini, maxi = edges
        images = []
        for i, matrix in enumerate(distance_matrix):
            mask = mask_list[i]
            matrix = np.clip(matrix, mini, maxi)
            normalized_matrix = (matrix - mini) / (maxi - mini)
            normalized_matrix[mask == 0.0] = 0.0  # Apply mask
            grayscale_image = (normalized_matrix * 255).astype(np.uint8)
            if save_images:
                ExperimentVisualizer.visualize_image_grey(grayscale_image,
                                                          self.config['image_path'] + "/" + str(i) + ".jpeg")
            image = Image.fromarray(grayscale_image)
            images.append(grayscale_image)
        return images

    @staticmethod
    def one_hot_encode(seq):
        """One-hot encode RNA sequences.

        Args:
            seq: A list of sequences represented as strings.

        Returns:
            A list of one-hot encoded sequences.
        """
        encoded_seq = []
        for chain in seq:
            chain_list = []
            for s in chain:
                if s == 'A':
                    chain_list.append([1.0, 0.0, 0.0, 0.0])
                elif s == 'C':
                    chain_list.append([0.0, 1.0, 0.0, 0.0])
                elif s == 'G':
                    chain_list.append([0.0, 0.0, 1.0, 0.0])
                else:  # Assumes 'U'
                    chain_list.append([0.0, 0.0, 0.0, 1.0])
            encoded_seq.append(chain_list)
        return encoded_seq

    def padding(self, sequences):
        padded_seqs = []
        for seq in sequences:
            seq_wid, seq_len = seq.shape
            pad_seq = np.ones((seq_wid, self.pad_size)) * self.pad_val
            pad_seq[:, :seq_len] = seq
            padded_seqs.append(pad_seq)
        return padded_seqs
