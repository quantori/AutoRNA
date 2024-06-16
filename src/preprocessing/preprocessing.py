import numpy as np
from utils.utils_rna import add_padding_list, add_padding_value, create_distance_matrix, rotate_molecule
import torch
from PIL import Image
from utils.visualization import ExperimentVisualizer

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


def train_val_test_split(dataset, ratio=[0.7, 0.9, 1.0], proportion=1.0):
    """Split the dataset into training, validation, and test sets based on provided ratios.

    Args:
        dataset: The dataset to be split.
        ratio: A list of three numbers indicating the proportion of train, validation, and test sets.
        proportion: A scaling factor for the dataset size.

    Returns:
        A tuple containing training, validation, and test datasets.
    """
    pos = dataset.df['backbones']
    length = len(pos)
    id_list = np.arange(length)
    np.random.shuffle(id_list)
    train_step = round(length * ratio[0] * proportion)
    val_step = round(length * ratio[1] * proportion)
    test_step = round(length * ratio[2] * proportion)

    train_idx, val_idx, test_idx = id_list[:train_step], id_list[train_step:val_step], id_list[val_step:test_step]

    print("LENGTH of training, validation, and test datasets: ", len(train_idx), len(val_idx), len(test_idx))
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
        print("MAX", np.array(distance_matrix).max())
        print("MIN", np.array(distance_matrix).min())

        # Create and store distance matrix images
        self.torch_images = torch.Tensor(self.create_distance_images(distance_matrix, mask_list, (0.0, 100.0), save_images))

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
                ExperimentVisualizer.visualize_image_grey(grayscale_image,self.config['image_path'] + "/" + str(i) + ".jpeg")
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
