import pickle
import numpy as np  # Added for array manipulation
import math
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import os


class ExperimentVisualizer:
    """
    A class used to visualize the results of experiments, specifically for comparing
    predicted and actual heatmaps based on experiment data.

    Attributes:
        experiment_path (str): The base path for the experiment data.
        folder_to_store (str): The path to the folder where images will be stored.
    """

    def __init__(self, experiment_name):
        """
        Initializes the ExperimentVisualizer with a given experiment name, creates a
        directory for storing images, and loads the experiment data.

        Args:
            experiment_name (str): The name of the experiment.
        """
        self.experiment_path = experiment_name
        self.folder_to_store = os.path.join(self.experiment_path, "images_comparison")
        os.makedirs(self.folder_to_store, exist_ok=True)  # Ensure the directory exists
        self.load_data()

    def load_data(self):
        """
        Loads the prediction, sequence, and true data from pickle files.
        """
        pred_path = os.path.join(self.experiment_path, 'test_calcs/pred.pickle')
        with open(pred_path, 'rb') as f:
            self.pred_arr = pickle.load(f)

        seq_path = os.path.join(self.experiment_path, 'test_calcs/sequences.pickle')
        with open(seq_path, 'rb') as f:
            self.sequences_arr = pickle.load(f)

        true_path = os.path.join(self.experiment_path, 'test_calcs/true.pickle')
        with open(true_path, 'rb') as f:
            self.true_arr = pickle.load(f)

        true_path = os.path.join(self.experiment_path, 'test_calcs/five.pickle')
        with open(true_path, 'rb') as f:
            self.five_arr = pickle.load(f)

    @staticmethod
    def visualize_image_grey(data, filename, y_index=0):
        """
        Plots an image of the given data with X and Y axis titled 'Nucleotides'
        and the number of non-zero values for a fixed row (specified by y_index) displayed in the title,
        then saves the plot to a file.

        Parameters:
        data (numpy array): 2D numpy array of data to be plotted.
        filename (str): The name of the file to save the plot.
        y_index (int): The index of the row for which to count non-zero elements.
        """
        non_zero_count = np.count_nonzero(data[y_index])

        plt.imshow(data, cmap='gray', interpolation='none')
        plt.colorbar()

        plt.xlabel("Nucleotides")
        plt.ylabel("Nucleotides")

        plt.title(f"Number of nucleotides : {non_zero_count}")
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close()

    @staticmethod
    def sequence_to_labels(seq):
        """
        Converts a sequence array into a list of labels based on a predefined mapping.

        Args:
            seq (array_like): The sequence array to convert.

        Returns:
            list: A list of character labels corresponding to the sequence.
        """
        label_map = {
            (1.0, 0.0, 0.0, 0.0): 'A',
            (0.0, 1.0, 0.0, 0.0): 'C',
            (0.0, 0.0, 1.0, 0.0): 'G',
            (0.0, 0.0, 0.0, 1.0): 'U',
        }
        labels = [label_map.get(tuple(s), '0') for s in seq]
        return labels

    @staticmethod
    def create_heatmap(data, ax, title="Heatmap", x_labels=None, y_labels=None, cbar=True,
                       title_fontsize=20, cbar_labelsize=14, axis_labelsize=14):
        """
        Creates a heatmap from the given data.

        Args:
            data (array_like): The data to display in the heatmap.
            ax (matplotlib.axes.Axes): The matplotlib axes object to draw the heatmap on.
            title (str, optional): The title of the heatmap. Defaults to "Heatmap".
            x_labels (list, optional): The labels for the x-axis. Defaults to None.
            y_labels (list, optional): The labels for the y-axis. Defaults to None.
            cbar (bool, optional): Whether to add a colorbar to the heatmap. Defaults to True.
            title_fontsize (int, optional): Font size of the title. Defaults to 20.
            cbar_labelsize (int, optional): Font size for color bar labels. Defaults to 14.
            axis_labelsize (int, optional): Font size for the x-axis and y-axis labels. Defaults to 14.
        """
        im = ax.imshow(data, cmap='YlOrRd', interpolation='nearest')
        ax.set_title(title, fontsize=title_fontsize)

        if cbar:
            colorbar = plt.colorbar(im, ax=ax)
            colorbar.ax.tick_params(labelsize=cbar_labelsize)  # Set the size of the labels on the color bar
            colorbar.set_label('Intensity',
                               size=cbar_labelsize)  # If you have a label for the colorbar, set its size here

        if x_labels is not None:
            ax.set_xticks(np.arange(len(x_labels)))
            ax.set_xticklabels(x_labels, rotation='vertical', fontsize=axis_labelsize)
        else:
            ax.set_xticks([])

        if y_labels is not None:
            ax.set_yticks(np.arange(len(y_labels)))
            ax.set_yticklabels(y_labels, fontsize=axis_labelsize)
        else:
            ax.set_yticks([])

    def visualize_structure(self):
        """
        Visualizes the structure by creating heatmaps for predictions and actual data,
        saves the plots as images.
        """

        print("Visualizing the 3d structures... This could take a while. ")
        for i in range(self.pred_arr.shape[0]):
            pred = self.pred_arr[i, :]
            true = self.true_arr[i, :]
            sequences = self.sequences_arr[i, :]
            img_size = int(math.sqrt(self.pred_arr.shape[1]))
            pred_reshaped = pred.reshape(img_size, img_size)
            true_reshaped = true.reshape(img_size, img_size)
            seq_reshaped = sequences.reshape(img_size, 4)
            x_labels = self.sequence_to_labels(seq_reshaped)
            non_zero_indices = [index for index, label in enumerate(x_labels) if label != '0']
            filtered_x_labels = [x_labels[i] for i in non_zero_indices]
            filtered_pred = pred_reshaped[non_zero_indices, :][:, non_zero_indices]
            filtered_true = true_reshaped[non_zero_indices, :][:, non_zero_indices]
            fig, axs = plt.subplots(1, 2, figsize=(20, 10))
            self.create_heatmap(filtered_pred,
                                axs[0],
                                title='Predicted Heatmap',
                                x_labels=filtered_x_labels,
                                y_labels=filtered_x_labels)
            self.create_heatmap(filtered_true,
                                axs[1],
                                title='True Heatmap',
                                x_labels=filtered_x_labels,
                                y_labels=filtered_x_labels)
            plt.tight_layout()
            fig.savefig(os.path.join(self.folder_to_store, f'heatmap_{i}.png'))
            plt.close(fig)

    def visualize_five(self):
        print("Visualizing the heatmaps (generating conformations with different noise)... This could take a while.")
        for i in range(self.true_arr.shape[0]):
            sequences = self.sequences_arr[i, :]
            img_size = int(math.sqrt(self.pred_arr.shape[1]))
            seq_reshaped = sequences.reshape(img_size, 4)
            x_labels = self.sequence_to_labels(seq_reshaped)
            non_zero_indices = [index for index, label in enumerate(x_labels) if label != '0']
            filtered_x_labels = [x_labels[i] for i in non_zero_indices]
            true = self.true_arr[i, :]
            five = self.five_arr[i, :, :]

            img_size = int(math.sqrt(self.true_arr.shape[1]))
            true_reshaped = true.reshape(img_size, img_size)

            non_zero_indices = [index for index in range(img_size) if true_reshaped[index].sum() != 0]
            filtered_true = true_reshaped[non_zero_indices, :][:, non_zero_indices]

            # Create a figure with a GridSpec that has no extra column
            fig = plt.figure(figsize=(50, 10))
            gs = gridspec.GridSpec(1, 6)  # Reduced the columns to 6

            # Set up only one color bar for all subplots, handled later
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Adjusted the colorbar position and size

            ax0 = fig.add_subplot(gs[0])
            self.create_heatmap(filtered_true,
                                ax0,
                                title='True',
                                x_labels=None,
                                y_labels=filtered_x_labels,
                                cbar=False,
                                title_fontsize=32,
                                axis_labelsize=24)
            ax0.set_aspect('equal')
            # Store the y-limits to apply the same to all plots
            y_limits = ax0.get_ylim()

            # Handle the rest of the subplots
            for j in range(5):
                axs = fig.add_subplot(gs[j + 1])  # Start from index 1 without a gap
                filtered_pred_single = five[j, :].reshape(img_size, img_size)
                filtered_pred_single = filtered_pred_single[non_zero_indices, :][:, non_zero_indices]
                if j == 2:
                    title = "Predicted versions"
                else:
                    title = None
                self.create_heatmap(filtered_pred_single,
                                    axs,
                                    title=title,  # No individual title for subplots
                                    x_labels=None,
                                    y_labels=None,
                                    cbar=False,
                                    title_fontsize=32,
                                    axis_labelsize=24)
                axs.set_aspect('equal')
                axs.set_ylim(y_limits)  # Set the same y-limits to match heights

            # Only draw color bar for the last subplot, shared across all
            cbar = plt.colorbar(ax0.get_images()[0], cax=cbar_ax)
            cbar.ax.tick_params(labelsize=20)  # Set the colorbar tick label size
            # plt.constrained_layout(rect=[0, 0, 0.9, 1])  # Adjust layout to make space for colorbar
            fig.savefig(os.path.join(self.folder_to_store, f'heatmap_generation_{i}.png'))
            plt.close(fig)


if __name__ == '__main__':
    experiment_name = "experiments/128_VAE_wider_encoder_nz_256_symmetric_black"
    visualizer = ExperimentVisualizer(experiment_name)
    visualizer.visualize_structure()
