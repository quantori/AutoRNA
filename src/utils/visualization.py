import pickle
import numpy as np  # Added for array manipulation
import math
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Circle

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

        print("Visualizing the structure")
        for i in range(self.pred_arr.shape[0]):
            sequences = self.sequences_arr[i, :]
            img_size = int(math.sqrt(self.pred_arr.shape[1]))
            seq_reshaped = sequences.reshape(img_size, 4)
            x_labels = self.sequence_to_labels(seq_reshaped)

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
        print("Visualizing the structure (generating conformations with different noise). This could take a while.")
        for i in range(self.true_arr.shape[0]):
            sequences = self.sequences_arr[i, :]
            img_size = int(math.sqrt(self.pred_arr.shape[1]))
            seq_reshaped = sequences.reshape(img_size, 4)
            x_labels = self.sequence_to_labels(seq_reshaped)
            non_zero_indices = [index for index, label in enumerate(x_labels) if label != '0']
            filtered_x_labels = [x_labels[i] for i in non_zero_indices]
            true = self.true_arr[i, :]
            five = self.five_arr[i, :, :]

            sequences = self.sequences_arr[i, :]
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
                if j==2:
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
            plt.tight_layout(rect=[0, 0, 0.9, 1])  # Adjust layout to make space for colorbar
            fig.savefig(os.path.join(self.folder_to_store, f'heatmap_generation_{i}.png'))
            plt.close(fig)

    def draw_point_cloud(points_3d, points_3d_true, features, output_filename, x_lim=None, y_lim=None, z_lim=None):
        plt.style.use('bmh')
        x = points_3d[:, 0]
        y = points_3d[:, 1]
        z = points_3d[:, 2]
        x1 = points_3d_true[:, 0]
        y1 = points_3d_true[:, 1]
        z1 = points_3d_true[:, 2]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        if x_lim:
            ax.set_xlim(x_lim)
        if y_lim:
            ax.set_ylim(y_lim)
        if z_lim:
            ax.set_zlim(z_lim)

        colors = ['black' for _ in range(len(x))]
        for i in range(len(x)):
            if features[i][0] > 0.0:  # A
                colors[i] = 'red'
            if features[i][1] > 0.0:  # C
                colors[i] = 'green'
            if features[i][2] > 0.0:  # G
                colors[i] = 'steelblue'
            if features[i][3] > 0.0:  # U
                colors[i] = 'orange'

        ax.scatter(x, y, z, c=colors, s=40, edgecolors='white')
        ax.scatter(x1, y1, z1, c=colors, s=40, edgecolors='white')
        ax.set_facecolor('white')
        ax.set_xlabel('X', fontsize=8)
        ax.set_ylabel('Y', fontsize=8)
        ax.set_zlabel('Z', fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=8)

        for i in range(len(x) - 1):
            ax.plot([x[i], x[i + 1]], [y[i], y[i + 1]], [z[i], z[i + 1]], color='grey', linestyle='-', alpha=0.3)
            ax.plot([x1[i], x1[i + 1]], [y1[i], y1[i + 1]], [z1[i], z1[i + 1]], color='darkred', linestyle='-',
                    alpha=0.3)

        ax.grid(True)

        legend_elements = [
            Circle((0, 0), 0.1, facecolor='red', edgecolor='red', linewidth=0, label='A'),
            Circle((0, 0), 0.1, facecolor='green', edgecolor='green', linewidth=0, label='C'),
            Circle((0, 0), 0.1, facecolor='steelblue', edgecolor='steelblue', linewidth=0, label='G'),
            Circle((0, 0), 0.1, facecolor='orange', edgecolor='orange', linewidth=0, label='U')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10, frameon=False)

        plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
        plt.savefig(output_filename, dpi=400)

    def compute_centroid(points):
        return np.mean(points, axis=0)

    def align_points(set1, set2):
        set1, set2 = np.asarray(set1), np.asarray(set2)
        centroid1 = compute_centroid(set1)
        centroid2 = compute_centroid(set2)
        set1_centered = set1 - centroid1
        set2_centered = set2 - centroid2
        H = set1_centered.T @ set2_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        t = centroid2 - R @ centroid1
        aligned_set1 = (R @ set1.T).T + t
        return aligned_set1

    def get_global_limits(points_3d_list):
        all_points = np.concatenate(points_3d_list, axis=0)
        x_lim = (all_points[:, 0].min(), all_points[:, 0].max())
        y_lim = (all_points[:, 1].min(), all_points[:, 1].max())
        z_lim = (all_points[:, 2].min(), all_points[:, 2].max())
        return x_lim, y_lim, z_lim

    def distance_geometry(non_zero_submatrix):
        N = non_zero_submatrix.shape[0]
        J = np.eye(N) - np.ones((N, N)) / N
        B = -0.5 * J @ (non_zero_submatrix ** 2) @ J
        eigvals, eigvecs = np.linalg.eigh(B)
        idx = eigvals.argsort()[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        eigvals_3d = eigvals[:3]
        eigvecs_3d = eigvecs[:, :3]
        eigvals_3d_sqrt = np.sqrt(eigvals_3d).reshape(-1, 1)
        points_3d = eigvecs_3d @ np.diag(eigvals_3d_sqrt.flatten())
        return points_3d

    def find_min(a):
        rows, cols = np.nonzero(a)
        min_row, max_row = rows.min(), rows.max()
        min_col, max_col = cols.min(), cols.max()
        non_zero_submatrix = a[min_row:max_row + 1, min_col:max_col + 1]
        return non_zero_submatrix

    def reflect_across_plane(array, array2, plane='xy'):
        planes = ["nn", 'xy', 'xz', 'yz', 'xxy', 'xzz', 'yzz', 'all']
        sum = 10e10
        for plane in planes:
            print(plane)
            if array.ndim != 2 or array.shape[1] != 3:
                raise ValueError("Array must be a Nx3 array where N is the number of points and each row is [x, y, z].")
            reflected_array = np.copy(array)
            if plane == 'nn':
                pass
            elif plane == 'xy':
                reflected_array[:, 2] = -reflected_array[:, 2]
            elif plane == 'xz':
                reflected_array[:, 1] = -reflected_array[:, 1]
            elif plane == 'yz':
                reflected_array[:, 0] = -reflected_array[:, 0]
            elif plane == 'xxy':
                reflected_array[:, 2] = -reflected_array[:, 2]
                reflected_array[:, 1] = -reflected_array[:, 1]
            elif plane == 'xzz':
                reflected_array[:, 1] = -reflected_array[:, 1]
                reflected_array[:, 0] = -reflected_array[:, 0]
            elif plane == 'yzz':
                reflected_array[:, 2] = -reflected_array[:, 2]
                reflected_array[:, 0] = -reflected_array[:, 0]
            elif plane == 'all':
                reflected_array[:, 2] = -reflected_array[:, 2]
                reflected_array[:, 1] = -reflected_array[:, 1]
                reflected_array[:, 0] = -reflected_array[:, 0]
            else:
                raise ValueError("Invalid plane specified. Use 'xy', 'xz', or 'yz'.")
            reflected_array = align_points(reflected_array, array2)
            if abs(reflected_array - array2).sum() < sum:
                sum = abs(reflected_array - array2).sum()
                reflected_array_final = reflected_array
            return reflected_array_final

    def cut_submatrix(array, N):
        non_zero_submatrix = array[0:N, 0:N]
        return non_zero_submatrix

    def visualization_3d(self):
        points_of_interests = [41, 23, 22, 80, 118, 28]
        for interest in points_of_interests:
            seq = seqs[interest].reshape(64, 4)
            array = pred[interest].reshape(64, 64)
            array_true = true[interest].reshape(64, 64)
            non_zero_submatrix_true = find_min(array_true)
            non_zero_submatrix = cut_submatrix(array, non_zero_submatrix_true.shape[0])
            points_3d = distance_geometry(non_zero_submatrix)
            points_3d_true = distance_geometry(non_zero_submatrix_true)
            points_3d_true = reflect_across_plane(points_3d_true, points_3d)
            x_lim, y_lim, z_lim = get_global_limits([points_3d, points_3d_true])
            draw_point_cloud(points_3d, points_3d_true, seq, os.path.join(folder, str(interest) + "pred.png"),
                             x_lim=x_lim,
                             y_lim=y_lim, z_lim=z_lim)
            print("INTEREST NUMBER", interest)
            print("3D Points shape:", points_3d.shape)

if __name__ == '__main__':
    experiment_name = "experiments/128_VAE_wider_encoder_nz_256_symmetric_black"
    visualizer = ExperimentVisualizer(experiment_name)
    visualizer.visualize_structure()
