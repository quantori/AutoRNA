import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from scipy.linalg import svd

class spatial_visualizer():
    @staticmethod
    def draw_point_cloud(points_3d,
                         points_3d_true,
                         features,
                         output_filename,
                         x_limit=None,
                         y_limit=None,
                         z_limit=None):
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
            ax.set_xlim(x_limit)
        if y_lim:
            ax.set_ylim(y_limit)
        if z_lim:
            ax.set_zlim(z_limit)

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
            ax.plot([x[i], x[i + 1]], [y[i], y[i + 1]], [z[i], z[i + 1]],
                    color='grey', linestyle='-', alpha=0.3)
            ax.plot([x1[i], x1[i + 1]], [y1[i], y1[i + 1]], [z1[i], z1[i + 1]],
                    color='darkred', linestyle='-', alpha=0.3)

        ax.grid(True)

        legend_elements = [
            Circle((0, 0), 0.1, facecolor='red', edgecolor='red', linewidth=0, label='A'),
            Circle((0, 0), 0.1, facecolor='green', edgecolor='green', linewidth=0, label='C'),
            Circle((0, 0), 0.1, facecolor='steelblue', edgecolor='steelblue', linewidth=0, label='G'),
            Circle((0, 0), 0.1, facecolor='orange', edgecolor='orange', linewidth=0, label='U'),
            Line2D([0], [0], color='grey', alpha=0.5, linestyle='-', linewidth=4, label='True'),
            Line2D([0], [0], color='darkred', alpha=0.5, linestyle='-', linewidth=4, label='Predicted')
        ]

        # In your plotting function:
        ax.legend(handles=legend_elements, loc='upper center', fontsize=10, frameon=False, handlelength=2,
                  handletextpad=1, labelspacing=0.5, ncol=6, bbox_to_anchor=(0.5, 1))
        plt.tight_layout()
        plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
        plt.savefig(output_filename, dpi=400)

    @staticmethod
    def compute_centroid(points):
        return np.mean(points, axis=0)

    @staticmethod
    def align_points(set1, set2):
        set1, set2 = np.asarray(set1), np.asarray(set2)
        centroid1 = spatial_visualizer.compute_centroid(set1)
        centroid2 = spatial_visualizer.compute_centroid(set2)
        set1_centered = set1 - centroid1
        set2_centered = set2 - centroid2
        h = set1_centered.T @ set2_centered
        u, s, vt = svd(h)
        r = vt.T @ u.T
        if np.linalg.det(r) < 0:
            vt[-1, :] *= -1
            r = vt.T @ u.T
        t = centroid2 - r @ centroid1
        aligned_set1 = (r @ set1.T).T + t
        return aligned_set1

    @staticmethod
    def get_global_limits(points_3d_list):
        all_points = np.concatenate(points_3d_list, axis=0)
        x_limit = (all_points[:, 0].min(), all_points[:, 0].max())
        y_limit = (all_points[:, 1].min(), all_points[:, 1].max())
        z_limit = (all_points[:, 2].min(), all_points[:, 2].max())
        return x_limit, y_limit, z_limit

    @staticmethod
    def distance_geometry(non_zero_submatrix):
        n = non_zero_submatrix.shape[0]
        j = np.eye(n) - np.ones((n, n)) / n
        b = -0.5 * j @ (non_zero_submatrix ** 2) @ j
        eigvals, eigvecs = np.linalg.eigh(b)
        idx = eigvals.argsort()[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        eigvals_3d = eigvals[:3]
        eigvecs_3d = eigvecs[:, :3]
        eigvals_3d_sqrt = np.sqrt(eigvals_3d).reshape(-1, 1)
        points_3d = eigvecs_3d @ np.diag(eigvals_3d_sqrt.flatten())
        return points_3d

    @staticmethod
    def find_min(a):
        rows, cols = np.nonzero(a)
        min_row, max_row = rows.min(), rows.max()
        min_col, max_col = cols.min(), cols.max()
        non_zero_submatrix = a[min_row:max_row + 1, min_col:max_col + 1]
        return non_zero_submatrix

    @staticmethod
    def reflect_across_plane(array, array2):
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
            reflected_array = spatial_visualizer.align_points(reflected_array, array2)
            reflected_array_final = None
            if abs(reflected_array - array2).sum() < sum:
                reflected_array_final = reflected_array
            return reflected_array_final

    @staticmethod
    def cut_submatrix(array, n):
        non_zero_matrix = array[0:n, 0:n]
        return non_zero_matrix
    @staticmethod
    def get_coordinates(seq, array, array_true):
        non_zero_submatrix_true = spatial_visualizer.find_min(array_true)
        non_zero_submatrix = spatial_visualizer.cut_submatrix(array, non_zero_submatrix_true.shape[0])
        points_3d = spatial_visualizer.distance_geometry(non_zero_submatrix)
        points_3d_true = spatial_visualizer.distance_geometry(non_zero_submatrix_true)
        points_3d_true = spatial_visualizer.reflect_across_plane(points_3d_true, points_3d)
        return points_3d, points_3d_true

if __name__ == "__main__":
    ff = open(
        'experiments/inference_batch/test_calcs/true.pickle',
        'rb')
    true = pickle.load(ff)
    ff.close()

    ff = open(
        'experiments/inference_batch/test_calcs/pred.pickle',
        'rb')
    pred = pickle.load(ff)
    ff.close()

    ff = open(
        'experiments/inference_batch/test_calcs/sequences.pickle',
        'rb')
    seqs = pickle.load(ff)
    ff.close()
    folder = "src/utils/viz_images"
    for interest in range(len(seqs)):
        try:
            seq = seqs[interest].reshape(64, 4)
            array = pred[interest].reshape(64, 64)
            array_true = true[interest].reshape(64, 64)
            points_3d, points_3d_true = spatial_visualizer.get_coordinates(seq, array,array_true)
            x_lim, y_lim, z_lim = spatial_visualizer.get_global_limits([points_3d, points_3d_true])
            spatial_visualizer.draw_point_cloud(points_3d,
                                            points_3d_true,
                                            seq,
                                            os.path.join(folder,
                                            str(interest) + "pred.png"),
                                            x_limit=x_lim,
                                            y_limit=y_lim,
                                            z_limit=z_lim)
            print("INTEREST NUMBER", interest)
            print("3D Points shape:", points_3d.shape)
        except:
            print("Error: NAN found")