import os
from Bio.Align import PairwiseAligner
from Bio.PDB import PDBParser
import shutil
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster,dendrogram
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from collections import Counter

def compute_distance_matrix(sequences):
    """Compute a distance matrix based on pairwise alignment scores."""
    aligner = PairwiseAligner()
    num_sequences = len(sequences)
    distance_matrix = np.zeros((num_sequences, num_sequences))
    for i, seq1 in enumerate(sequences.values()):
        for j, seq2 in enumerate(sequences.values()):
            if i < j:
                alignment_score = aligner.align(seq1, seq2).score
                distance_matrix[i, j] = alignment_score
                distance_matrix[j, i] = alignment_score
    return distance_matrix


def calculate_homology(seq1, seq2):
    aligner = PairwiseAligner()
    alignments = aligner.align(seq1, seq2)
    best_alignment = alignments[0]
    matches = sum(1 for a, b in zip(best_alignment.target, best_alignment.query) if a == b)
    homology_percentage = (matches / min(len(seq1), len(seq2))) * 100
    return homology_percentage


def compute_homology_matrix(sequences):
    """Compute a homology matrix where each entry is the homology score between two sequences."""
    print("Computing the homology matrix...")
    num_sequences = len(sequences)
    homology_matrix = np.zeros((num_sequences, num_sequences))
    for i, seq1 in enumerate(sequences.values()):
        for j, seq2 in enumerate(sequences.values()):
            if i < j:
                homology = calculate_homology(seq1, seq2)
                homology_matrix[i, j] = homology
                homology_matrix[j, i] = homology
    return homology_matrix


def cluster_sequences_by_homology(homology_matrix, threshold):
    """
    Cluster sequences based on homology using hierarchical clustering.
    Homology values are used to create a distance matrix (100 - homology).
    """
    print("Performing hierarchical clustering ... ")
    distance_matrix = 100 - homology_matrix
    np.fill_diagonal(distance_matrix, 0)
    condensed_distance_matrix = squareform(distance_matrix)
    print(condensed_distance_matrix)
    Z = linkage(condensed_distance_matrix, 'average')
    fig = plt.figure(figsize=(25, 10))
    # dn = dendrogram(Z)
    # plt.show()
    # print("linkage Z",Z)
    clusters = fcluster(Z, t= threshold, criterion='distance')
    return clusters


def visualize_clusters_with_pca(homology_matrix, clusters, filename):
    """Visualize RNA clusters in 2D using PCA, highlighting the 20 largest clusters."""

    distance_matrix = 100 - homology_matrix
    np.fill_diagonal(distance_matrix, 0)
    pca_model = PCA(n_components=2)
    pca_result = pca_model.fit_transform(distance_matrix)

    cluster_counts = Counter(clusters)
    largest_clusters = [cluster for cluster, _ in cluster_counts.most_common(10)]
    largest_clusters_sorted = sorted(largest_clusters)
    print("Largest 10 clusters and their sizes:")
    for cluster, count in cluster_counts.most_common(20):
        print(f"Cluster {cluster}: {count} points")

    # Default color for the "other" clusters
    default_color = (0.68, 0.85, 0.9, 1.0)  # Light blue

    # Initialize colors array with default light blue color
    colors = np.array([default_color] * len(clusters))

    cmap = plt.get_cmap('tab20')  # Colormap with 20 distinct colors

    # Assign colors to the largest 20 clusters
    for i, cluster in enumerate(largest_clusters):
        cluster_indices = np.where(clusters == cluster)[0]  # Get indices of points in this cluster
        color = cmap(i % 10)  # Get color from colormap as a string, not as an RGBA array
        colors[cluster_indices] = color  # Assign the color to those indices

    # Create a scatter plot of the PCA results with customized colors
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=colors, s=2.5,alpha=0.9)

    # Add plot title and labels
    plt.title('PCA Visualization of RNA Clusters (Largest 20 Highlighted)')
    plt.xlabel('PCA First Component')
    plt.ylabel('PCA Second Component')

    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(i % 20), markersize=10,
                              label=f'Cluster {cluster}')
                       for i, cluster in enumerate(largest_clusters_sorted)]
    legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=default_color, markersize=10,
                                  label='Other clusters'))

    plt.legend(handles=legend_elements, loc='best', title='Clusters')
    plt.savefig(filename)


def perform_kmeans_and_visualize(homology_matrix, clusters_original, filename):
    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(homology_matrix)
    unique, counts = np.unique(clusters, return_counts=True)
    largest_cluster = unique[np.argmax(counts)]
    largest_cluster_indices = np.where(clusters == largest_cluster)[0]
    largest_cluster_homology_matrix = homology_matrix[largest_cluster_indices][:, largest_cluster_indices]

    distance_matrix = 100 - largest_cluster_homology_matrix
    np.fill_diagonal(distance_matrix, 0)

    # PCA with 2 components
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(distance_matrix)

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=clusters_original[largest_cluster_indices],
                          cmap='tab20c', alpha=0.9)
    plt.colorbar(scatter, label='Original Cluster')
    plt.title(f'PCA Visualization of the Largest K-means Cluster (Cluster {largest_cluster}) with Original Colors')
    plt.xlabel('PCA First Component')
    plt.ylabel('PCA Second Component')
    plt.savefig(filename)

def cluster_rna_structures(directory, output, analysis_folder, threshold=60):
    print("Filtering and clustering homology with threshold = ", threshold, " This could take several minutes...")
    pdb_files = [f for f in os.listdir(directory) if (f.endswith('.pdb')or f.endswith('.ent')) ]
    sequences = {}
    for pdb_file in pdb_files:
        pdb_path = os.path.join(directory, pdb_file)
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('structure', pdb_path)
        s = ""
        for chain in structure[0]:
            for residue in chain:
                if residue.get_resname() in ['A', 'C', 'G', 'U']:
                    s += residue.get_resname()
            if len(s) > 0:
                sequences[pdb_file] = s
                break

    homology_matrix = compute_homology_matrix(sequences)
    clusters = cluster_sequences_by_homology(homology_matrix, threshold)
    print("Distinct clusters", len(set(clusters)))
    clustered_sequences = {}

    for idx, pdb_file in enumerate(sequences.keys()):
        cluster_num = clusters[idx]
        if cluster_num not in clustered_sequences:
            clustered_sequences[cluster_num] = []
        clustered_sequences[cluster_num].append(pdb_file)

    # Save the clusters
    for key in clustered_sequences.keys():
        i = int(key)
        cluster_dir = f'{output}/cluster_{i}/'
        os.makedirs(cluster_dir, exist_ok=True)
        for pdb_file in clustered_sequences[i]:
            shutil.copy(os.path.join(directory, pdb_file), cluster_dir)

    umap_filename = os.path.join(analysis_folder, "pca_full_dataset.jpeg")
    visualize_clusters_with_pca(homology_matrix, clusters, umap_filename)
    #umap_kmeans_filename = os.path.join(analysis_folder, "pca_one_cluster.jpeg")
    #perform_kmeans_and_visualize(homology_matrix, clusters, umap_kmeans_filename)

def one_hot_encode(sequence):
    """One-hot encode RNA sequence."""
    mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'U': [0, 0, 0, 1]}
    return np.array([mapping[residue] for residue in sequence])


def compute_features(sequences):
    """Compute feature vectors for each RNA sequence."""
    features = {}
    for pdb_file, sequence in sequences.items():
        features[pdb_file] = np.mean(one_hot_encode(sequence), axis=0)
    return features



def clean_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"Deleted the folder: {folder_path}")


    # Create a new folder
    os.makedirs(folder_path)
    print(f"Created a new folder: {folder_path}")


if __name__ == '__main__':
    directory, output = 'data/rna_data_v1_small', 'data/clusters'
    clean_folder(output)
    cluster_rna_structures(directory, output, "example.jpg", threshold=60)
