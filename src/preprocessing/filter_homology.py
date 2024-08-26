import os
from Bio.Align import PairwiseAligner
from Bio.PDB import PDBParser
import shutil
import numpy as np
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt


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

def perform_tsne_with_distance(distance_matrix):
    """Perform t-SNE on a precomputed distance matrix."""
    tsne = TSNE(n_components=2, metric='precomputed', init='random', random_state=42)
    tsne_result = tsne.fit_transform(distance_matrix)
    return tsne_result

def cluster_rna_structures(directory, output, threshold=80):
    print("Filtering and clustering homology with threshold = ", threshold," This could take several minutes...")
    pdb_files = [f for f in os.listdir(directory) if f.endswith('.ent')]
    sequences = {}
    for pdb_file in pdb_files:
        pdb_path = os.path.join(directory, pdb_file)
        #print(f" Processing {pdb_path} ")
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('structure', pdb_path)
        s = ""
        for chain in structure[0]:
            for residue in chain:
                if residue.get_resname() in ['A', 'C', 'G', 'U']:
                    s += residue.get_resname()
            if (len(s) > 0):
                sequences[pdb_file] = s
                break
    clusters = []
    used_files = set()
    for pdb_file, sequence in sequences.items():
        if pdb_file in used_files:
            continue
        cluster = [pdb_file]
        used_files.add(pdb_file)
        for other_pdb, other_sequence in sequences.items():
            if other_pdb in used_files:
                continue
            homology = calculate_homology(sequence, other_sequence)
            if homology >= threshold:
                cluster.append(other_pdb)
                used_files.add(other_pdb)
        clusters.append(cluster)

    #distance_matrix = compute_distance_matrix(sequences)
    #tsne_result = perform_tsne_with_distance(distance_matrix)

    labels = np.zeros(len(sequences))
    for i, cluster in enumerate(clusters):
        for pdb_file in cluster:
            labels[list(sequences.keys()).index(pdb_file)] = i

    #plot_tsne(tsne_result, labels)
    for i, cluster in enumerate(clusters, 1):
        cluster_dir = f'{output}/cluster_{i}/'
        os.makedirs(cluster_dir, exist_ok=True)
        for pdb_file in cluster:
            shutil.copy(os.path.join(directory, pdb_file), cluster_dir)



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


def perform_tsne(features):
    """Perform t-SNE on feature vectors."""
    feature_matrix = np.array(list(features.values()))
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(feature_matrix)
    return tsne_result

def plot_tsne(tsne_result, labels):
    """Plot t-SNE results."""
    df = pd.DataFrame(tsne_result, columns=['x', 'y'])
    df['label'] = labels

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(df['x'], df['y'], c=df['label'], cmap='tab10', alpha=0.7)
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title('t-SNE of RNA Clusters')
    plt.show()

def clean_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

if __name__ == '__main__':
    directory, output = 'data/rna_data_v1', 'data/clusters'
    clean_folder(output)
    cluster_rna_structures(directory, output, threshold=80)