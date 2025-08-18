import pickle
import numpy as np
from scipy.linalg import eigh
import os
import argparse

HOME_DIR = "/home/leo/Research/MaximData/last_src/src/exp/test_calcs/"
SIZE = 64

def find_distance_submatrix(matrix, threshold=1.0):
    """
    Find the submatrix containing distances (values > threshold).
    Returns the submatrix and the indices of rows/cols that contain it.
    """
    # Find all positions where values > threshold
    mask = matrix > threshold
    
    # Find rows and columns that have at least one value > threshold
    rows_with_distances = np.any(mask, axis=1)
    cols_with_distances = np.any(mask, axis=0)
    
    # Get indices
    row_indices = np.where(rows_with_distances)[0]
    col_indices = np.where(cols_with_distances)[0]
    
    if len(row_indices) == 0 or len(col_indices) == 0:
        raise ValueError("No distances > threshold found in matrix")
    
    # Extract submatrix
    min_idx = min(row_indices.min(), col_indices.min())
    max_idx = max(row_indices.max(), col_indices.max()) + 1
    
    submatrix = matrix[min_idx:max_idx, min_idx:max_idx]
    
    return submatrix, min_idx, max_idx


def distance_matrix_to_3d_multiple(distance_matrix, num_solutions=8):
    """
    Convert a distance matrix to multiple 3D coordinate sets using classical MDS.
    Returns multiple solutions using different eigenvector combinations and orientations.
    """
    n = distance_matrix.shape[0]
    
    # Check if matrix is symmetric
    if not np.allclose(distance_matrix, distance_matrix.T):
        print("Warning: Distance matrix is not perfectly symmetric. Symmetrizing...")
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
    
    # Center the distance matrix
    # J = I - (1/n) * 1 * 1^T  (centering matrix)
    J = np.eye(n) - np.ones((n, n)) / n
    
    # B = -0.5 * J * D^2 * J  (double-centered squared distance matrix)
    D_squared = distance_matrix ** 2
    B = -0.5 * J @ D_squared @ J
    
    # Eigendecomposition
    eigenvalues, eigenvectors = eigh(B)
    
    # Sort by eigenvalues (largest first)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Set negative eigenvalues to 0
    eigenvalues[eigenvalues < 0] = 0
    
    # Find how many positive eigenvalues we have
    num_positive = np.sum(eigenvalues > 1e-10)
    print(f"Number of positive eigenvalues: {num_positive}")
    
    solutions = []
    
    # Solution 1-4: Using top 3 eigenvectors with different sign combinations
    if num_positive >= 3:
        base_coords = eigenvectors[:, :3] @ np.diag(np.sqrt(eigenvalues[:3]))
        
        # Different sign combinations for the three axes
        sign_combinations = [
            (1, 1, 1),   # Original
            (-1, 1, 1),  # Flip X
            (1, -1, 1),  # Flip Y
            (1, 1, -1),  # Flip Z
            (-1, -1, 1), # Flip X and Y
            (-1, 1, -1), # Flip X and Z
            (1, -1, -1), # Flip Y and Z
            (-1, -1, -1) # Flip all
        ]
        
        for i, signs in enumerate(sign_combinations[:min(num_solutions, 8)]):
            coords = base_coords.copy()
            coords[:, 0] *= signs[0]
            coords[:, 1] *= signs[1]
            coords[:, 2] *= signs[2]
            solutions.append(coords)
    # If we have more than 3 positive eigenvalues, try different combinations
    if num_positive >= 4 and len(solutions) < num_solutions:
        # Solution using eigenvectors 1, 2, 4
        coords = eigenvectors[:, [0, 1, 3]] @ np.diag(np.sqrt([eigenvalues[0], eigenvalues[1], eigenvalues[3]]))
        solutions.append(coords)
    
    if num_positive >= 4 and len(solutions) < num_solutions:
        # Solution using eigenvectors 1, 3, 4
        coords = eigenvectors[:, [0, 2, 3]] @ np.diag(np.sqrt([eigenvalues[0], eigenvalues[2], eigenvalues[3]]))
        solutions.append(coords)
    
    if num_positive >= 4 and len(solutions) < num_solutions:
        # Solution using eigenvectors 2, 3, 4
        coords = eigenvectors[:, [1, 2, 3]] @ np.diag(np.sqrt([eigenvalues[1], eigenvalues[2], eigenvalues[3]]))
        solutions.append(coords)
    
    # Ensure we have at least one solution
    if len(solutions) == 0:
        print("Warning: Could not generate valid 3D coordinates. Using first 3 dimensions anyway.")
        coords = eigenvectors[:, :3] @ np.diag(np.sqrt(np.maximum(eigenvalues[:3], 0)))
        solutions.append(coords)
    
    print(f"Generated {len(solutions)} different 3D solutions")
    return solutions


def write_rna_pdb(coordinates, sequence, filename, chain_id='A'):
    """
    Write RNA 3D coordinates to a PDB file.
    
    Args:
        coordinates: Nx3 array of 3D coordinates
        sequence: String of RNA nucleotides (A, U, G, C)
        filename: Output PDB filename
        chain_id: Chain identifier (default 'A')
    """
    if len(coordinates) != len(sequence):
        raise ValueError(f"Number of coordinates ({len(coordinates)}) must match sequence length ({len(sequence)})")
    
    # RNA residue names in PDB format
    rna_residue_names = {
        'A': '  A',
        'U': '  U', 
        'G': '  G',
        'C': '  C'
    }
    
    with open(filename, 'w') as f:
        # Write header
        f.write("REMARK   Generated from distance matrix\n")
        f.write(f"REMARK   Sequence: {sequence}\n")
        
        atom_number = 1
        
        for i, (coord, nucleotide) in enumerate(zip(coordinates, sequence)):
            residue_number = i + 1
            
            if nucleotide not in rna_residue_names:
                print(f"Warning: Unknown nucleotide '{nucleotide}' at position {i}. Using 'UNK'.")
                res_name = 'UNK'
            else:
                res_name = rna_residue_names[nucleotide]
            
            # Write atom record
            # Using P (phosphorus) as the representative atom for each nucleotide
            atom_name = 'P'
            x, y, z = coord
            
            # PDB format: 
            # ATOM serial# name resName chain resNum x y z occupancy tempFactor element
            pdb_line = (
                f"ATOM  {atom_number:5d}  {atom_name:<3s} {res_name} {chain_id}{residue_number:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           P\n"
            )
            f.write(pdb_line)
            
            atom_number += 1
        
        # Write END record
        f.write("END\n")


def convert_distance_matrix_to_pdb_multiple(matrix, sequence, output_filename='output.pdb', threshold=1.0, num_solutions=8):
    """
    Main function to convert a distance matrix to multiple PDB files.
    
    Args:
        matrix: 64x64 numpy array containing the distance matrix
        sequence: RNA sequence string
        output_filename: Base name of output PDB file (will append _1, _2, etc.)
        threshold: Minimum value to consider as a distance (default 1.0)
        num_solutions: Number of different 3D solutions to generate (default 8)
    """
    print("Finding distance submatrix...")
    submatrix, start_idx, end_idx = find_distance_submatrix(matrix, threshold)
    submatrix_size = end_idx - start_idx
    
    print(f"Found submatrix of size {submatrix_size}x{submatrix_size} at indices [{start_idx}:{end_idx}]")
    
    # Extract corresponding subsequence
    if len(sequence) < submatrix_size:
        print(f"Sequence length ({len(sequence)}) is shorter than submatrix size ({submatrix_size})")
        return None, None, None
    
    subsequence = sequence[start_idx:end_idx]
    print(f"Corresponding sequence: {subsequence}")
    
    # Convert to multiple 3D coordinate sets
    print(f"Converting to {num_solutions} different 3D coordinate sets...")
    all_coords_3d = distance_matrix_to_3d_multiple(submatrix, num_solutions)
    # Save each solution
    base_name = output_filename.rsplit('.', 1)[0]  # Remove extension
    extension = '.pdb' if '.' in output_filename else ''
    
    saved_files = []
    for i, coords_3d in enumerate(all_coords_3d):
        # Generate filename with index
        indexed_filename = f"{base_name}_{i+1}{extension}"
        
        # Print statistics for this solution
        print(f"\nSolution {i+1}:")
        print(f"  Coordinate ranges:")
        print(f"    X: [{coords_3d[:, 0].min():.3f}, {coords_3d[:, 0].max():.3f}]")
        print(f"    Y: [{coords_3d[:, 1].min():.3f}, {coords_3d[:, 1].max():.3f}]")
        print(f"    Z: [{coords_3d[:, 2].min():.3f}, {coords_3d[:, 2].max():.3f}]")
        
        # Calculate and print RMSD from first solution (if not first)
        if i > 0:
            rmsd = np.sqrt(np.mean(np.sum((coords_3d - all_coords_3d[0])**2, axis=1)))
            print(f"  RMSD from solution 1: {rmsd:.3f}")
        
        # Write PDB file
        print(f"  Writing PDB file: {indexed_filename}")
        write_rna_pdb(coords_3d, subsequence, indexed_filename)
        saved_files.append(indexed_filename)
    
    print(f"\nDone! Saved {len(saved_files)} PDB files")
    return all_coords_3d, subsequence, saved_files


# Keep the old function for backward compatibility
def convert_distance_matrix_to_pdb(matrix, sequence, output_filename='output.pdb', threshold=1.0):
    """
    Main function to convert a distance matrix to a single PDB file.
    (Backward compatible version that returns only the first solution)
    """
    # Use the multiple version but only return the first solution
    all_coords, subseq, _ = convert_distance_matrix_to_pdb_multiple(
        matrix, sequence, output_filename, threshold, num_solutions=8
    )
    if (all_coords is None):
        return None
    return all_coords[0], subseq

def fasta_to_dict(filename):
    result = dict()
    key = ""
    for line in open(filename):
        if (">" in line):
            key = line.rstrip("\n").replace(">","")
        else:
            result[key] = line.rstrip("\n")
    return(result)

with open(f'{HOME_DIR}/pred.pickle', 'rb') as f:
	data = pickle.load(f)

with open(f'{HOME_DIR}/pdb.pickle', 'rb') as f:
	pdbs = pickle.load(f)

with open(f'{HOME_DIR}/sequences.pickle', 'rb') as f:
	seqs = pickle.load(f)


parser = argparse.ArgumentParser(
    description='Parse VAE output and making structures for further processing'
)
parser.add_argument('input_dir', help='Input RNA PDB file')
parser.add_argument('--fasta_sequences', type=str, help='FASTA file with original sequence', required=False)
args = parser.parse_args()

if args.fasta_sequences is None:
    for index, pdb in enumerate(pdbs):
        matrix = data[index].reshape((SIZE,SIZE))
        seq_one_hot_encoding = seqs[index,:].reshape(256//4,4)
        res = ""
        for seq in seq_one_hot_encoding:
            if np.array_equal(seq.astype(int), np.array([1,0,0,0])):
                res += "A"
            if np.array_equal(seq.astype(int), np.array([0,1,0,0])):
                res += "C"
            if np.array_equal(seq.astype(int), np.array([0,0,1,0])):
                res += "G"
            if np.array_equal(seq.astype(int), np.array([0,0,0,1])):
                res += "U"
    #        print(seq.astype(int))
        print(pdb)
    #    print(res)
        result = convert_distance_matrix_to_pdb(matrix, res, pdb+".pdb")
else:
    seqs = fasta_to_dict(args.fasta_sequences)
    for index, pdb in enumerate(pdbs):
        if (pdb in seqs.keys()):
            matrix = data[index].reshape((SIZE,SIZE))
            result = convert_distance_matrix_to_pdb(matrix, seqs[pdb], pdb+".pdb")