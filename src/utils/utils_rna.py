from Bio.PDB.vectors import calc_dihedral, calc_angle
from Bio.PDB import Chain
from scipy.spatial import distance_matrix
import numpy as np
import copy


def create_distance_matrix(pos_matrix, padding_value=0.0, max_length =None):
    dist_matr_list = []
    mask_list = []
    for i in range(len(pos_matrix)):
        dist_matr = distance_matrix(pos_matrix[i], pos_matrix[i])
        length = len(pos_matrix[i])
        new_dist_matr = np.pad(dist_matr, [(0, max_length - length), (0, max_length - length)], mode='constant', constant_values=0.0)
        dist_matr_list.append(new_dist_matr)
        mask = np.zeros((max_length, max_length))
        mask[:length, :length] = 1.0
        mask_list.append(mask)
    return dist_matr_list, mask_list


def modify_shift(coords):
    """
    Modify coordinates by shifting them based on the first coordinate.

    Args:
        coords: List of coordinates arrays to be shifted.

    Returns:
        A list of modified coordinates arrays.
    """
    full_coords = []
    for i in range(len(coords)):
        np_matr = np.array(coords[i])
        # determine the length of the sequence
        np_matr_sum = np_matr.sum(axis=1)
        size_trimmed = len(np.trim_zeros(np_matr_sum))
        np_matr_clean = np_matr[:size_trimmed]
        # subtract the first coordinate
        np_matr_clean = np_matr_clean - np_matr_clean[0]
        # attach to the original matrix
        np_matr[:size_trimmed] = np_matr_clean

        full_coords.append(np_matr)

    return full_coords


def rotate_molecule(sequence):
    """
    Rotate molecules in a sequence to a standardized orientation.

    Args:
        sequence: A numpy array of shape (N, 3, max_len) representing a batch of sequences.

    Returns:
        A numpy array of rotated sequences.
    """
    #
    # sequence [N -batch size, 3, max_len]
    #
    seq_output = []
    for j in range(0, sequence.shape[0]):
        chain_center_of_mass = sequence[j, :, :]
        translation = copy.deepcopy(chain_center_of_mass[0])
        translation_mask = chain_center_of_mass.sum(axis=1)
        translation_mask[translation_mask != 0] = 1
        print(translation_mask.shape)
        print(chain_center_of_mass.shape)
        for i in range(len(chain_center_of_mass)):
            chain_center_of_mass[i] = chain_center_of_mass[i] - translation
        x = chain_center_of_mass[1][0]
        y = chain_center_of_mass[1][1]
        z = chain_center_of_mass[1][2]
        theta = np.arctan(-z / y)
        alpha = np.arctan(-(np.cos(theta) * y - np.sin(theta) * z) / x)
        x_rotation = np.array([[1, 0, 0],
                               [0, np.cos(theta), -np.sin(theta)],
                               [0, np.sin(theta), np.cos(theta)]])
        z_rotation = np.array([[np.cos(alpha), -np.sin(alpha), 0],
                              [np.sin(alpha), np.cos(alpha), 0],
                              [0, 0, 1]])
        transponded_chain_center_of_mass = np.transpose(chain_center_of_mass)
        first_rotation = np.dot(x_rotation, transponded_chain_center_of_mass)
        second_rotation = np.dot(z_rotation, first_rotation)
        matrix = np.transpose(second_rotation)
        first_2_nucleotides, all_other_nucleotides = np.vsplit(matrix, [2])
        translation_after_2nd_nucleotide = first_2_nucleotides[1]

        for i in range(len(all_other_nucleotides)):
            all_other_nucleotides[i] = all_other_nucleotides[i] - translation_after_2nd_nucleotide
        z_3d_nucleotid = all_other_nucleotides[0][2]
        y_3d_nucleotid = all_other_nucleotides[0][1]
        beta = np.arctan(-z_3d_nucleotid / y_3d_nucleotid)
        x_rotation_1 = np.array([[1, 0, 0],
                                [0, np.cos(beta), -np.sin(beta)],
                                [0, np.sin(beta), np.cos(beta)]])
        transponed_all_after_2nd = np.transpose(all_other_nucleotides)
        final_rotation = np.dot(x_rotation_1, transponed_all_after_2nd)
        temporal_matrix = np.transpose(final_rotation)
        for i in range(len(temporal_matrix)):
            temporal_matrix[i] = temporal_matrix[i] + translation_after_2nd_nucleotide
        molecule = np.vstack((first_2_nucleotides, temporal_matrix))

        translation_mask_reshaped = translation_mask[:, np.newaxis]
        molecule = molecule * translation_mask_reshaped
        seq_output.append(molecule)
    seq_output = np.array(seq_output)
    return seq_output


def add_padding_list(listy, value, max_length=None):
    """
    Add padding to a list of matrices to ensure uniform length.

    Args:
        listy: List of matrices to pad.
        value: Padding value.
        max_length: Optional maximum length to pad to.

    Returns:
        A numpy array with padded matrices.
    """
    if max_length is None:
        limit = len(max(listy, key=lambda x: len(x)))
    else:
        limit = max_length
    pre_tensor = np.full([len(listy), limit, len(listy[0][0])], value)
    for i, j in enumerate(listy):
        pre_tensor[i][0:len(j)] = j
    return pre_tensor


def add_padding_value(listy, value, max_length=None):
    """
    Add padding to a list to ensure uniform length, for a single value padding.

    Args:
       listy: List to pad.
       value: Padding value.
       max_length: Optional maximum length to pad to.

    Returns:
       A numpy array with padded values.
    """
    if max_length is None:
        limit = len(max(listy, key=lambda x: len(x)))
    else:
        limit = max_length
    pre_tensor = np.full([len(listy), limit], value)
    for i, j in enumerate(listy):
        pre_tensor[i][0:len(j)] = j
    return pre_tensor


def single_chain_angles(chain: Chain.Chain) -> np.ndarray:
    """
    Calculate angles for a single chain of residues.

    Args:
        chain: A Chain object from Bio.PDB.

    Returns:
        A numpy array of angles for the given chain.
    """
    angles = []
    residues = []
    for i in list(chain.get_residues()):
        if i.get_resname() in constants.RNA_NUCLEOBASE:
            residues.append(i)
        else:
            pass
            # print(i.get_resname())

    for i in range(len(residues)):
        if i == 0:
            p, c, n = None, residues[i], residues[i + 1]
        elif i == len(residues) - 1:
            p, c, n = residues[i - 1], residues[i], None
        else:
            p, c, n = residues[i - 1], residues[i], residues[i + 1]

        ang = torsion_angles(previous_atom=p, current_atom=c, next_atom=n)
        angles.append(ang)

    return np.array(angles)


def torsion_angles(previous_atom, current_atom, next_atom):
    """
    Calculate torsion angles given three atoms.

    Args:
        previous_atom: The previous atom in the sequence.
        current_atom: The current atom.
        next_atom: The next atom in the sequence.

    Returns:
        A numpy array of calculated torsion angles.
    """
    atom1 = {a.get_name(): a.get_vector() for a in current_atom}
    alpha, beta, gamma, delta, epsilon, zeta, chi = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    if all(i in atom1.keys() for i in ["O5'", "C4'", "C3'", "O3'"]):
        gamma = calc_dihedral(atom1["O5'"], atom1["C5'"], atom1["C4'"], atom1["C3'"])

    if all(i in atom1.keys() for i in ["C5'", "C4'", "C3'", "O3'"]):
        delta = calc_dihedral(atom1["C5'"], atom1["C4'"], atom1["C3'"], atom1["O3'"])

    if previous_atom is not None:
        atom0 = {a.get_name(): a.get_vector() for a in previous_atom}
        if all(i in atom1.keys() for i in ["P", "O5'", "C5'", "C4'"]):
            beta = calc_dihedral(atom1["P"], atom1["O5'"], atom1["C5'"], atom1["C4'"])
        if all(i in atom1.keys() for i in ["P", "O5'", "C5'"]) and "O3'" in atom0.keys():
            alpha = calc_dihedral(atom0["O3'"], atom1["P"], atom1["O5'"], atom1["C5'"])

    if next_atom is not None:
        atom2 = {a.get_name(): a.get_vector() for a in next_atom}
        if all(i in atom1.keys() for i in ["C4'", "C3'", "O3'"]) and "P" in atom2.keys():
            epsilon = calc_dihedral(atom1["C4'"], atom1["C3'"], atom1["O3'"], atom2["P"])
        if all(i in atom1.keys() for i in ["C3'", "O3'"]) and all(i in atom2.keys() for i in ["P", "O5'"]):
            zeta = calc_dihedral(atom1["C3'"], atom1["O3'"], atom2["P"], atom2["O5'"])

    if current_atom.get_resname() in ['A', 'G']:
        if all(i in atom1.keys() for i in ["O4'", "C1'", "N9", "C4"]):
            chi = calc_dihedral(atom1["O4'"], atom1["C1'"], atom1["N9"], atom1["C4"])
    else:
        if all(i in atom1.keys() for i in ["O4'", "C1'", "N1", "C2"]):
            chi = calc_dihedral(atom1["O4'"], atom1["C1'"], atom1["N1"], atom1["C2"])

    return np.array([alpha, beta, gamma, delta, epsilon, zeta, chi])


def bond_dist(previous_atom, current_atom, next_atom):
    """
    Calculate bond distances given atoms.

    Args:
        previous_atom: The previous atom in the sequence.
        current_atom: The current atom.
        next_atom: The next atom in the sequence.

    Returns:
        A numpy array of bond distances.
    """

    distances = {}

    atom1 = {a.get_name(): a for a in current_atom}

    distances["O5'C5'"] = atom1["C5'"] - atom1["O5'"]
    distances["C5'C4'"] = atom1["C4'"] - atom1["C5'"]
    distances["C4'O4'"] = atom1["O4'"] - atom1["C4'"]
    distances["C4'C3'"] = atom1["C3'"] - atom1["C4'"]
    distances["C3'O3'"] = atom1["O3'"] - atom1["C3'"]

    if previous_atom is None:
        distances["bO3'P"] = np.nan
        distances["bPO5'"] = np.nan
    else:
        atom0 = {a.get_name(): a for a in previous_atom}
        distances["bO3'P"] = atom1["P"] - atom0["O3'"]
        distances["bPO5'"] = atom1["O5'"] - atom1["P"]

    if next_atom is None:
        distances["eO3'P"] = np.nan
        distances["ePO5'"] = np.nan
    else:
        atom2 = {a.get_name(): a for a in next_atom}
        distances["eO3'P"] = atom2["P"] - atom1["O3'"]
        distances["ePO5'"] = atom2["O5'"] - atom2["P"]

    return np.array([distances["bO3'P"], distances["bPO5'"], distances["O5'C5'"], distances["C5'C4'"],
                     distances["C4'O4'"], distances["C4'C3'"], distances["C3'O3'"], distances["eO3'P"],
                     distances["ePO5'"]])


def bond_angles(previous_atom, current_atom, next_atom):
    """
    Calculate bond angles given atoms.

    Args:
        previous_atom: The previous atom in the sequence.
        current_atom: The current atom.
        next_atom: The next atom in the sequence.

    Returns:
        A numpy array of bond angles.
    """
    angles = {}
    atom1 = {a.get_name(): a.get_vector() for a in current_atom}

    angles["O5'C5'C4'"] = calc_angle(atom1["O5'"], atom1["C5'"], atom1["C4'"])
    angles["C5'C4'C3'"] = calc_angle(atom1["C5'"], atom1["C4'"], atom1["C3'"])
    angles["C5'C4'O4'"] = calc_angle(atom1["C5'"], atom1["C4'"], atom1["O4'"])
    angles["C4'C3'O3'"] = calc_angle(atom1["C4'"], atom1["C3'"], atom1["O3'"])

    if previous_atom is None:
        angles["bO3'PO5'"] = np.nan
        angles["bPO5'C5'"] = np.nan
    else:
        atom0 = {a.get_name(): a.get_vector() for a in previous_atom}
        angles["bO3'PO5'"] = calc_angle(atom0["O3'"], atom1["P"], atom1["O5'"])
        angles["bPO5'C5'"] = calc_angle(atom1["P"], atom1["O5'"], atom1["C5'"])

    if next_atom is None:
        angles["eO3'PO5'"] = np.nan
        angles["ePO5'C5'"] = np.nan
    else:
        atom2 = {a.get_name(): a.get_vector() for a in next_atom}
        angles["eO3'PO5'"] = calc_angle(atom1["C3'"], atom1["O3'"], atom2["P"])
        angles["ePO5'C5'"] = calc_angle(atom1["O3'"], atom2["P"], atom2["O5'"])

    return np.array([angles["bO3'PO5'"], angles["bPO5'C5'"], angles["O5'C5'C4'"], angles["C5'C4'C3'"],
                     angles["C5'C4'O4'"], angles["C4'C3'O3'"], angles["eO3'PO5'"], angles["ePO5'C5'"]])


def center_of_mass_chain(chain: Chain.Chain):
    """
    Calculate the center of mass for each residue in a chain.

    Args:
        chain: A Chain object from Bio.PDB.

    Returns:
        A numpy array representing the center of mass of each residue.
    """
    coms = []
    residues = []

    for i in list(chain.get_residues()):
        if i.get_resname() in {'A', 'U', 'G', 'C'}:
            residues.append(i)

    for residue in residues:
        atoms = list(residue.get_atoms())
        x, y, z = 0, 0, 0
        for atom in atoms:
            coords = atom.get_coord()
            x += coords[0]
            y += coords[1]
            z += coords[2]
        coms.append([x/len(atoms), y/len(atoms), z/len(atoms)])

    return np.array(coms)


def distance(pointA, pointB):
    """
    Calculate the Euclidean distance between two points.

    Args:
        pointA: The first point as a numpy array.
        pointB: The second point as a numpy array.

    Returns:
        The Euclidean distance between pointA and pointB.
    """
    return np.sqrt((pointB[0]-pointA[0])**2 + (pointB[1]-pointA[1])**2 + (pointB[2]-pointA[2])**2)


def calculate_distance_matrix(center_mass_list):
    """
    Calculate the distance matrix for a list of center mass points.

    Args:
        center_mass_list: List of center mass points.

    Returns:
        A distance matrix for the given list of points.
    """
    dist_mat = np.empty((len(center_mass_list), len(center_mass_list)))
    for i in range(len(center_mass_list)):
        for j in range(len(center_mass_list)):
            dist_mat[i][j] = distance(center_mass_list[i], center_mass_list[j])
    return dist_mat


def extract_backbone_coords(residue):
    """
    Extract coordinates of backbone atoms from a residue.

    Args:
        residue: A residue object from Bio.PDB.

    Returns:
        A list of coordinates for backbone atoms.
    """
    coords = []
    atoms = list(residue.get_atoms())
    for atom in atoms:
        if atom.get_name() in ["P", "O5'", "O3'", "C5'", "C4'", "C3'"]:
            coords.append(atom.get_coord())
    return coords


def extract_backbone_chain(chain: Chain.Chain):
    """
    Extract backbone coordinates for each residue in a chain.

    Args:
        chain: A Chain object from Bio.PDB.

    Returns:
        A list of backbone coordinates for the chain.
    """
    backbone_coords = []
    residues = list(chain.get_residues())
    for i in range(len(residues)):
        backbone = extract_backbone_coords(residues[i])
        for back in backbone:
            backbone_coords.append(back)
    return backbone_coords
