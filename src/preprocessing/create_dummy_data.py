import random
import numpy as np

# Define possible sequence values as a global constant.
SEQ_VALUES = [
    [0, 0, 0, 1],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [1, 0, 0, 0]
]


def create_seq_pos(max_num=80):
    """Generates a sequence and positions based on random selection from SEQ_VALUES.

    This function creates a sequence of random length between 10 and `max_num` (inclusive),
    selecting elements from `SEQ_VALUES` at random. It also generates position data for
    each element in the sequence.

    Args:
        max_num: An int specifying the maximum length of the sequence. Default is 80.

    Returns:
        A tuple of two lists:
            - The first list contains the randomly selected sequences from `SEQ_VALUES`.
            - The second list contains position data for each sequence element.
    """
    randy = random.randrange(10, max_num)
    sequence_index = np.random.choice(np.arange(len(SEQ_VALUES)), randy)
    sequence = []
    position = []
    count = 0
    for index in sequence_index:
        count += 1
        sequence.append(SEQ_VALUES[index])
        position.append([count * 1.0, count * 1.0])
    return sequence, position


def create_dummy_dataset(n_seq=100, max_num=80):
    """Generates a dummy dataset consisting of sequences and their positions.

    This function iterates `N_seq` times, each time generating a sequence and its positions
    using the `create_seq_pos` function. It collects all sequences and positions into a dataset.

    Args:
        n_seq: An int specifying the number of sequences to generate. Default is 100.
        max_num: An int specifying the maximum length of any sequence. Passed to `create_seq_pos`.

    Returns:
        A dictionary containing two keys: 'coords' and 'seq'. Each key maps to a list of positions
        and sequences, respectively, for all generated data.
    """
    dataset = {'coords': [], 'seq': []}
    for _ in range(n_seq):
        sequence, position = create_seq_pos(max_num)
        dataset['seq'].append(sequence)
        dataset['coords'].append(position)
    return dataset


if __name__ == '__main__':
    # Generate a dummy dataset with 100 sequences, each having a maximum length of 15.
    dataset = create_dummy_dataset(n_seq=100, max_num=15)
