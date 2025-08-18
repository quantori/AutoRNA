import os
from collections import Counter

# List the paths of the three folders
folder1 = 'experiments/one_chain_1.1dropout0.6_base_test/train_val_test_split/test'
folder2 = 'experiments/one_chain_1.1dropout0.6_base_test/train_val_test_split/train'
folder3 = 'experiments/one_chain_1.1dropout0.6_base_test/train_val_test_split/val'

# Get all filenames from the folders
files_in_folder1 = os.listdir(folder1)
files_in_folder2 = os.listdir(folder2)
files_in_folder3 = os.listdir(folder3)

# Print number of files in each folder
print(f"Number of files in {folder1}:  {len(files_in_folder1)}")
print(f"Number of files in {folder2}: {len(files_in_folder2)}")
print(f"Number of files in {folder3}:  {len(files_in_folder3)}")

# Check for duplicates within each folder
duplicates_in_folder1 = [file for file, count in Counter(files_in_folder1).items() if count > 1]
duplicates_in_folder2 = [file for file, count in Counter(files_in_folder2).items() if count > 1]
duplicates_in_folder3 = [file for file, count in Counter(files_in_folder3).items() if count > 1]

print("Duplicate files within folder 1:")
print(duplicates_in_folder1)

print("Duplicate files within folder 2:")
print(duplicates_in_folder2)

print("Duplicate files within folder 3:")
print(duplicates_in_folder3)

# Combine all filenames in one list
all_files = files_in_folder1 + files_in_folder2 + files_in_folder3

# Count occurrences of each filename
file_count = Counter(all_files)

# Find files that appear in at least two folders
duplicate_files = [file for file, count in file_count.items() if count > 1]

print("Files found in at least two folders:")
print(duplicate_files)