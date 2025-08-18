import pandas as pd

# Load the CSV file
df = pd.read_csv("/Users/maksimkazanskii2/RNA/RNA/experiments/exp1/results/logs.csv")  # Replace with your actual filename

# Find the row with the minimum Validation MAE
min_val_mae_row = df.loc[df["Validation MAE"].idxmin()]

# Print the entire row
print("Row with minimum Validation MAE:")
print(min_val_mae_row)