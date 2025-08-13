import pandas as pd
import numpy as np
import os

# --- Configuration and File Paths ---
DELIMITER = ';'
ROAD_NETWORK_SPATIAL_FILE = 'road-responsibility-clean-spatial.csv'

# Output directory and prefix
OUTPUT_DIR = '../working'
OUTPUT_PREFIX = 'road-network-part'
NUM_PARTS = 80 # Number of parts to split the data into

def load_data(file_path):
    """Loads data from a CSV file."""
    try:
        return pd.read_csv(file_path, delimiter=DELIMITER)
    except FileNotFoundError:
        print(f"Error: Required file not found: {file_path}. Aborting.")
        return None
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def split_and_save_road_data(road_df, output_dir, prefix, num_parts):
    """Splits a DataFrame into parts and saves them to CSV files."""
    print(f"Splitting the road network DataFrame into {num_parts} parts...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Use numpy's array_split to handle non-divisible numbers cleanly
    split_dfs = np.array_split(road_df, num_parts)
    
    for i, part_df in enumerate(split_dfs):
        output_file = os.path.join(output_dir, f"{prefix}{i + 1}.csv")
        part_df.to_csv(output_file, index=False, sep=DELIMITER)
        print(f"  - Saved part {i + 1}/{num_parts} with {len(part_df)} records to '{output_file}'")

    print("\nSplitting complete.")

if __name__ == "__main__":
    road_df = load_data(ROAD_NETWORK_SPATIAL_FILE)
    if road_df is not None:
        split_and_save_road_data(road_df, OUTPUT_DIR, OUTPUT_PREFIX, NUM_PARTS)