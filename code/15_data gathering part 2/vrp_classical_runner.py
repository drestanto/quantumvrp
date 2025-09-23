import json
import itertools
import math
import time
import csv
import os
from datetime import datetime
import sys

# Assume 'cluster_complete_graphs.json' is in the current directory
JSON_FILE_PATH = 'cluster_complete_graphs.json'
CLASSICAL_OUT_CSV_FILE_PATH = 'out_classical.csv' # Output CSV for classical benchmarks
DATA_ENTRY_CLASSICAL_FILE_PATH = 'data_entries_classical.csv' # Input configuration CSV

# --- Utility to redirect stdout (copied from vrp_quantum_runner.py) ---
class Tee:
    def __init__(self, filename, mode='a'):
        self.file = open(filename, mode)
        self.stdout = sys.stdout

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def close(self):
        self.file.close()
        sys.stdout = self.stdout # Restore original stdout

def load_vrp_data(json_file_path, cluster_type, target_id):
    """
    Loads the VRP distance matrix for a specific cluster from a JSON file.
    (Copied from vrp_quantum_runner.py to maintain consistency)
    """
    print(f"\nSTEP 0: Loading VRP data for {cluster_type} ID {target_id}...")
    try:
        with open(json_file_path, 'r') as f:
            all_cluster_graphs = json.load(f)

        cluster_list = []
        cluster_id_key = ""
        cluster_description = ""

        if cluster_type == 'waste_facility':
            cluster_list = all_cluster_graphs.get('waste_areas_to_facilities_graphs', [])
            cluster_id_key = 'waste_facility_id'
            cluster_description = "Waste Facility"
        elif cluster_type == 'waste_collection_area':
            cluster_list = all_cluster_graphs.get('litter_bins_to_waste_areas_graphs', [])
            cluster_id_key = 'waste_collection_area_id'
            cluster_description = "Waste Collection Area"
        else:
            print("Invalid 'cluster_type'. Please set to 'waste_facility' or 'waste_collection_area'.")
            return None, None, None, None

        found_cluster_data = None
        for cluster_data in cluster_list:
            if int(cluster_data.get(cluster_id_key)) == target_id:
                found_cluster_data = cluster_data
                break

        if found_cluster_data:
            raw_distance_matrix = found_cluster_data.get('distance_matrix_km', {})
            if raw_distance_matrix:
                processed_distance_matrix = {}
                for r_key, row_val in raw_distance_matrix.items():
                    processed_row = {}
                    for c_key, dist_val in row_val.items():
                        processed_row[int(c_key)] = dist_val
                    processed_distance_matrix[int(r_key)] = processed_row
                
                nodes = sorted(list(processed_distance_matrix.keys()))
                depot_node = target_id
                customer_nodes = [node for node in nodes if node != depot_node]

                print(f"VRP data loaded successfully for {cluster_description} ID {target_id}.")
                print(f"Nodes: {nodes}")
                print(f"Depot: {depot_node}")
                print(f"Customers: {customer_nodes}")
                return processed_distance_matrix, nodes, depot_node, customer_nodes
            else:
                print(f"No distance matrix found for {cluster_description} ID {target_id}.")
        else:
            print(f"Cluster for {cluster_description} ID {target_id} not found.")

    except FileNotFoundError:
        print(f"Error: The file '{json_file_path}' was not found. Please ensure it's in the same directory.")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{json_file_path}'. Please check if the file is valid JSON.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return None, None, None, None

def calculate_path_distance(path, distance_matrix):
    """
    Calculates the total distance of a given path using the provided distance matrix.
    Returns float('inf') if any segment has a None distance.
    Assumes distance_matrix keys are integers.
    """
    total_distance = 0.0
    for i in range(len(path) - 1):
        from_node = path[i]
        to_node = path[i+1]
        
        if from_node not in distance_matrix or to_node not in distance_matrix[from_node]:
            return math.inf 
        
        segment_distance = distance_matrix[from_node][to_node]
        
        if segment_distance is None:
            return math.inf 
        
        total_distance += segment_distance
    return total_distance

def run_classical_vrp_experiment(cluster_type_param, target_id_param):
    """
    Solves the VRP for a single cluster using a brute-force (permutations) approach.
    The target_id is considered the depot.
    Calculates min, max, average distances, and writes to CSV.
    """
    print(f"\n--- Running Classical Experiment: Cluster={cluster_type_param}, ID={target_id_param} ---")
    start_time = time.perf_counter() # Start timing for processing
    current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    distance_matrix, all_nodes, depot, customers = load_vrp_data(JSON_FILE_PATH, cluster_type_param, target_id_param)

    if distance_matrix is None:
        print("Failed to load VRP data for classical run. Skipping this experiment.")
        return False, None, None, None # Indicate failure

    num_nodes = len(all_nodes) # Get the total number of nodes (depot + customers)
    print(f"Total number of nodes (including depot): {num_nodes}")

    if not customers:
        print("No customers in this cluster to form a route (only depot).")
        print(f"Time processed: {0:.4f} seconds")
        # For a single depot, min/max/avg cost is 0.0, as no travel is needed beyond depot.
        return True, 0.0, 0.0, 0.0, num_nodes # Return num_nodes even if no customers

    min_total_distance = math.inf
    max_total_distance = 0.0
    total_distances_sum = 0.0
    num_reachable_paths = 0
    
    # Iterate through all permutations of customers
    for permutation in itertools.permutations(customers):
        current_path_nodes = [depot] + list(permutation) + [depot]
        current_path_distance = calculate_path_distance(current_path_nodes, distance_matrix)

        if not math.isinf(current_path_distance):
            num_reachable_paths += 1
            total_distances_sum += current_path_distance
            
            if current_path_distance < min_total_distance:
                min_total_distance = current_path_distance
            if current_path_distance > max_total_distance:
                max_total_distance = current_path_distance

    end_time = time.perf_counter() # End timing for processing
    time_processed = end_time - start_time

    if num_reachable_paths == 0:
        print("No reachable path found for this cluster (all permutations resulted in infinite distance).")
        print(f"Time processed: {time_processed:.4f} seconds")
        return False, None, None, None, num_nodes # Indicate failure, but still return num_nodes
    else:
        average_total_distance = total_distances_sum / num_reachable_paths
        
        print(f"Minimum Distance = {min_total_distance:.3f} km")
        print(f"Worst Distance (Maximum) = {max_total_distance:.3f} km")
        print(f"Average Distance = {average_total_distance:.3f} km")
        print(f"Time processed: {time_processed:.4f} seconds")

        # Prepare data for CSV
        row_data = [
            cluster_type_param,
            target_id_param,
            num_nodes,  # Added number of nodes
            min_total_distance,
            average_total_distance,
            max_total_distance,
            current_timestamp,
            time_processed
        ]
        
        # Append to the main classical output CSV file
        file_exists = os.path.exists(CLASSICAL_OUT_CSV_FILE_PATH)
        with open(CLASSICAL_OUT_CSV_FILE_PATH, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                writer.writerow([
                    "conf_cluster_type", 
                    "conf_target_id", 
                    "num_nodes", # Added header for num_nodes
                    "benchmark_min_cost", 
                    "benchmark_avg_cost", 
                    "benchmark_max_cost", 
                    "timestamp",
                    "time_processed_seconds"
                ])
            writer.writerow(row_data)

        print(f"\nClassical benchmark results for this run have been appended to '{CLASSICAL_OUT_CSV_FILE_PATH}'")
        print("Classical experiment finished successfully!")
        return True, min_total_distance, average_total_distance, max_total_distance, num_nodes # Indicate success

def main():
    print("Starting Classical VRP Runner script with batch processing...")

    # Ensure data_entries_classical.csv exists with header if not present
    if not os.path.exists(DATA_ENTRY_CLASSICAL_FILE_PATH):
        with open(DATA_ENTRY_CLASSICAL_FILE_PATH, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['conf_cluster_type', 'conf_target_id', 'done'])
        print(f"Created empty '{DATA_ENTRY_CLASSICAL_FILE_PATH}'. Please populate it with experiment configurations.")
        return

    # Read all data entries
    data_entries = []
    try:
        with open(DATA_ENTRY_CLASSICAL_FILE_PATH, 'r', newline='') as f:
            reader = csv.DictReader(f)
            data_entries = list(reader)
    except FileNotFoundError:
        print(f"Error: '{DATA_ENTRY_CLASSICAL_FILE_PATH}' not found. Please create it.")
        return
    except Exception as e:
        print(f"Error reading '{DATA_ENTRY_CLASSICAL_FILE_PATH}': {e}")
        return

    if not data_entries:
        print(f"'{DATA_ENTRY_CLASSICAL_FILE_PATH}' is empty. No classical experiments to run.")
        return

    processed_any_entry = False

    for i, entry in enumerate(data_entries):
        if entry.get('done', 'no').lower() == 'no':
            processed_any_entry = True
            cluster_type = entry['conf_cluster_type']
            target_id = int(entry['conf_target_id'])

            # Generate a consistent log file name
            log_filename_base = (
                f"classical_{cluster_type}_{target_id}"
                .replace(" ", "_").replace("/", "-")
            )
            log_filename = f"{log_filename_base}.out"

            # Redirect stdout to a file
            original_stdout = sys.stdout
            try:
                sys.stdout = Tee(log_filename, mode='w') # Overwrite each time for a new log
                print(f"Log started for classical experiment: {cluster_type}, {target_id}")

                # Run the classical experiment
                success, min_cost, avg_cost, max_cost, num_nodes_processed = run_classical_vrp_experiment(cluster_type, target_id) # Capture num_nodes

                if success:
                    # Update 'done' status in memory
                    data_entries[i]['done'] = 'yes'
                    print(f"Classical experiment for entry {i+1} marked as 'done'.")
                else:
                    print(f"Classical experiment for entry {i+1} failed. 'done' status remains 'no'.")

            except Exception as e:
                print(f"An unexpected error occurred during classical experiment for entry {i+1}: {e}")
                print(f"Check '{log_filename}' for details. 'done' status remains 'no'.")
            finally:
                if isinstance(sys.stdout, Tee):
                    sys.stdout.close() # This restores sys.stdout
                else:
                    sys.stdout = original_stdout # Fallback if Tee wasn't set up
                print(f"Output for this classical run also saved to '{log_filename}'") # This prints to original stdout after redirection is off

            # Write updated data_entries back to DATA_ENTRY_CLASSICAL_FILE_PATH
            try:
                with open(DATA_ENTRY_CLASSICAL_FILE_PATH, 'w', newline='') as f:
                    fieldnames = data_entries[0].keys() if data_entries else ['conf_cluster_type', 'conf_target_id', 'done']
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(data_entries)
                print(f"'{DATA_ENTRY_CLASSICAL_FILE_PATH}' updated.")
            except Exception as e:
                print(f"CRITICAL ERROR: Could not write back to '{DATA_ENTRY_CLASSICAL_FILE_PATH}': {e}")
                print("Manual intervention may be required to prevent data loss.")
        else:
            print(f"Skipping classical entry {i+1}: Already marked as 'done'.")

    if not processed_any_entry:
        print("\nNo pending classical experiments found in 'data_entries_classical.csv'. All 'done'.")
    else:
        print("\nAll pending classical experiments processed (or skipped due to errors). Script finished.")

if __name__ == "__main__":
    main()
