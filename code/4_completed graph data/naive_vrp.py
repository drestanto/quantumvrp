import json
import itertools
import math
import time

# Assume 'cluster_complete_graphs.json' is in the current directory
JSON_FILE_PATH = 'cluster_complete_graphs.json'

# --- Variables you can edit ---
cluster_type = 'waste_facility' 
target_id = 5 
# ---------------------------

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

def solve_vrp_for_cluster(cluster_id, raw_distance_matrix, cluster_type_name):
    """
    Solves the VRP for a single cluster using a brute-force (permutations) approach.
    The cluster_id is considered the depot.
    Calculates min, max, average distances, and savings.
    """
    start_time = time.perf_counter() # Start timing for processing

    processed_distance_matrix = {}
    for r_key, row_val in raw_distance_matrix.items():
        processed_row = {}
        for c_key, dist_val in row_val.items():
            processed_row[int(c_key)] = dist_val 
        processed_distance_matrix[int(r_key)] = processed_row 

    nodes = list(processed_distance_matrix.keys())
    
    depot = cluster_id 
    customers = [node for node in nodes if node != depot]

    print(f"\n--- {cluster_type_name} ID: {depot} ---")

    if not customers:
        print("No customers in this cluster to form a route (only depot).")
        print(f"Time processed: {0:.4f} seconds")
        return

    min_total_distance = math.inf
    optimal_path = []
    
    all_path_distances = [] # To store distances of all reachable permutations

    # Iterate through all permutations of customers
    for permutation in itertools.permutations(customers):
        current_path_nodes = [depot] + list(permutation) + [depot]
        current_path_distance = calculate_path_distance(current_path_nodes, processed_distance_matrix)

        if not math.isinf(current_path_distance):
            all_path_distances.append(current_path_distance) # Collect reachable distances
            if current_path_distance < min_total_distance:
                min_total_distance = current_path_distance
                optimal_path = current_path_nodes

    end_time = time.perf_counter() # End timing for processing
    time_processed = end_time - start_time

    if not all_path_distances:
        print("No reachable path found for this cluster (all permutations resulted in infinite distance).")
    else:
        # Calculate statistics from reachable paths
        max_total_distance = max(all_path_distances)
        average_total_distance = sum(all_path_distances) / len(all_path_distances)
        
        potential_distance_saving = max_total_distance - min_total_distance
        average_distance_saving = average_total_distance - min_total_distance

        formatted_path = " -> ".join(map(str, optimal_path))
        print(f"Optimal Path: {formatted_path}")
        print(f"Minimum Distance = {min_total_distance:.3f} km")
        print(f"Worst Distance (Maximum) = {max_total_distance:.3f} km")
        print(f"Average Distance = {average_total_distance:.3f} km")
        print(f"Potential Distance Saving (Max - Min) = {potential_distance_saving:.3f} km")
        print(f"Average Distance Saving (Avg - Min) = {average_distance_saving:.3f} km")
    
    print(f"Time processed: {time_processed:.4f} seconds")

def main_single_cluster():
    """
    Main function to solve VRP for a single, user-defined cluster.
    """
    try:
        with open(JSON_FILE_PATH, 'r') as f:
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
            return

        found_cluster_data = None
        for cluster_data in cluster_list:
            if int(cluster_data.get(cluster_id_key)) == target_id:
                found_cluster_data = cluster_data
                break
        
        if found_cluster_data:
            raw_distance_matrix = found_cluster_data.get('distance_matrix_km', {})
            if raw_distance_matrix:
                solve_vrp_for_cluster(target_id, raw_distance_matrix, cluster_description)
            else:
                print(f"No distance matrix found for {cluster_description} ID {target_id}.")
        else:
            print(f"Cluster for {cluster_description} ID {target_id} not found.")

    except FileNotFoundError:
        print(f"Error: The file '{JSON_FILE_PATH}' was not found. Please ensure it's in the same directory.")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{JSON_FILE_PATH}'. Please check if the file is valid JSON.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main_single_cluster()