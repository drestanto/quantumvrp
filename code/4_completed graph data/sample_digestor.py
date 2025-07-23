import json
import pandas as pd

# Assume 'cluster_complete_graphs.json' is in the current directory
json_file_path = 'cluster_complete_graphs.json'

# --- Variables you can edit ---
# Set to 'waste_facility' to get graphs for Waste Facilities (clustering of Waste Areas to Facilities)
# Set to 'waste_collection_area' to get graphs for Waste Collection Areas (clustering of Litter Bins to Waste Areas)
cluster_type = 'waste_facility' 

# Set the ID of the specific cluster you want to digest
# E.g., if cluster_type is 'waste_facility', this should be a waste_facility_id
# E.g., if cluster_type is 'waste_collection_area', this should be a waste_collection_area_id
target_id = 5 
# ---------------------------

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
        print("Invalid cluster_type. Please set to 'waste_facility' or 'waste_collection_area'.")
        exit()

    found_cluster = None
    for cluster in cluster_list:
        if cluster.get(cluster_id_key) == target_id:
            found_cluster = cluster
            break

    if found_cluster:
        distance_matrix_dict = found_cluster.get('distance_matrix_km', {})
        
        if distance_matrix_dict:
            # Convert the dictionary of dictionaries to a pandas DataFrame
            # This automatically handles creating a matrix with IDs as labels
            matrix_df = pd.DataFrame(distance_matrix_dict)

            print(f"Distance Matrix for {cluster_description} ID {target_id}:\n")
            print(matrix_df)
            print("\n")

            # If you specifically need a NumPy array (values only, without row/col labels)
            numpy_matrix = matrix_df.values
            print("Numpy Array (values only):\n")
            print(numpy_matrix)

        else:
            print(f"No distance matrix found for {cluster_description} ID {target_id}.")
    else:
        print(f"Cluster for {cluster_description} ID {target_id} not found.")

except FileNotFoundError:
    print(f"Error: The file '{json_file_path}' was not found. Please ensure it's in the correct directory.")
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from '{json_file_path}'. Please check if the file is valid JSON.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")