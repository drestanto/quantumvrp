import json

# Assume 'cluster_complete_graphs.json' is in the current directory
json_file_path = 'cluster_complete_graphs.json'

try:
    with open(json_file_path, 'r') as f:
        all_cluster_graphs = json.load(f)

    print("--- Cluster Node Counts ---")

    # Lister for Public Litter Bins to Waste Collection Areas clusters
    litter_bins_to_waste_areas_graphs = all_cluster_graphs.get('litter_bins_to_waste_areas_graphs', [])
    if litter_bins_to_waste_areas_graphs:
        print("\nClusters for Public Litter Bins to Waste Collection Areas:")
        for cluster in litter_bins_to_waste_areas_graphs:
            cluster_id = cluster.get('waste_collection_area_id')
            distance_matrix = cluster.get('distance_matrix_km', {})
            num_nodes = len(distance_matrix)
            print(f"  Waste Collection Area ID: {cluster_id}, Nodes: {num_nodes}")
    else:
        print("\nNo Public Litter Bins to Waste Collection Areas clusters found.")

    # Lister for Waste Collection Areas to Waste Facilities clusters
    waste_areas_to_facilities_graphs = all_cluster_graphs.get('waste_areas_to_facilities_graphs', [])
    if waste_areas_to_facilities_graphs:
        print("\nClusters for Waste Collection Areas to Waste Facilities:")
        for cluster in waste_areas_to_facilities_graphs:
            cluster_id = cluster.get('waste_facility_id')
            distance_matrix = cluster.get('distance_matrix_km', {})
            num_nodes = len(distance_matrix)
            print(f"  Waste Facility ID: {cluster_id}, Nodes: {num_nodes}")
    else:
        print("\nNo Waste Collection Areas to Waste Facilities clusters found.")

except FileNotFoundError:
    print(f"Error: The file '{json_file_path}' was not found. Please ensure it's in the correct directory.")
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from '{json_file_path}'. Please check if the file is valid JSON.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")