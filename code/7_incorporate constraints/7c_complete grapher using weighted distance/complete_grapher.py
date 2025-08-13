import pandas as pd
import numpy as np
import json
import networkx as nx
from scipy.spatial import KDTree
import os

# --- Configuration and File Paths ---
DELIMITER = ';'

# Input spatial CSVs (from isolate.py)
PUBLIC_LITTER_BINS_SPATIAL_FILE = 'public-litter-bins-clean-spatial.csv'
WASTE_COLLECTION_AREAS_SPATIAL_FILE = 'waste-collection-area-clean-spatial.csv'
WASTE_FACILITY_LOCATIONS_SPATIAL_FILE = 'waste-facility-locations-clean-spatial.csv'
ROAD_NETWORK_SPATIAL_FILE = 'combined-preprocessed-road-network.csv'

# Input clustering JSONs (from previous step)
LITTER_BINS_CLUSTERING_JSON = 'public_litter_bins_to_waste_collection_areas_clustering.json'
WASTE_AREAS_CLUSTERING_JSON = 'waste_collection_areas_to_waste_facilities_clustering.json'

# Output JSON file for complete graphs
OUTPUT_GRAPHS_JSON = 'cluster_complete_graphs_with_constraints.json'

# --- Helper Functions ---
def load_data(file_path, is_json=False):
    """Loads data from a CSV or JSON file."""
    try:
        if is_json:
            with open(file_path, 'r') as f:
                return json.load(f)
        else:
            return pd.read_csv(file_path, delimiter=DELIMITER)
    except FileNotFoundError:
        print(f"Error: Required file not found: {file_path}. Please ensure all input files exist.")
        return None
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def build_road_graph(road_df):
    """
    Builds a NetworkX graph from road segments using the pre-calculated weights.
    Nodes are unique (lat, lon) points, edges are road segments with
    the 'modified_weight_km' column as the edge weight.
    """
    print("Building road network graph with pre-calculated edge weights...")
    G = nx.Graph()
    node_coords = {} # Maps node ID to (lat, lon)
    coord_to_node_id = {} # Maps (lat, lon) tuple to node ID
    node_id_counter = 0

    # Add nodes and edges
    for _, row in road_df.iterrows():
        start_coord = (row['START_LAT'], row['START_LON'])
        end_coord = (row['END_LAT'], row['END_LON'])
        weight = row['modified_weight_km']

        # Get/create node IDs for start and end coordinates
        if start_coord not in coord_to_node_id:
            coord_to_node_id[start_coord] = node_id_counter
            node_coords[node_id_counter] = start_coord
            node_id_counter += 1
        start_node_id = coord_to_node_id[start_coord]

        if end_coord not in coord_to_node_id:
            coord_to_node_id[end_coord] = node_id_counter
            node_coords[node_id_counter] = end_coord
            node_id_counter += 1
        end_node_id = coord_to_node_id[end_coord]

        if not G.has_edge(start_node_id, end_node_id): # Avoid adding duplicate edges if they exist
            G.add_edge(start_node_id, end_node_id, weight=weight)

    print(f"Road network graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges (weighted).")
    return G, node_coords, coord_to_node_id

def map_points_to_nearest_road_node(points_df, node_coords):
    """
    Maps each point in points_df to its nearest node in the road network.
    Returns a dictionary: {original_point_id: closest_road_node_id}
    """
    print("Mapping points to nearest road network nodes...")
    if points_df.empty or not node_coords:
        return {}

    # Prepare road network nodes for KDTree
    road_node_ids = list(node_coords.keys())
    road_node_latlons = np.array([node_coords[nid] for nid in road_node_ids])

    # Create KDTree for efficient nearest neighbor search
    kdtree = KDTree(road_node_latlons)

    point_to_node_mapping = {}
    for _, row in points_df.iterrows():
        point_latlon = np.array([row['Latitude'], row['Longitude']])

        # Query KDTree for the nearest road node
        # k=1 for single nearest neighbor
        distance, index = kdtree.query(point_latlon, k=1)

        if index < len(road_node_ids): # Ensure a valid index was returned
            closest_road_node_id = road_node_ids[index]
            point_to_node_mapping[int(row['ID'])] = closest_road_node_id # Cast ID to int here
        else:
            print(f"Warning: Could not find a closest road node for point ID {row['ID']} ({row['Latitude']}, {row['Longitude']}). This point will be skipped in graph calculations.")

    print(f"Mapped {len(point_to_node_mapping)} points to road network nodes.")
    return point_to_node_mapping

def calculate_complete_graph_distances(cluster_points_data, road_graph, point_to_node_mapping):
    """
    Calculates all-pairs shortest path distances within a cluster using the road network graph.
    Returns a distance matrix dictionary.
    """
    cluster_node_ids = []
    cluster_original_ids = []

    # Collect road network node IDs for all points in the current cluster
    for original_id in cluster_points_data:
        # Ensure ID is an integer
        int_original_id = int(original_id)
        if int_original_id in point_to_node_mapping:
            cluster_node_ids.append(point_to_node_mapping[int_original_id])
            cluster_original_ids.append(int_original_id) # Store as int
        else:
            print(f"  Skipping original ID {int_original_id} as it was not mapped to a road node.")

    if len(cluster_node_ids) < 2:
        print("  Cluster has fewer than 2 mappable points. Skipping distance matrix calculation.")
        return {}

    distance_matrix = {}
    for i, source_original_id in enumerate(cluster_original_ids):
        source_node_id = cluster_node_ids[i]
        distance_matrix[source_original_id] = {} # Key as int

        if source_node_id not in road_graph:
            print(f"  Warning: Source node {source_node_id} (for original ID {source_original_id}) not found in road graph. Skipping paths from this point.")
            for target_original_id in cluster_original_ids:
                distance_matrix[source_original_id][target_original_id] = None # Mark as unreachable
            continue

        try:
            paths_from_source = nx.shortest_path_length(road_graph, source=source_node_id, weight='weight')

            for j, target_original_id in enumerate(cluster_original_ids):
                target_node_id = cluster_node_ids[j]
                if source_original_id == target_original_id:
                    distance_matrix[source_original_id][target_original_id] = 0.0
                elif target_node_id in paths_from_source:
                    distance_matrix[source_original_id][target_original_id] = paths_from_source[target_node_id]
                else:
                    distance_matrix[source_original_id][target_original_id] = None # Not reachable
        except Exception as e:
            print(f"  Error calculating paths from {source_original_id} (node {source_node_id}): {e}")
            for target_original_id in cluster_original_ids:
                distance_matrix[source_original_id][target_original_id] = None

    return distance_matrix


# --- Main Execution ---
def main():
    # 1. Load all necessary data
    print("Loading input data...")
    bins_df = load_data(PUBLIC_LITTER_BINS_SPATIAL_FILE)
    waste_areas_df = load_data(WASTE_COLLECTION_AREAS_SPATIAL_FILE)
    facilities_df = load_data(WASTE_FACILITY_LOCATIONS_SPATIAL_FILE)
    road_df = load_data(ROAD_NETWORK_SPATIAL_FILE)

    litter_bins_clustering = load_data(LITTER_BINS_CLUSTERING_JSON, is_json=True)
    waste_areas_clustering = load_data(WASTE_AREAS_CLUSTERING_JSON, is_json=True)

    # Check for critical file loading errors
    if any(df is None for df in [bins_df, waste_areas_df, facilities_df, road_df]) or \
       any(json_data is None for json_data in [litter_bins_clustering, waste_areas_clustering]):
        print("Aborting graph creation due to missing or unreadable critical input files.")
        return

    # 2. Build the Road Network Graph with pre-calculated weights
    road_graph, node_coords, coord_to_node_id = build_road_graph(road_df)

    # 3. Map all relevant points to the nearest road network node
    all_points_df = pd.concat([bins_df, waste_areas_df, facilities_df], ignore_index=True)
    point_to_road_node_mapping = map_points_to_nearest_road_node(all_points_df, node_coords)

    # Dictionary to store all cluster graphs
    all_cluster_graphs = {
        "litter_bins_to_waste_areas_graphs": [],
        "waste_areas_to_facilities_graphs": []
    }

    # 4. Calculate Complete Graphs for Public Litter Bins to Waste Collection Areas Clusters
    print("\nCalculating complete graphs for Public Litter Bins to Waste Collection Areas clusters...")
    if litter_bins_clustering and 'assignments' in litter_bins_clustering:
        for cluster_assignment in litter_bins_clustering['assignments']:
            # Ensure ID is an integer
            waste_area_id = int(cluster_assignment['waste_collection_area_id'])
            # Ensure bin IDs are integers
            public_litter_bin_ids = [int(bid) for bid in cluster_assignment['public_litter_bin_ids']]

            # Combine waste area ID and bin IDs for this cluster
            cluster_points_ids = [waste_area_id] + public_litter_bin_ids

            print(f"  Processing cluster for Waste Collection Area ID: {waste_area_id} with {len(public_litter_bin_ids)} bins.")

            distance_matrix = calculate_complete_graph_distances(
                cluster_points_ids, road_graph, point_to_road_node_mapping
            )

            all_cluster_graphs["litter_bins_to_waste_areas_graphs"].append({
                "waste_collection_area_id": waste_area_id, # Stored as int
                "distance_matrix_km": distance_matrix
            })
    else:
        print("No Public Litter Bins clustering assignments found or JSON format is incorrect.")


    # 5. Calculate Complete Graphs for Waste Collection Areas to Waste Facilities Clusters
    print("\nCalculating complete graphs for Waste Collection Areas to Waste Facilities clusters...")
    if waste_areas_clustering and 'assignments' in waste_areas_clustering:
        for cluster_assignment in waste_areas_clustering['assignments']:
            # Ensure ID is an integer
            waste_facility_id = int(cluster_assignment['waste_facility_id'])
            # Ensure waste area IDs are integers
            waste_collection_area_ids = [int(aid) for aid in cluster_assignment['waste_collection_area_ids']]

            # Combine facility ID and waste area IDs for this cluster
            cluster_points_ids = [waste_facility_id] + waste_collection_area_ids

            print(f"  Processing cluster for Waste Facility ID: {waste_facility_id} with {len(waste_collection_area_ids)} waste areas.")

            distance_matrix = calculate_complete_graph_distances(
                cluster_points_ids, road_graph, point_to_road_node_mapping
            )

            all_cluster_graphs["waste_areas_to_facilities_graphs"].append({
                "waste_facility_id": waste_facility_id, # Stored as int
                "distance_matrix_km": distance_matrix
            })
    else:
        print("No Waste Collection Areas clustering assignments found or JSON format is incorrect.")

    # Save the final output
    with open(OUTPUT_GRAPHS_JSON, 'w') as f:
        json.dump(all_cluster_graphs, f, indent=4)
    print(f"\nAll complete graphs saved to '{OUTPUT_GRAPHS_JSON}'")

if __name__ == "__main__":
    main()