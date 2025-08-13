import json
import math
import time
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

JSON_FILE_PATH = 'cluster_complete_graphs.json'

# --- Variables you can edit ---
cluster_type = 'waste_facility'
target_id = 5
# ------------------------------

def solve_vrp_with_ortools(cluster_id, raw_distance_matrix, cluster_type_name):
    """
    Solves VRP for a cluster using Google OR-Tools instead of brute force.
    """
    start_time = time.perf_counter()

    # Convert keys to integers
    processed_distance_matrix = {}
    for r_key, row_val in raw_distance_matrix.items():
        processed_row = {}
        for c_key, dist_val in row_val.items():
            processed_row[int(c_key)] = dist_val if dist_val is not None else math.inf
        processed_distance_matrix[int(r_key)] = processed_row

    nodes = list(processed_distance_matrix.keys())
    depot = cluster_id
    num_vehicles = 1  # Same as your brute force — single route from depot
    manager = pywrapcp.RoutingIndexManager(len(nodes), num_vehicles, nodes.index(depot))
    routing = pywrapcp.RoutingModel(manager)

    # Distance callback
    def distance_callback(from_index, to_index):
        from_node = nodes[manager.IndexToNode(from_index)]
        to_node = nodes[manager.IndexToNode(to_index)]
        dist = processed_distance_matrix[from_node][to_node]
        return int(dist * 1000) if dist != math.inf else 10**9  # Convert km to meters for integer solver

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Optional: Add a large penalty for skipping nodes (forces visiting all if possible)
    penalty = 10**9
    for node_index in range(len(nodes)):
        if nodes[node_index] != depot:
            routing.AddDisjunction([manager.NodeToIndex(node_index)], penalty)

    # Search parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.seconds = 10  # Adjustable

    # Solve and time solver execution only
    solve_start = time.perf_counter()
    solution = routing.SolveWithParameters(search_parameters)
    solve_end = time.perf_counter()
    solve_time = solve_end - solve_start

    end_time = time.perf_counter()
    total_time = end_time - start_time

    print(f"\n--- {cluster_type_name} ID: {depot} ---")

    if not solution:
        print("No solution found.")
        print(f"Solver time: {solve_time:.4f} seconds")
        print(f"Total time: {total_time:.4f} seconds")
        return

    # Distance calculation timing
    dist_calc_start = time.perf_counter()

    index = routing.Start(0)
    route_nodes = []
    route_distance_m = 0
    while not routing.IsEnd(index):
        node = nodes[manager.IndexToNode(index)]
        route_nodes.append(node)
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        route_distance_m += routing.GetArcCostForVehicle(previous_index, index, 0)
    route_nodes.append(nodes[manager.IndexToNode(index)])  # return to depot

    dist_calc_end = time.perf_counter()
    dist_calc_time = dist_calc_end - dist_calc_start

    route_distance_km = route_distance_m / 1000.0
    formatted_path = " -> ".join(map(str, route_nodes))

    print(f"Optimal Path: {formatted_path}")
    print(f"Minimum Distance (OR-Tools) = {route_distance_km:.3f} km")
    print(f"Distance calculation time: {dist_calc_time:.4f} seconds")
    print(f"Solver time: {solve_time:.4f} seconds")
    print(f"Total function time: {total_time:.4f} seconds")

def main_single_cluster():
    try:
        with open(JSON_FILE_PATH, 'r') as f:
            all_cluster_graphs = json.load(f)

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
                solve_vrp_with_ortools(target_id, raw_distance_matrix, cluster_description)
            else:
                print(f"No distance matrix found for {cluster_description} ID {target_id}.")
        else:
            print(f"Cluster for {cluster_description} ID {target_id} not found.")

    except FileNotFoundError:
        print(f"Error: The file '{JSON_FILE_PATH}' was not found.")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{JSON_FILE_PATH}'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main_single_cluster()
