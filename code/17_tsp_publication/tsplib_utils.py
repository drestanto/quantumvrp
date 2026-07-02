"""
tsplib_utils.py
Utilities for loading TSPLIB instances and generating small random TSP instances
in TSPLIB format, for use in the publication experiments.
"""

import math
import random
import itertools
import json


def load_tsplib_file(filepath):
    """
    Parse a TSPLIB .tsp file (EUC_2D or EXPLICIT FULL_MATRIX format).
    Returns: (name, n, distance_matrix) where distance_matrix[i][j] is the
    integer-rounded distance between city i and city j (0-indexed).
    """
    name = ""
    edge_weight_type = ""
    coords = {}
    matrix_rows = []
    n = 0

    with open(filepath, 'r') as f:
        lines = f.readlines()

    section = None
    for line in lines:
        line = line.strip()
        if not line or line == "EOF":
            continue

        if line.startswith("NAME"):
            name = line.split(":")[1].strip()
        elif line.startswith("DIMENSION"):
            n = int(line.split(":")[1].strip())
        elif line.startswith("EDGE_WEIGHT_TYPE"):
            edge_weight_type = line.split(":")[1].strip()
        elif line.startswith("NODE_COORD_SECTION"):
            section = "COORDS"
        elif line.startswith("EDGE_WEIGHT_SECTION"):
            section = "MATRIX"
        elif section == "COORDS":
            parts = line.split()
            if len(parts) >= 3:
                city_id = int(parts[0]) - 1  # 0-indexed
                x, y = float(parts[1]), float(parts[2])
                coords[city_id] = (x, y)
        elif section == "MATRIX":
            parts = line.split()
            matrix_rows.extend([float(v) for v in parts])

    # Build distance matrix
    dist = [[0.0] * n for _ in range(n)]

    if edge_weight_type == "EUC_2D" and coords:
        for i in range(n):
            for j in range(n):
                if i != j:
                    dx = coords[i][0] - coords[j][0]
                    dy = coords[i][1] - coords[j][1]
                    dist[i][j] = round(math.sqrt(dx*dx + dy*dy))

    elif section == "MATRIX" and matrix_rows:
        idx = 0
        for i in range(n):
            for j in range(n):
                dist[i][j] = matrix_rows[idx]
                idx += 1

    return name, n, dist


def generate_random_tsp(n, seed=None, scale=100.0):
    """
    Generate a random symmetric TSP instance with n cities placed uniformly
    in a [0, scale] x [0, scale] square. Uses Euclidean distances.

    Returns:
        name (str): instance name
        n (int): number of cities
        dist (list of list of float): n x n distance matrix
        coords (list of (x, y)): city coordinates
    """
    rng = random.Random(seed)
    coords = [(rng.uniform(0, scale), rng.uniform(0, scale)) for _ in range(n)]

    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                dx = coords[i][0] - coords[j][0]
                dy = coords[i][1] - coords[j][1]
                dist[i][j] = math.sqrt(dx*dx + dy*dy)

    name = f"random{n}_seed{seed if seed is not None else 'none'}"
    return name, n, dist, coords


def brute_force_tsp(dist, n):
    """
    Solve TSP exactly by enumerating all (n-1)! permutations (fix city 0 as start).
    Returns: (optimal_cost, optimal_tour)
    optimal_tour: list of city indices forming the tour (starting and ending at 0)
    """
    best_cost = float('inf')
    best_tour = None

    cities = list(range(1, n))
    for perm in itertools.permutations(cities):
        tour = [0] + list(perm) + [0]
        cost = sum(dist[tour[i]][tour[i+1]] for i in range(len(tour)-1))
        if cost < best_cost:
            best_cost = cost
            best_tour = tour

    return best_cost, best_tour


def load_casey_instance(json_path, cluster_type, target_id):
    """
    Load a City of Casey TSP instance from cluster_complete_graphs.json.
    The depot node is treated as city 0 (fixed start in TSP).

    Returns:
        name (str): instance identifier
        n (int): number of cities (including depot as city 0)
        dist (dict of dict): distance_matrix[i][j] in km
        city_ids (list): original node IDs (depot first)
        optimal_cost (float): brute-force optimal tour cost
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    if cluster_type == 'waste_facility':
        cluster_list = data.get('waste_areas_to_facilities_graphs', [])
        id_key = 'waste_facility_id'
    else:
        cluster_list = data.get('litter_bins_to_waste_areas_graphs', [])
        id_key = 'waste_collection_area_id'

    for item in cluster_list:
        if int(item.get(id_key)) == target_id:
            raw_dist = item.get('distance_matrix_km', {})
            dist = {}
            for r, row in raw_dist.items():
                dist[int(r)] = {int(c): v for c, v in row.items()}

            all_nodes = sorted(dist.keys())
            depot = target_id
            customers = [n for n in all_nodes if n != depot]
            city_ids = [depot] + customers
            n = len(city_ids)

            # Reindex to 0..n-1 for simplicity
            idx_map = {cid: i for i, cid in enumerate(city_ids)}
            dist_matrix = [[0.0]*n for _ in range(n)]
            for r in city_ids:
                for c in city_ids:
                    dist_matrix[idx_map[r]][idx_map[c]] = dist[r].get(c, 0.0)

            optimal_cost, _ = brute_force_tsp(dist_matrix, n)
            name = f"casey_{cluster_type}_{target_id}"
            return name, n, dist_matrix, city_ids, optimal_cost

    raise ValueError(f"Instance {cluster_type} ID {target_id} not found in {json_path}")


def get_all_casey_instance_ids(json_path):
    """
    Return all available (cluster_type, target_id) pairs in the JSON file.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    ids = []
    for item in data.get('waste_areas_to_facilities_graphs', []):
        ids.append(('waste_facility', int(item.get('waste_facility_id'))))
    for item in data.get('litter_bins_to_waste_areas_graphs', []):
        ids.append(('waste_collection_area', int(item.get('waste_collection_area_id'))))
    return ids


if __name__ == '__main__':
    # Quick test with a random instance
    name, n, dist, coords = generate_random_tsp(5, seed=42)
    print(f"Instance: {name}, cities: {n}")
    opt_cost, opt_tour = brute_force_tsp(dist, n)
    print(f"Optimal tour: {opt_tour}, cost: {opt_cost:.4f}")
