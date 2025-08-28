import json
import itertools
import math
import time
from collections import Counter
import csv
import os
from datetime import datetime

import numpy as np
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import minimize

# --- Unified Configuration ---
# File paths
JSON_FILE_PATH = 'cluster_complete_graphs.json'
CSV_FILE_PATH = 'out.csv'

# Method and VRP configuration
# Choose 'node_visit_time' or 'naive_adjacency'
METHOD = 'node_visit_time' # Change this to switch methods
cluster_type = 'waste_facility'
target_id = 5

# QAOA configuration
QAOA_P = 3  # Number of QAOA layers (depth)
SHOTS = 2048 # Number of shots for the sampler
# ---------------------------

# --- Penalty Weights (common to both methods but with different interpretations) ---
# For 'node_visit_time'
PENALTY_DUPLICATE_VISIT = 1000.0
PENALTY_OUT_OF_RANGE = 1000.0

# For 'naive_adjacency' (QUBO)
QUBO_PENALTY_A = 1000.0
QUBO_WEIGHT_B = 1.0
# ----------------------------------------------------------------------------------

def load_vrp_data(json_file_path, cluster_type, target_id):
    """
    Loads the VRP distance matrix for a specific cluster from a JSON file.
    (Function from both original scripts, kept for clarity)
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

def format_measurement_counts(bit_array_data):
    """
    Takes a BitArray object from Sampler results and returns a dictionary
    with bitstring keys formatted to the correct number of qubits.
    (Function from both original scripts, kept for clarity)
    """
    if hasattr(bit_array_data, 'get_counts'):
        counts = bit_array_data.get_counts()
        return counts
    else:
        raise RuntimeError("The BitArray object from Sampler results does not have a 'get_counts()' method.")

# --- Functions for 'node_visit_time' method ---

def decode_bitstring_to_route_nvt(bitstring, num_customers, bits_per_node, depot_node, customer_nodes):
    """
    Decodes a bitstring into a VRP route based on node-visit-time encoding.
    """
    decoded_visit_times = []
    sorted_customer_nodes = sorted(customer_nodes)
    for i in range(num_customers):
        start_idx = i * bits_per_node
        end_idx = start_idx + bits_per_node
        customer_bits = bitstring[start_idx:end_idx]
        try:
            visit_time = int(customer_bits, 2)
        except ValueError:
            visit_time = -1
        decoded_visit_times.append(visit_time)
    
    customer_visits = []
    for i, time_val in enumerate(decoded_visit_times):
        customer_visits.append((time_val, sorted_customer_nodes[i]))
    customer_visits.sort()
    ordered_customer_nodes = [node_id for time_val, node_id in customer_visits]
    route = [depot_node] + ordered_customer_nodes + [depot_node]
    return route, decoded_visit_times

def calculate_route_cost_nvt(route, distance_matrix):
    """
    Calculates the total distance of a given route.
    """
    total_distance = 0.0
    for i in range(len(route) - 1):
        from_node = route[i]
        to_node = route[i+1]
        if from_node not in distance_matrix or to_node not in distance_matrix[from_node]:
            return math.inf
        segment_distance = distance_matrix[from_node][to_node]
        if segment_distance is None:
            return math.inf
        total_distance += segment_distance
    return total_distance

def penalize_duplicates_and_out_of_range_nvt(decoded_visit_times, total_nodes):
    """
    Calculates penalties for duplicate visit times and out-of-range visit times.
    """
    penalty = 0.0
    valid_range_min = 0
    valid_range_max = total_nodes - 1
    for vt in decoded_visit_times:
        if not (valid_range_min <= vt <= valid_range_max):
            penalty += PENALTY_OUT_OF_RANGE
    counts = Counter(decoded_visit_times)
    for vt, count in counts.items():
        if count > 1:
            penalty += PENALTY_DUPLICATE_VISIT * (count - 1)
    return penalty

def total_cost_from_bitstring_nvt(bitstring, distance_matrix, num_customers, bits_per_node, depot_node, customer_nodes, total_nodes):
    """
    Calculates the total cost (distance + penalties) for a given bitstring.
    """
    route, decoded_visit_times = decode_bitstring_to_route_nvt(bitstring, num_customers, bits_per_node, depot_node, customer_nodes)
    distance_cost = calculate_route_cost_nvt(route, distance_matrix)
    penalty_cost = penalize_duplicates_and_out_of_range_nvt(decoded_visit_times, total_nodes)
    return distance_cost + penalty_cost

def create_qaoa_circuit(num_qubits, p, gamma, beta):
    """
    Creates a simplified QAOA circuit using RZ and RZZ for the cost layer.
    """
    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.h(range(num_qubits))
    qc.barrier()
    for k in range(p):
        for i in range(num_qubits):
            qc.rz(2 * gamma[k], i)
        for i in range(num_qubits):
            for j in range(i + 1, num_qubits):
                qc.rzz(2 * gamma[k], i, j)
        qc.barrier()
        for i in range(num_qubits):
            qc.rx(2 * beta[k], i)
        qc.barrier()
    qc.measure(range(num_qubits), range(num_qubits))
    return qc

def calculate_classical_benchmarks_nvt(distance_matrix, depot_node, customer_nodes):
    """
    Calculates classical benchmarks for the node-visit-time method.
    """
    print("\nCalculating classical benchmark costs (brute-force)...")
    costs = []
    all_customer_permutations = list(itertools.permutations(customer_nodes))
    for perm in all_customer_permutations:
        route = [depot_node] + list(perm) + [depot_node]
        cost = calculate_route_cost_nvt(route, distance_matrix)
        if cost != math.inf:
            costs.append(cost)
    if not costs:
        return float('inf'), 0.0, float('inf')
    min_cost = min(costs)
    max_cost = max(costs)
    avg_cost = sum(costs) / len(costs)
    print(f"Classical minimum cost found: {min_cost:.3f}")
    return min_cost, avg_cost, max_cost

# --- Functions for 'naive_adjacency' (QUBO) method ---

def get_num_qubits_qubo(num_cities):
    """Returns number of qubits = N^2 for N cities."""
    return num_cities * num_cities

def decode_bitstring_qubo(bitstring, num_cities):
    """
    Decode bitstring of length N^2 into a route.
    """
    if len(bitstring) != num_cities * num_cities:
        return [-1] * num_cities
    x = np.array(list(map(int, bitstring))).reshape((num_cities, num_cities))
    route = []
    for p in range(num_cities):
        city_indices_at_pos_p = np.where(x[:, p] == 1)[0]
        if len(city_indices_at_pos_p) == 1:
            route.append(int(city_indices_at_pos_p[0]))
        else:
            route.append(-1)
    return route

def calculate_cost_qubo(route, distance_matrix, all_nodes, A=QUBO_PENALTY_A, B=QUBO_WEIGHT_B):
    """
    Calculate the QUBO cost for a given route and distance matrix.
    """
    N = len(all_nodes)
    penalty = 0
    if len(route) != N:
        penalty += A * abs(len(route) - N)
    city_counts = Counter(route)
    for city_idx in all_nodes:
        count = city_counts.get(city_idx, 0)
        if count != 1:
            penalty += A * abs(count - 1)
    if -1 in city_counts:
        penalty += A * city_counts[-1]
    
    travel_cost = 0
    if -1 not in route and len(route) == N:
        full_path_for_cost = route + [route[0]]
        for i in range(len(full_path_for_cost) - 1):
            from_city = full_path_for_cost[i]
            to_city = full_path_for_cost[i+1]
            dist = distance_matrix.get(from_city, {}).get(to_city, None)
            if dist is None:
                penalty += A
            else:
                travel_cost += dist
    else:
        travel_cost = 0
        penalty += A * N
    
    total_cost = penalty + B * travel_cost
    return total_cost

def create_qaoa_circuit_qubo(num_cities, p, gamma, beta):
    """
    Build QAOA circuit for a QUBO formulation with N^2 qubits.
    """
    num_qubits = get_num_qubits_qubo(num_cities)
    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.h(range(num_qubits))
    qc.barrier()
    for layer in range(p):
        for q in range(num_qubits):
            qc.rz(2 * gamma[layer], q)
        for i in range(num_qubits):
            for j in range(i + 1, num_qubits):
                qc.rzz(2 * gamma[layer], i, j)
        qc.barrier()
        for i in range(num_qubits):
            qc.rx(2 * beta[layer], i)
        qc.barrier()
    qc.measure(range(num_qubits), range(num_qubits))
    return qc

def calculate_classical_benchmarks_qubo(distance_matrix, all_nodes, A=QUBO_PENALTY_A, B=QUBO_WEIGHT_B):
    """
    Calculates the minimum, maximum, and average cost for the QUBO formulation.
    """
    print("\nCalculating classical benchmark costs (brute-force for QUBO)...")
    costs = []
    all_city_permutations = list(itertools.permutations(all_nodes))
    for perm_route in all_city_permutations:
        cost = calculate_cost_qubo(list(perm_route), distance_matrix, all_nodes, A, B)
        if cost != math.inf:
            costs.append(cost)
    if not costs:
        return float('inf'), 0.0, float('inf')
    min_cost = min(costs)
    max_cost = max(costs)
    avg_cost = sum(costs) / len(costs)
    print(f"Classical minimum cost found: {min_cost:.3f}")
    return min_cost, avg_cost, max_cost

def convert_permutation_to_qubo_bitstring(permutation_route, all_nodes):
    """
    Converts a permutation of city IDs into its corresponding QUBO bitstring.
    """
    N = len(all_nodes)
    x_matrix = np.zeros((N, N), dtype=int)
    node_to_idx = {node: i for i, node in enumerate(all_nodes)}
    for position, city_id in enumerate(permutation_route):
        if city_id in node_to_idx:
            city_idx = node_to_idx[city_id]
            x_matrix[city_idx][position] = 1
    bitstring = "".join(map(str, x_matrix.flatten()))
    return bitstring

def main():
    print(f"Starting QAOA for VRP script with '{METHOD}' method...")
    current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # STEP 0: Load VRP Data
    distance_matrix, all_nodes, depot_node, customer_nodes = load_vrp_data(JSON_FILE_PATH, cluster_type, target_id)
    if distance_matrix is None:
        print("Failed to load VRP data. Exiting.")
        return

    # STEP 1: Qubit Allocation and Objective Function Selection
    if METHOD == 'node_visit_time':
        total_nodes = len(all_nodes)
        num_customers = len(customer_nodes)
        bits_per_node = math.ceil(math.log2(total_nodes))
        num_qubits = num_customers * bits_per_node
        print(f"\nSTEP 1: Qubit Allocation (Node-Visit-Time Encoding)")
        print(f"Total Qubits required: {num_qubits}")
        
        qaoa_circuit_func = create_qaoa_circuit
        objective_func = lambda params, sampler, backend: qaoa_objective_function_nvt(params, sampler, num_qubits, QAOA_P, distance_matrix, num_customers, bits_per_node, depot_node, customer_nodes, total_nodes, backend)
        
        def qaoa_objective_function_nvt(params, sampler, num_qubits, p, distance_matrix, num_customers, bits_per_node, depot_node, customer_nodes, total_nodes, backend):
            gamma = params[:p]
            beta = params[p:]
            qc = qaoa_circuit_func(num_qubits, p, gamma, beta)
            qc_transpiled = transpile(qc, backend=backend)
            job = sampler.run([qc_transpiled], shots=SHOTS)
            result = job.result()
            meas_data = result[0].data.c
            counts = format_measurement_counts(meas_data)
            expected_cost = sum(
                (count / SHOTS) * total_cost_from_bitstring_nvt(bitstring, distance_matrix, num_customers, bits_per_node, depot_node, customer_nodes, total_nodes)
                for bitstring, count in counts.items()
            )
            print(f"  [QAOA Obj] Expected Cost calculated: {expected_cost:.3f}")
            return expected_cost
            
    elif METHOD == 'naive_adjacency':
        num_cities_qubo = len(all_nodes)
        num_qubits = get_num_qubits_qubo(num_cities_qubo)
        print(f"\nSTEP 1: Qubit Allocation (QUBO Encoding)")
        print(f"Total Qubits required (N^2): {num_qubits}")
        
        qaoa_circuit_func = create_qaoa_circuit_qubo
        objective_func = lambda params, sampler, backend: qaoa_objective_function_qubo(params, sampler, num_cities_qubo, QAOA_P, distance_matrix, all_nodes, backend)

        def qaoa_objective_function_qubo(params, sampler, num_cities, p, distance_matrix, all_nodes, backend):
            gamma = params[:p]
            beta = params[p:]
            qc = qaoa_circuit_func(num_cities, p, gamma, beta)
            qc_transpiled = transpile(qc, backend=backend)
            job = sampler.run([qc_transpiled], shots=SHOTS)
            result = job.result()
            meas_data = result[0].data.c
            counts = format_measurement_counts(meas_data)
            expected_cost = sum(
                (count / SHOTS) * calculate_cost_qubo(decode_bitstring_qubo(bitstring, num_cities), distance_matrix, all_nodes)
                for bitstring, count in counts.items()
            )
            print(f"  [QAOA Obj] Expected Cost calculated: {expected_cost:.3f}")
            return expected_cost
    else:
        print("Invalid METHOD selected. Exiting.")
        return

    if num_qubits == 0:
        print("No nodes to route. Exiting.")
        return

    # STEP 2: Load IBM Quantum credentials and choose a backend
    print("\nSTEP 2: Loading IBM Quantum credentials and choosing a backend...")
    try:
        service = QiskitRuntimeService()
        backend = service.backend("ibm_brisbane")
        print(f"Backend selected: {backend.name}")
    except Exception as e:
        print(f"Error loading IBM Quantum service or backend: {e}")
        print("Falling back to AerSimulator (local Qiskit simulator).")
        from qiskit.providers.aer import AerSimulator
        backend = AerSimulator()
        print(f"Backend selected: {backend.name}")
    sampler = Sampler(mode=backend)
    print("Sampler initialized.")

    # STEP 3: Classical Optimization of QAOA Parameters
    print(f"\nSTEP 3: Starting classical optimization for QAOA (p={QAOA_P} layers)...")
    initial_gamma = np.random.rand(QAOA_P) * (np.pi / 4) 
    initial_beta = np.random.rand(QAOA_P) * 2 * np.pi
    initial_params = np.concatenate((initial_gamma, initial_beta))
    bounds = [(0, np.pi / 4)] * QAOA_P + [(0, 2 * np.pi)] * QAOA_P 
    optimization_result = minimize(
        lambda params: objective_func(params, sampler, backend),
        initial_params,
        method='COBYLA',
        bounds=bounds,
        options={'maxiter': QAOA_P*2+3}
    )
    optimal_params = optimization_result.x
    optimal_gamma = optimal_params[:QAOA_P]
    optimal_beta = optimal_params[QAOA_P:]
    print("\nOptimization finished.")
    print(f"Optimal Parameters: gamma={np.round(optimal_gamma, 3)}, beta={np.round(optimal_beta, 3)}")
    print(f"Minimum Expected Cost found: {optimization_result.fun:.3f}")

    # STEP 4: Run the QAOA circuit with optimal parameters
    print("\nSTEP 4: Running QAOA circuit with optimal parameters...")
    if METHOD == 'node_visit_time':
        optimal_qc = qaoa_circuit_func(num_qubits, QAOA_P, optimal_gamma, optimal_beta)
    else: # naive_adjacency
        optimal_qc = qaoa_circuit_func(num_cities_qubo, QAOA_P, optimal_gamma, optimal_beta)
    optimal_qc_transpiled = transpile(optimal_qc, backend=backend)
    job = sampler.run([optimal_qc_transpiled], shots=SHOTS)
    final_result = job.result()

    # STEP 5: Process results and save to CSV
    print("\nSTEP 5: Processing results and saving to CSV...")
    final_meas_data = final_result[0].data.c
    final_counts = format_measurement_counts(final_meas_data)
    print(f"All measurement counts from {SHOTS} shots:\n{final_counts}")

    data_to_write = []
    if METHOD == 'node_visit_time':
        min_classical_cost, avg_classical_cost, max_classical_cost = calculate_classical_benchmarks_nvt(distance_matrix, depot_node, customer_nodes)
        
        customer_time_permutations = list(itertools.permutations(range(num_customers)))
        valid_bitstrings = []
        for perm in customer_time_permutations:
            bitstring_parts = [format(visit_time, '0' + str(bits_per_node) + 'b') for visit_time in perm]
            valid_bitstrings.append("".join(bitstring_parts))
        
        for bitstring in valid_bitstrings:
            count = final_counts.get(bitstring, 0)
            route, _ = decode_bitstring_to_route_nvt(bitstring, num_customers, bits_per_node, depot_node, customer_nodes)
            cost = calculate_route_cost_nvt(route, distance_matrix)
            data_to_write.append(["node_visit_time", cluster_type, target_id, QAOA_P, SHOTS, bitstring, count, cost, min_classical_cost, avg_classical_cost, max_classical_cost, current_timestamp])
    
    elif METHOD == 'naive_adjacency':
        min_classical_cost, avg_classical_cost, max_classical_cost = calculate_classical_benchmarks_qubo(distance_matrix, all_nodes)
        
        all_city_permutations = list(itertools.permutations(all_nodes))
        valid_qubo_bitstrings_info = []
        for perm_route in all_city_permutations:
            bitstring = convert_permutation_to_qubo_bitstring(perm_route, all_nodes)
            valid_qubo_bitstrings_info.append((bitstring, list(perm_route)))
        
        for bitstring, perm_route in valid_qubo_bitstrings_info:
            count = final_counts.get(bitstring, 0)
            cost = calculate_cost_qubo(perm_route, distance_matrix, all_nodes)
            data_to_write.append(["naive_adjacency", cluster_type, target_id, QAOA_P, SHOTS, bitstring, count, cost, min_classical_cost, avg_classical_cost, max_classical_cost, current_timestamp])

    file_exists = os.path.exists(CSV_FILE_PATH)
    with open(CSV_FILE_PATH, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["conf_qubit_alloc", "conf_cluster_type", "conf_target_id", "conf_qaoa_depth", "shots", "bit_string", "count", "cost", "benchmark_min_cost", "benchmark_avg_cost", "benchmark_max_cost", "timestamp"])
        writer.writerows(data_to_write)

    print(f"\nResults have been successfully saved to '{CSV_FILE_PATH}'")
    print("Script finished successfully!")

if __name__ == "__main__":
    main()