import json
import itertools
import math
import time
from collections import Counter

import numpy as np
from qiskit import QuantumCircuit, transpile
# 🛑 Removed Sampler and QiskitRuntimeService imports
from qiskit.providers.fake_provider import GenericBackendV2 # 👈 ADD THIS IMPORT
from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import minimize

# Assume 'cluster_complete_graphs.json' is in the current directory
JSON_FILE_PATH = 'cluster_complete_graphs.json'

# --- VRP Configuration (as per your request) ---
cluster_type = 'waste_facility'
target_id = 5
# ---------------------------------------------

# --- QAOA Configuration ---
QAOA_P = 2  # Number of QAOA layers (depth). Higher 'p' can lead to better solutions but increases circuit complexity.
SHOTS = 128 # Number of shots for the backend run
# --------------------------

# --- Penalty Weights for Cost Function ---
PENALTY_DUPLICATE_VISIT = 1000.0
PENALTY_OUT_OF_RANGE = 1000.0
# ---------------------------------------

def load_vrp_data(json_file_path, cluster_type, target_id):
    """
    Loads the VRP distance matrix for a specific cluster from a JSON file.
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

# 🛑 This function is no longer needed with the change to backend.run().
# The result object from backend.run() has a get_counts() method
# that we can use directly. We will remove this function.
# def format_measurement_counts(bit_array_data):
#     """
#     Takes a BitArray object from Sampler results and returns a dictionary
#     with bitstring keys formatted to the correct number of qubits.
#     """
#     if hasattr(bit_array_data, 'get_counts'):
#         counts = bit_array_data.get_counts()
#         return counts
#     else:
#         raise RuntimeError("The BitArray object from Sampler results does not have a 'get_counts()' method.")

def decode_bitstring_to_route(bitstring, num_customers, bits_per_node, depot_node, customer_nodes):
    """
    Decodes a bitstring into a VRP route based on node-visit-time encoding.
    Returns the route and a list of decoded visit times.
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

def calculate_route_cost(route, distance_matrix):
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

def penalize_duplicates_and_out_of_range(decoded_visit_times, total_nodes):
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

def total_cost_from_bitstring(bitstring, distance_matrix, num_customers, bits_per_node, depot_node, customer_nodes, total_nodes):
    """
    Calculates the total cost (distance + penalties) for a given bitstring.
    """
    route, decoded_visit_times = decode_bitstring_to_route(bitstring, num_customers, bits_per_node, depot_node, customer_nodes)

    distance_cost = calculate_route_cost(route, distance_matrix)
    penalty_cost = penalize_duplicates_and_out_of_range(decoded_visit_times, total_nodes)

    return distance_cost + penalty_cost

# 🛑 Modified function signature to accept `backend` instead of `sampler`
def qaoa_objective_function(params, backend, num_qubits, p, distance_matrix, num_customers, bits_per_node, depot_node, customer_nodes, total_nodes):
    """
    Objective function for the classical optimizer.
    Calculates the expected cost of the QAOA circuit for given parameters.
    """
    gamma = params[:p]
    beta = params[p:]

    print(f"  [QAOA Obj] Current parameters: gamma={np.round(gamma, 2)}, beta={np.round(beta, 2)}")
    print("  [QAOA Obj] Creating QAOA circuit...")
    qc = create_qaoa_circuit(num_qubits, p, gamma, beta)
    print("  [QAOA Obj] Circuit created. Transpiling...")
    
    qc_transpiled = transpile(qc, backend=backend)
    print("  [QAOA Obj] Circuit transpiled. Running on backend...")

    start_job_time = time.time()
    # 🛑 Replaced sampler.run with backend.run
    job = backend.run(qc_transpiled, shots=SHOTS)
    result = job.result()
    end_job_time = time.time()
    print(f"  [QAOA Obj] Backend job completed in {end_job_time - start_job_time:.2f} seconds. Processing results...")

    # 🛑 Changed how counts are retrieved to use the result object's method
    counts = result.get_counts(qc_transpiled)

    expected_cost = 0.0
    print("  [QAOA Obj] Calculating expected cost from measurements...")
    for bitstring, count in counts.items():
        prob = count / SHOTS
        cost = total_cost_from_bitstring(bitstring, distance_matrix, num_customers, bits_per_node, depot_node, customer_nodes, total_nodes)
        expected_cost += prob * cost
    print(f"  [QAOA Obj] Expected Cost calculated: {expected_cost:.3f}")

    print(f"Expected Cost: {expected_cost:.3f}")
    return expected_cost

def create_qaoa_circuit(num_qubits, p, gamma, beta):
    """
    Creates a QAOA circuit for a given depth p and parameters gamma, beta.
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

def main():
    print("Starting QAOA for VRP script...")

    # STEP 0: Load VRP Data
    distance_matrix, nodes, depot_node, customer_nodes = load_vrp_data(JSON_FILE_PATH, cluster_type, target_id)

    if distance_matrix is None:
        print("Failed to load VRP data. Exiting.")
        return

    total_nodes = len(nodes)
    num_customers = len(customer_nodes)

    # STEP 1: Calculate Qubit Allocation (Node-Visit-Time)
    bits_per_node = math.ceil(math.log2(total_nodes))
    num_qubits = num_customers * bits_per_node

    print(f"\nSTEP 1: Qubit Allocation (Node-Visit-Time Encoding)")
    print(f"Total Nodes (N): {total_nodes}")
    print(f"Number of Customers (n): {num_customers}")
    print(f"Bits per node (ceil(log2(N))): {bits_per_node}")
    print(f"Total Qubits required: {num_qubits}")

    if num_qubits == 0:
        print("No customers to route. Exiting.")
        return

    # 🛑 STEP 2: Use a Fake Provider instead of QiskitRuntimeService
    print("\nSTEP 2: Using Fake Provider for backend...")
    try:
        # 🛑 Instantiate GenericBackendV2 with the required number of qubits
        backend = GenericBackendV2(num_qubits=num_qubits)
        print(f"Backend selected: {backend.name}")
    except Exception as e:
        print(f"Error initializing fake backend: {e}")
        print("Exiting.")
        return
    
    # 🛑 The sampler is no longer used
    # sampler = Sampler(backend=backend)
    # print("Sampler initialized.")

    # STEP 3: Classical Optimization of QAOA Parameters
    print(f"\nSTEP 3: Starting classical optimization for QAOA (p={QAOA_P} layers)...")

    initial_gamma = np.random.rand(QAOA_P) * (np.pi / 4)
    initial_beta = np.random.rand(QAOA_P) * 2 * np.pi
    initial_params = np.concatenate((initial_gamma, initial_beta))

    bounds = [(0, np.pi / 4)] * QAOA_P + [(0, 2 * np.pi)] * QAOA_P

    optimization_result = minimize(
        qaoa_objective_function,
        initial_params,
        # 🛑 Pass `backend` instead of `sampler`
        args=(backend, num_qubits, QAOA_P, distance_matrix, num_customers, bits_per_node, depot_node, customer_nodes, total_nodes),
        method='COBYLA',
        bounds=bounds,
        options={'maxiter': 7}
    )

    optimal_params = optimization_result.x
    optimal_gamma = optimal_params[:QAOA_P]
    optimal_beta = optimal_params[QAOA_P:]

    print("\nOptimization finished.")
    print(f"Optimal Parameters: gamma={np.round(optimal_gamma, 3)}, beta={np.round(optimal_beta, 3)}")
    print(f"Minimum Expected Cost found: {optimization_result.fun:.3f}")

    # STEP 4: Run the QAOA circuit with optimal parameters and get results
    print("\nSTEP 4: Running QAOA circuit with optimal parameters...")
    optimal_qc = create_qaoa_circuit(num_qubits, QAOA_P, optimal_gamma, optimal_beta)
    
    optimal_qc_transpiled = transpile(optimal_qc, backend=backend)

    # 🛑 Replaced sampler.run with backend.run
    job = backend.run(optimal_qc_transpiled, shots=SHOTS)
    final_result = job.result()

    # STEP 5: Process the results and interpret the optimal route...
    print("\nSTEP 5: Processing results and interpreting optimal route...")
    
    # 🛑 Changed how counts are retrieved to use the result object's method
    final_counts = final_result.get_counts(optimal_qc_transpiled)

    if not final_counts:
        print("No measurement results obtained. Cannot determine optimal route.")
        return

    most_probable_bitstring = max(final_counts, key=final_counts.get)
    most_probable_count = final_counts[most_probable_bitstring]
    most_probable_probability = most_probable_count / SHOTS

    print(f"Most probable bitstring: {most_probable_bitstring} (Count: {most_probable_count}, Probability: {most_probable_probability:.3f})")

    optimal_route, decoded_visit_times = decode_bitstring_to_route(
        most_probable_bitstring, num_customers, bits_per_node, depot_node, customer_nodes
    )
    final_cost = total_cost_from_bitstring(
        most_probable_bitstring, distance_matrix, num_customers, bits_per_node, depot_node, customer_nodes, total_nodes
    )

    print(f"Decoded visit times for customers {sorted(customer_nodes)}: {decoded_visit_times}")
    print(f"Optimal VRP Route: {' -> '.join(map(str, optimal_route))}")
    print(f"Total Cost of Optimal Route (Distance + Penalties): {final_cost:.3f}")

    print("\nScript finished successfully!")

if __name__ == "__main__":
    main()