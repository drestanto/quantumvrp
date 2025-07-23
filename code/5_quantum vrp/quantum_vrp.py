import json
import math
import time
import numpy as np

from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer

from qiskit.algorithms import QAOA
from qiskit.algorithms.optimizers import COBYLA

from qiskit_ibm_runtime import QiskitRuntimeService, Sampler

# Constants
JSON_FILE_PATH = 'cluster_complete_graphs.json'

# --- Editable variables ---
cluster_type = 'waste_facility'
target_id = 5
# --------------------------

def calculate_path_distance(path, distance_matrix):
    total_distance = 0.0
    for i in range(len(path) - 1):
        from_node = path[i]
        to_node = path[i + 1]
        if from_node not in distance_matrix or to_node not in distance_matrix[from_node] or distance_matrix[from_node][to_node] is None:
            return math.inf
        total_distance += distance_matrix[from_node][to_node]
    return total_distance

def decode_qaoa_result(result_bitstring, n_customers, num_bits_per_pos, all_nodes_in_cluster, depot_original_id, processed_distance_matrix):
    customer_positions = {}
    decoded_positions_list = []
    bit_idx_counter = 0
    for i in range(n_customers):
        customer_bits = result_bitstring[bit_idx_counter: bit_idx_counter + num_bits_per_pos]
        decoded_pos = 0
        for k in range(num_bits_per_pos):
            decoded_pos += customer_bits[k] * (2 ** (num_bits_per_pos - 1 - k))
        customer_positions[i] = decoded_pos
        decoded_positions_list.append((decoded_pos, i))
        bit_idx_counter += num_bits_per_pos

    assigned_positions = [pos for pos, _ in decoded_positions_list]
    if len(set(assigned_positions)) != n_customers:
        return None, math.inf

    N_total_nodes = len(all_nodes_in_cluster)
    if any(pos < 0 or pos >= N_total_nodes for pos in assigned_positions):
        return None, math.inf

    decoded_positions_list.sort()
    customer_sequence_internal_idxs = [customer_idx for pos, customer_idx in decoded_positions_list]

    final_path = [depot_original_id]
    original_customers_in_order = [all_nodes_in_cluster[c_internal_idx] for c_internal_idx in customer_sequence_internal_idxs]
    final_path.extend(original_customers_in_order)
    final_path.append(depot_original_id)

    total_distance = calculate_path_distance(final_path, processed_distance_matrix)
    return final_path, total_distance

def solve_vrp_qaoa_for_cluster(cluster_id, raw_distance_matrix, cluster_type_name):
    start_time = time.perf_counter()

    # Process distance matrix keys as ints
    processed_distance_matrix = {}
    for r_key, row_val in raw_distance_matrix.items():
        processed_row = {}
        for c_key, dist_val in row_val.items():
            processed_row[int(c_key)] = dist_val
        processed_distance_matrix[int(r_key)] = processed_row

    all_nodes_original_ids = sorted(list(processed_distance_matrix.keys()))
    depot_original_id = cluster_id
    customers_original_ids = [node for node in all_nodes_original_ids if node != depot_original_id]
    n_customers = len(customers_original_ids)
    N_total_nodes = len(all_nodes_original_ids)

    print(f"\n--- QAOA for {cluster_type_name} ID: {depot_original_id} ---")
    print(f"Total nodes in cluster (N): {N_total_nodes}")
    print(f"Number of customers (n): {n_customers}")
    print(f"Depot (original ID): {depot_original_id}")
    print(f"Customers (original IDs): {customers_original_ids}")

    if n_customers == 0:
        print("No customers in this cluster to form a route (only depot).")
        print(f"Time processed: {0:.4f} seconds")
        return None, math.inf

    num_bits_per_pos = math.ceil(math.log2(N_total_nodes))
    total_qubits = n_customers * num_bits_per_pos

    print(f"Qubits per customer (b = ceil(log2(N))): {num_bits_per_pos} bits")
    print(f"Total qubits for QAOA (n * b): {total_qubits}")

    # Build Quadratic Program
    qp = QuadraticProgram(name="VRP")
    z_vars_names = []
    for i in range(n_customers):
        customer_z_bits = []
        for k in range(num_bits_per_pos):
            var_name = f'z_{i}_{k}'
            qp.binary_var(var_name)
            customer_z_bits.append(var_name)
        z_vars_names.append(customer_z_bits)

    # Placeholder objective for QP to run - minimize sum of all binary variables
    var_list = [qp.get_variable(name) for sublist in z_vars_names for name in sublist]
    qp.minimize(linear={v.name: 1 for v in var_list})

    # Setup IBM Quantum runtime service
    service = QiskitRuntimeService()  # Make sure your IBM Quantum API token is saved or set env variable

    sampler = Sampler(session=service)

    # Setup QAOA with COBYLA optimizer and 1 repetition
    optimizer = COBYLA(maxiter=100)
    qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=1, seed=123)

    print("\nRunning QAOA on IBM Quantum hardware (simulator or real)...")

    # Convert quadratic program to Ising operator and offset
    operator, offset = qp.to_ising()

    qaoa_result = qaoa.compute_minimum_eigenvalue(operator=operator)

    # Qiskit returns eigenstate as complex amplitudes, extract measured bitstring (in reverse)
    # Here we convert amplitudes to bitstring by taking the index of max amplitude
    max_amp_idx = np.argmax(np.abs(qaoa_result.eigenstate)**2)
    bitstring = [int(x) for x in bin(max_amp_idx)[2:].zfill(total_qubits)]
    
    print(f"QAOA run completed. Best bitstring: {bitstring}")

    final_path, final_distance = decode_qaoa_result(
        bitstring, n_customers, num_bits_per_pos, all_nodes_original_ids, depot_original_id, processed_distance_matrix
    )

    end_time = time.perf_counter()
    time_processed = end_time - start_time

    if final_path:
        formatted_path = " -> ".join(map(str, final_path))
        print(f"\nOptimal Path (QAOA decoded): {formatted_path}")
        print(f"Total Distance (QAOA decoded): {final_distance:.3f} km")
    else:
        print("\nQAOA failed to find a valid/reachable path or decoding resulted in an invalid path.")

    print(f"Time processed: {time_processed:.4f} seconds")
    return final_path, final_distance

def main_single_cluster():
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
                solve_vrp_qaoa_for_cluster(target_id, raw_distance_matrix, cluster_description)
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
