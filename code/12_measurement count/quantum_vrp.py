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
from qiskit.quantum_info import SparsePauliOp # For potential Hamiltonian construction, though we'll use a simpler gate approach for QAOA layers
from scipy.optimize import minimize

# Assume 'cluster_complete_graphs.json' is in the current directory
JSON_FILE_PATH = 'cluster_complete_graphs.json'
CSV_FILE_PATH = 'out.csv'

# --- VRP Configuration (as per your request) ---
cluster_type = 'waste_facility'
target_id = 5
# ---------------------------------------------

# --- QAOA Configuration ---
QAOA_P = 3  # Number of QAOA layers (depth). Higher 'p' can lead to better solutions but increases circuit complexity.
SHOTS = 2048 # Number of shots for the sampler
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
                depot_node = target_id # The target_id is the depot
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
    """
    if hasattr(bit_array_data, 'get_counts'):
        counts = bit_array_data.get_counts()
        return counts
    else:
        raise RuntimeError("The BitArray object from Sampler results does not have a 'get_counts()' method. Cannot process measurement data manually without explicit classical register bit mappings. Please ensure your Qiskit Runtime environment is correctly set up.")

def decode_bitstring_to_route(bitstring, num_customers, bits_per_node, depot_node, customer_nodes):
    """
    Decodes a bitstring into a VRP route based on node-visit-time encoding.
    Returns the route and a list of decoded visit times.
    """
    decoded_visit_times = []
    # Ensure customer_nodes are sorted to maintain a consistent mapping
    sorted_customer_nodes = sorted(customer_nodes)

    for i in range(num_customers):
        # Extract the bits for the current customer's visit time
        start_idx = i * bits_per_node
        end_idx = start_idx + bits_per_node
        customer_bits = bitstring[start_idx:end_idx]

        # Decode binary string to integer visit time
        try:
            visit_time = int(customer_bits, 2)
        except ValueError:
            # Handle cases where bitstring might be malformed (shouldn't happen with sampler results)
            visit_time = -1 # Mark as invalid

        decoded_visit_times.append(visit_time)

    # Create a list of (visit_time, customer_node_id) pairs
    # We use sorted_customer_nodes to map the index 'i' to the actual node ID
    customer_visits = []
    for i, time_val in enumerate(decoded_visit_times):
        customer_visits.append((time_val, sorted_customer_nodes[i]))

    # Sort by visit time
    customer_visits.sort()

    # Extract the ordered customer nodes
    ordered_customer_nodes = [node_id for time_val, node_id in customer_visits]

    # Form the complete route including the depot
    route = [depot_node] + ordered_customer_nodes + [depot_node]

    return route, decoded_visit_times

def calculate_route_cost(route, distance_matrix):
    """
    Calculates the total distance of a given route.
    Returns float('inf') if any segment has a None distance.
    """
    total_distance = 0.0
    for i in range(len(route) - 1):
        from_node = route[i]
        to_node = route[i+1]

        if from_node not in distance_matrix or to_node not in distance_matrix[from_node]:
            return math.inf # Path segment not found in matrix

        segment_distance = distance_matrix[from_node][to_node]

        if segment_distance is None:
            return math.inf # Invalid segment (e.g., no direct path)

        total_distance += segment_distance
    return total_distance

def penalize_duplicates_and_out_of_range(decoded_visit_times, total_nodes):
    """
    Calculates penalties for duplicate visit times and out-of-range visit times.
    """
    penalty = 0.0
    valid_range_min = 0 # Assuming visit times can be 0-indexed (0 to N-1)
    valid_range_max = total_nodes - 1

    # Check for out-of-range visit times
    for vt in decoded_visit_times:
        if not (valid_range_min <= vt <= valid_range_max):
            penalty += PENALTY_OUT_OF_RANGE

    # Check for duplicate visit times
    counts = Counter(decoded_visit_times)
    for vt, count in counts.items():
        if count > 1:
            penalty += PENALTY_DUPLICATE_VISIT * (count - 1) # Penalize each duplicate instance

    return penalty

def total_cost_from_bitstring(bitstring, distance_matrix, num_customers, bits_per_node, depot_node, customer_nodes, total_nodes):
    """
    Calculates the total cost (distance + penalties) for a given bitstring.
    """
    route, decoded_visit_times = decode_bitstring_to_route(bitstring, num_customers, bits_per_node, depot_node, customer_nodes)

    distance_cost = calculate_route_cost(route, distance_matrix)
    penalty_cost = penalize_duplicates_and_out_of_range(decoded_visit_times, total_nodes)

    return distance_cost + penalty_cost

def create_qaoa_circuit(num_qubits, p, gamma, beta):
    """
    Creates a QAOA circuit for a given depth p and parameters gamma, beta.
    This uses a simplified cost layer for demonstration.
    """
    qc = QuantumCircuit(num_qubits, num_qubits) # num_qubits for classical bits for measurement

    # Initial layer: Apply Hadamard to all qubits
    qc.h(range(num_qubits))
    qc.barrier()

    for k in range(p):
        # Cost Layer (U_C): Apply problem Hamiltonian (simplified)
        # For VRP, this would encode distances and constraints.
        # Here, we use RZ for single-qubit terms and RZZ for two-qubit interactions.
        # The actual cost is evaluated classically based on measurement outcomes.
        for i in range(num_qubits):
            qc.rz(2 * gamma[k], i) # Single qubit terms
        for i in range(num_qubits):
            for j in range(i + 1, num_qubits):
                # Apply RZZ gate for interactions between all pairs (can be optimized for specific problem)
                qc.rzz(2 * gamma[k], i, j)
        qc.barrier()

        # Mixer Layer (U_M): Apply mixer Hamiltonian
        # Typically a sum of Pauli X operators, implemented with RX gates.
        for i in range(num_qubits):
            qc.rx(2 * beta[k], i)
        qc.barrier()

    # Measure all qubits
    qc.measure(range(num_qubits), range(num_qubits))

    return qc

def qaoa_objective_function(params, sampler, num_qubits, p, distance_matrix, num_customers, bits_per_node, depot_node, customer_nodes, total_nodes, backend):
    """
    Objective function for the classical optimizer.
    Calculates the expected cost of the QAOA circuit for given parameters.
    """
    # Split parameters into gamma and beta
    gamma = params[:p]
    beta = params[p:]

    print(f"  [QAOA Obj] Current parameters: gamma={np.round(gamma, 2)}, beta={np.round(beta, 2)}")
    print("  [QAOA Obj] Creating QAOA circuit...")
    qc = create_qaoa_circuit(num_qubits, p, gamma, beta)
    print("  [QAOA Obj] Circuit created. Transpiling...")
    
    # Transpile the circuit for the selected backend BEFORE running on the sampler
    qc_transpiled = transpile(qc, backend=backend)
    print("  [QAOA Obj] Circuit transpiled. Running on Sampler...")

    # Run the transpiled circuit on the sampler
    start_job_time = time.time() # Start timer for job
    job = sampler.run([qc_transpiled], shots=SHOTS)
    result = job.result()
    end_job_time = time.time() # End timer for job
    print(f"  [QAOA Obj] Sampler job completed in {end_job_time - start_job_time:.2f} seconds. Processing results...")

    # Get raw measurement data and format into counts
    meas_data = result[0].data.c 
    counts = format_measurement_counts(meas_data) # This function now has no prints

    expected_cost = 0.0
    print("  [QAOA Obj] Calculating expected cost from measurements...")
    # Iterate through counts to calculate expected cost
    for bitstring, count in counts.items():
        # Convert count to probability for expected value calculation
        prob = count / SHOTS 
        cost = total_cost_from_bitstring(bitstring, distance_matrix, num_customers, bits_per_node, depot_node, customer_nodes, total_nodes)
        expected_cost += prob * cost
    print(f"  [QAOA Obj] Expected Cost calculated: {expected_cost:.3f}")

    print(f"Expected Cost: {expected_cost:.3f}") # This print statement is for the overall optimization progress
    return expected_cost

def calculate_classical_benchmarks(distance_matrix, depot_node, customer_nodes):
    """
    Calculates the minimum, maximum, and average cost using a brute-force classical approach.
    """
    print("\nCalculating classical benchmark costs (brute-force)...")
    costs = []

    # Generate all permutations of customer nodes
    all_customer_permutations = list(itertools.permutations(customer_nodes))

    for perm in all_customer_permutations:
        route = [depot_node] + list(perm) + [depot_node]
        cost = calculate_route_cost(route, distance_matrix)
        if cost != math.inf:
            costs.append(cost)

    if not costs:
        return float('inf'), 0.0, float('inf')

    min_cost = min(costs)
    max_cost = max(costs)
    avg_cost = sum(costs) / len(costs)

    print(f"Classical minimum cost found: {min_cost:.3f}")
    print(f"Classical average cost: {avg_cost:.3f}")
    print(f"Classical maximum cost: {max_cost:.3f}")
    
    return min_cost, avg_cost, max_cost

def main():
    print("Starting QAOA for VRP script...")

    current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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
        qaoa_objective_function,
        initial_params,
        args=(sampler, num_qubits, QAOA_P, distance_matrix, num_customers, bits_per_node, depot_node, customer_nodes, total_nodes, backend),
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

    # STEP 4: Run the QAOA circuit with optimal parameters and get results
    print("\nSTEP 4: Running QAOA circuit with optimal parameters...")
    optimal_qc = create_qaoa_circuit(num_qubits, QAOA_P, optimal_gamma, optimal_beta)
    
    if backend.name != 'aer_simulator':
        optimal_qc_transpiled = transpile(optimal_qc, backend=backend)
    else:
        optimal_qc_transpiled = optimal_qc

    job = sampler.run([optimal_qc_transpiled], shots=SHOTS)
    final_result = job.result()

    # STEP 5: Process results, calculate costs, and save to CSV
    print("\nSTEP 5: Processing results and saving to CSV...")
    final_meas_data = final_result[0].data.c
    final_counts = format_measurement_counts(final_meas_data)
    print(f"All measurement counts from {SHOTS} shots:\n{final_counts}")

    # Calculate classical benchmark costs
    min_classical_cost, avg_classical_cost, max_classical_cost = calculate_classical_benchmarks(distance_matrix, depot_node, customer_nodes)
    
    # Generate all valid bitstrings from permutations
    print("\nGenerating all possible valid bitstrings for analysis...")
    customer_time_permutations = list(itertools.permutations(range(num_customers)))
    valid_bitstrings = []
    sorted_customer_nodes = sorted(customer_nodes)
    for perm in customer_time_permutations:
        bitstring_parts = []
        for i in range(num_customers):
            visit_time = perm[i]
            bit_str = format(visit_time, '0' + str(bits_per_node) + 'b')
            bitstring_parts.append(bit_str)
        full_bitstring = "".join(bitstring_parts)
        valid_bitstrings.append(full_bitstring)

    # Prepare data for CSV
    data_to_write = []
    
    # Iterate through all valid bitstrings to get a complete picture, even if count is 0
    for bitstring in valid_bitstrings:
        count = final_counts.get(bitstring, 0) # Get count, default to 0 if not measured
        route, _ = decode_bitstring_to_route(bitstring, num_customers, bits_per_node, depot_node, customer_nodes)
        cost = calculate_route_cost(route, distance_matrix)
        
        row = [
            "node_visit_time",
            cluster_type,
            target_id,
            QAOA_P,
            SHOTS,
            bitstring,
            count,
            cost,
            min_classical_cost,
            avg_classical_cost,
            max_classical_cost,
            current_timestamp
        ]
        data_to_write.append(row)

    # Write to CSV file
    file_exists = os.path.exists(CSV_FILE_PATH)
    with open(CSV_FILE_PATH, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow([
                "conf_qubit_alloc", 
                "conf_cluster_type", 
                "conf_target_id", 
                "conf_qaoa_depth", 
                "shots", 
                "bit_string", 
                "count", 
                "cost", 
                "benchmark_min_cost", 
                "benchmark_avg_cost", 
                "benchmark_max_cost", 
                "timestamp"
            ])
        writer.writerows(data_to_write)

    print(f"\nResults have been successfully saved to '{CSV_FILE_PATH}'")
    print("Script finished successfully!")

if __name__ == "__main__":
    main()