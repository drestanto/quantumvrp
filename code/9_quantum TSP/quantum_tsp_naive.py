import json
import itertools
import math
import time
from collections import Counter

import numpy as np
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import minimize

# Assume 'cluster_complete_graphs.json' is in the current directory
JSON_FILE_PATH = 'cluster_complete_graphs.json'

# --- VRP Configuration (as per your request) ---
cluster_type = 'waste_facility'
target_id = 5
# ---------------------------------------------

# --- QAOA Configuration ---
QAOA_P = 3  # Number of QAOA layers (depth). Higher 'p' can lead to better solutions but increases circuit complexity.
SHOTS = 512 # Number of shots for the sampler
# --------------------------

# --- Penalty Weights for Cost Function (Adjusted for QUBO) ---
# These are the A and B coefficients in the QUBO formulation
QUBO_PENALTY_A = 1000.0
QUBO_WEIGHT_B = 1.0
# -----------------------------------------------------------

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
                # Convert string keys to integers
                for r_key, row_val in raw_distance_matrix.items():
                    processed_row = {}
                    for c_key, dist_val in row_val.items():
                        processed_row[int(c_key)] = dist_val
                    processed_distance_matrix[int(r_key)] = processed_row

                nodes = sorted(list(processed_distance_matrix.keys()))
                # For VRP, the depot is typically one of the nodes.
                # Assuming the target_id is the depot node.
                depot_node = target_id
                # All other nodes are customer nodes.
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
        # This fallback is for local AerSimulator results where .get_counts() might not be directly available
        # or if the data structure changes. For Qiskit Runtime, .get_counts() is standard.
        raise RuntimeError("The BitArray object from Sampler results does not have a 'get_counts()' method. Cannot process measurement data manually without explicit classical register bit mappings. Please ensure your Qiskit Runtime environment is correctly set up.")

# --- New functions for QUBO formulation ---

def get_num_qubits(num_cities):
    """Returns number of qubits = N^2 for N cities (including depot)."""
    return num_cities * num_cities

def decode_bitstring_qubo(bitstring, num_cities):
    """
    Decode bitstring of length N^2 into a route.
    bitstring is ordered as [x_{0,0}, x_{0,1}, ..., x_{0,N-1}, x_{1,0}, ..., x_{N-1,N-1}]
    Each x_{i,p} is 1 if city i visited at position p.
    Returns the route as list of city indices in visiting order.
    """
    # Qiskit bitstrings are typically ordered such that index 0 is the least significant bit
    # or the rightmost bit, so we reverse it to match the logical x_{i,p} ordering.
    # The reshape assumes the bits are flattened in row-major order:
    # x_{0,0}, x_{0,1}, ..., x_{0,N-1}, x_{1,0}, ..., x_{N-1,N-1}
    # Reshape from 1D array to N x N matrix.
    x = np.array(list(map(int, bitstring))).reshape((num_cities, num_cities))

    route = []
    # For each position p, find city i with x[i,p] == 1
    for p in range(num_cities):
        # Find all city indices where x_{i,p} is 1 for the current position p
        city_indices_at_pos_p = np.where(x[:, p] == 1)[0]
        if len(city_indices_at_pos_p) == 1:
            route.append(int(city_indices_at_pos_p[0]))
        else:
            # If a position has no city or multiple cities, it's an invalid assignment.
            # Mark with -1 to indicate an issue, which will be penalized by calculate_cost_qubo.
            route.append(-1)
    return route

def calculate_cost_qubo(route, distance_matrix, all_nodes, A=QUBO_PENALTY_A, B=QUBO_WEIGHT_B):
    """
    Calculate the QUBO cost for a given route and distance matrix.
    - Penalty A for invalid tours (city visited more than once, position not assigned exactly one city,
      or missing start/end depot constraint if applicable).
    - Cost B * sum of distances for consecutive cities in route.
    """
    N = len(all_nodes) # Total number of nodes (cities including depot)
    penalty = 0

    # Constraint 1: Each position p must be visited by exactly one city i (i.e., sum_i x_{i,p} = 1 for all p)
    # This is implicitly handled by decode_bitstring_qubo marking -1 for invalid positions.
    # We check if the route has the correct number of cities
    if len(route) != N:
        penalty += A * abs(len(route) - N) # Penalize if route length is not N

    # Constraint 2: Each city i must appear exactly once in the route (i.e., sum_p x_{i,p} = 1 for all i)
    # Count occurrences of each city in the decoded route
    city_counts = Counter(route)
    for city_idx in all_nodes: # Iterate through all expected city indices
        count = city_counts.get(city_idx, 0)
        if count != 1:
            penalty += A * abs(count - 1) # Penalize if a city is not visited exactly once
    
    # Penalize if any -1 (invalid city) exists in the route
    if -1 in city_counts:
        penalty += A * city_counts[-1]


    # Calculate travel cost (distance)
    travel_cost = 0
    # The route from decode_bitstring_qubo is just the permutation of cities.
    # We need to form a cycle to calculate the total travel cost.
    # Assuming the route starts from the depot (node_0), visits all cities, and returns to depot.
    # The QUBO formulation generally implicitly handles the cycle by connecting p=N-1 to p=0.
    # Let's construct the full cycle for distance calculation.
    
    # Ensure the route represents a valid permutation of cities (excluding depot for now if it's not explicitly in route)
    # The QUBO is typically for TSP, which finds a permutation of all nodes.
    # If VRP with a depot, we need to adapt the cost function.
    # For a general TSP/VRP, the QUBO maps an ordered sequence of ALL nodes.
    
    # For TSP, a cycle means connecting the last city back to the first.
    # If route is [c1, c2, ..., cN], the edges are (c1,c2), (c2,c3), ..., (cN-1,cN), (cN,c1).
    # Assuming `route` contains *all* N cities in order.
    
    # Create the full traversal, linking the last city back to the first to form a cycle.
    full_path_for_cost = route + [route[0]] # Completes the cycle

    for i in range(len(full_path_for_cost) - 1):
        from_city = full_path_for_cost[i]
        to_city = full_path_for_cost[i+1]

        # Penalize if any city in the path is marked as invalid (-1)
        if from_city == -1 or to_city == -1:
            penalty += A # Should already be covered by city_counts check, but adding as safeguard
            continue # Skip distance calculation for invalid segments

        # Ensure the keys exist in the distance matrix
        dist = distance_matrix.get(from_city, {}).get(to_city, None)
        if dist is None:
            penalty += A  # Penalize if no direct path exists in matrix
        else:
            travel_cost += dist

    total_cost = penalty + B * travel_cost
    return total_cost


def create_qaoa_circuit_qubo(num_cities, p, gamma, beta):
    """
    Build QAOA circuit for a QUBO formulation with N^2 qubits.
    The circuit applies cost and mixer layers.
    """
    num_qubits = get_num_qubits(num_cities)
    qc = QuantumCircuit(num_qubits, num_qubits) # num_qubits for classical bits for measurement

    # Initial layer: Apply Hadamard to all qubits
    qc.h(range(num_qubits))
    qc.barrier()

    for layer in range(p):
        # Cost Layer (U_C): Apply problem Hamiltonian
        # For QUBO, this involves RZ rotations for diagonal terms and RZZ for quadratic terms.
        # This is a general QAOA structure; the specific coefficients would depend on the QUBO matrix.
        # For simplicity, we apply a general structure with gamma.
        
        # Single qubit rotations (diagonal terms / penalty terms)
        for q in range(num_qubits):
            qc.rz(2 * gamma[layer], q)

        # Pairwise interactions (quadratic terms / distance terms)
        # Apply RZZ gates between all pairs of qubits for a dense interaction graph.
        # In a real QUBO, these would be specific to the problem's interactions.
        for i in range(num_qubits):
            for j in range(i + 1, num_qubits):
                qc.rzz(2 * gamma[layer], i, j) # Simplified general interaction
        qc.barrier()

        # Mixer Layer (U_M): Apply mixer Hamiltonian
        # Typically a sum of Pauli X operators, implemented with RX gates.
        for i in range(num_qubits):
            qc.rx(2 * beta[layer], i)
        qc.barrier()

    # Measure all qubits
    qc.measure(range(num_qubits), range(num_qubits))

    return qc

def qaoa_objective_function_qubo(params, sampler, num_cities, p, distance_matrix, all_nodes, backend, A=QUBO_PENALTY_A, B=QUBO_WEIGHT_B):
    """
    Objective function for the classical optimizer, using the QUBO cost calculation.
    Calculates the expected cost of the QAOA circuit for given parameters.
    """
    # Split parameters into gamma and beta
    gamma = params[:p]
    beta = params[p:]

    print(f"  [QAOA Obj] Current parameters: gamma={np.round(gamma, 2)}, beta={np.round(beta, 2)}")
    print("  [QAOA Obj] Creating QAOA circuit (QUBO formulation)...")
    qc = create_qaoa_circuit_qubo(num_cities, p, gamma, beta)
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
    counts = format_measurement_counts(meas_data)

    expected_cost = 0.0
    print("  [QAOA Obj] Calculating expected cost from measurements...")
    # Iterate through counts to calculate expected cost
    for bitstring, count in counts.items():
        # Convert count to probability for expected value calculation
        prob = count / SHOTS 
        # Decode the bitstring using the QUBO specific decoder
        # Note: num_cities here refers to the total number of nodes (including depot)
        # as the QUBO formulation typically encodes all nodes.
        route = decode_bitstring_qubo(bitstring, num_cities) 
        cost = calculate_cost_qubo(route, distance_matrix, all_nodes, A, B)
        expected_cost += prob * cost
    print(f"  [QAOA Obj] Expected Cost calculated: {expected_cost:.3f}")

    print(f"Expected Cost: {expected_cost:.3f}") # This print statement is for the overall optimization progress
    return expected_cost

def main():
    print("Starting QAOA for VRP script (using QUBO formulation)...")

    # STEP 0: Load VRP Data
    distance_matrix, all_nodes, depot_node, customer_nodes = load_vrp_data(JSON_FILE_PATH, cluster_type, target_id)

    if distance_matrix is None:
        print("Failed to load VRP data. Exiting.")
        return

    # In QUBO, num_cities for the N^2 qubits refers to all nodes (depot + customers)
    num_cities_qubo = len(all_nodes) 
    num_qubits = get_num_qubits(num_cities_qubo)

    print(f"\nSTEP 1: Qubit Allocation (QUBO Encoding)")
    print(f"Total Cities (N for QUBO): {num_cities_qubo}")
    print(f"Total Qubits required (N^2): {num_qubits}")

    if num_qubits == 0:
        print("No cities to route. Exiting.")
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

    # The bounds for gamma and beta are standard for QAOA
    bounds = [(0, np.pi / 4)] * QAOA_P + [(0, 2 * np.pi)] * QAOA_P 

    # Use COBYLA optimizer (suitable for black-box functions)
    optimization_result = minimize(
        qaoa_objective_function_qubo, # Use the QUBO specific objective function
        initial_params,
        # Pass the arguments required by qaoa_objective_function_qubo
        args=(sampler, num_cities_qubo, QAOA_P, distance_matrix, all_nodes, backend, QUBO_PENALTY_A, QUBO_WEIGHT_B),
        method='COBYLA',
        bounds=bounds,
        options={'maxiter': 7} # Limit iterations for demonstration
    )

    optimal_params = optimization_result.x
    optimal_gamma = optimal_params[:QAOA_P]
    optimal_beta = optimal_params[QAOA_P:]

    print("\nOptimization finished.")
    print(f"Optimal Parameters: gamma={np.round(optimal_gamma, 3)}, beta={np.round(optimal_beta, 3)}")
    print(f"Minimum Expected Cost found: {optimization_result.fun:.3f}")

    # STEP 4: Run the QAOA circuit with optimal parameters and get results
    print("\nSTEP 4: Running QAOA circuit with optimal parameters...")
    optimal_qc = create_qaoa_circuit_qubo(num_cities_qubo, QAOA_P, optimal_gamma, optimal_beta)
    
    # Transpile for the selected backend
    optimal_qc_transpiled = transpile(optimal_qc, backend=backend)

    job = sampler.run([optimal_qc_transpiled], shots=SHOTS)
    final_result = job.result()

    # STEP 5: Process the results and interpret the optimal route...
    print("\nSTEP 5: Processing results and interpreting optimal route...")
    final_meas_data = final_result[0].data.c
    final_counts = format_measurement_counts(final_meas_data)

    # Find the most probable bitstring
    if not final_counts:
        print("No measurement results obtained. Cannot determine optimal route.")
        return

    most_probable_bitstring = max(final_counts, key=final_counts.get)
    most_probable_count = final_counts[most_probable_bitstring]
    most_probable_probability = most_probable_count / SHOTS

    print(f"Most probable bitstring: {most_probable_bitstring} (Count: {most_probable_count}, Probability: {most_probable_probability:.3f})")

    # Decode the most probable bitstring using the QUBO decoder
    optimal_route_qubo = decode_bitstring_qubo(most_probable_bitstring, num_cities_qubo)
    final_cost_qubo = calculate_cost_qubo(optimal_route_qubo, distance_matrix, all_nodes, QUBO_PENALTY_A, QUBO_WEIGHT_B)

    print(f"Optimal VRP Route (decoded from QUBO): {' -> '.join(map(str, optimal_route_qubo))}")
    print(f"Total Cost of Optimal Route (Distance + Penalties, QUBO): {final_cost_qubo:.3f}")

    print("\nScript finished successfully!")

if __name__ == "__main__":
    main()
