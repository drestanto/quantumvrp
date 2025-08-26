# 12\_measurement\_count/

This directory contains two scripts, `quantum_vrp.py` and `quantum_tsp_naive.py`, that solve the Traveling Salesman Problem (a simplified version of the VRP) using the Quantum Approximate Optimization Algorithm (QAOA). Both scripts benchmark their quantum results against a classical brute-force solver and log the output to a CSV file. The primary difference between the two is the method used to encode the problem on the quantum circuit.

-----

## Contents

  * `quantum_vrp.py`:

      * **Purpose**: This script uses a **"node-visit-time"** encoding to solve the TSP. This approach maps each city and its position in the tour to a set of qubits, where the number of qubits required scales with `(N-1) * log2(N-1)` for a problem with `N` cities. It's often more qubit-efficient than the naive QUBO approach for small-to-medium problems.
      * **Dependencies**: This script relies on `qiskit`, `numpy`, `scipy`, and the `qiskit-ibm-runtime` library.
      * **Usage**:

    <!-- end list -->

    ```bash
    python quantum_vrp.py
    ```

  * `quantum_tsp_naive.py`:

      * **Purpose**: This script uses a **"naive adjacency"** or **QUBO (Quadratic Unconstrained Binary Optimization)** encoding to solve the TSP. This formulation requires a large number of qubits, specifically `N^2` qubits for a problem with `N` cities. While less efficient in terms of qubit count, it is a standard approach for mapping combinatorial optimization problems to quantum computers.
      * **Dependencies**: This script also relies on `qiskit`, `numpy`, `scipy`, and the `qiskit-ibm-runtime` library.
      * **Usage**:

    <!-- end list -->

    ```bash
    python quantum_tsp_naive.py
    ```

  * `cluster_complete_graphs.json`:

      * **Purpose**: This file serves as the input data source for both Python scripts. It contains pre-calculated distance matrices for various VRP instances (clusters), which are used to define the cost function that the QAOA algorithm aims to minimize.

-----

## Workflow

1.  **Preparation**: Ensure all three files (`quantum_vrp.py`, `quantum_tsp_naive.py`, and `cluster_complete_graphs.json`) are in the same directory.
2.  **Configuration**: Open either of the Python scripts and adjust the `cluster_type` and `target_id` variables to select the specific VRP instance you want to solve from the `cluster_complete_graphs.json` file.
3.  **Run Solver**: Execute one of the scripts from your terminal. The script will:
      * Load the specified data.
      * Connect to an IBM Quantum backend (or a local simulator if not available).
      * Use a classical optimizer to find the optimal parameters for the QAOA circuit.
      * Run the optimized circuit on the selected backend for a number of `SHOTS`.
      * Process the results and log the counts, costs, and classical benchmarks to a new or existing `out.csv` file.

-----

## Notes

  * **Qubit Allocation**: The two scripts demonstrate two different qubit allocation strategies for the same problem. The `quantum_vrp.py` script's **"node-visit-time"** encoding is generally more scalable as it uses fewer qubits than the **"naive adjacency"** encoding in `quantum_tsp_naive.py`, especially as the number of cities increases.
  * **Classical Benchmarking**: Both scripts perform a brute-force classical search to find the true minimum, average, and maximum costs for the given problem instance. This allows you to evaluate how close the quantum algorithm's result is to the optimal classical solution.
  * **Measurement Counts**: The name of this directory, **"12\_measurement\_count"**, highlights the core output of these algorithms. The final result is a distribution of measurement counts, where the highest count corresponds to the most probable solution found by the quantum computer.