# 9\_quantum\_tsp/

This directory contains `quantum_tsp_naive.py`, a script that implements a Quantum Approximate Optimization Algorithm (QAOA) to solve the Traveling Salesman Problem (TSP) by reformulating it as a Quadratic Unconstrained Binary Optimization (QUBO) problem. This approach demonstrates a method for finding an approximate solution to a classically difficult combinatorial optimization problem using a hybrid quantum-classical workflow.

-----

## Contents

  * `quantum_tsp_naive.py`:
      * **Purpose**: This script serves as the main program for solving a specific TSP instance for a single cluster. It loads a pre-computed distance matrix from `cluster_complete_graphs.json`, which represents the cities and their distances. It then encodes this problem into a QUBO formulation and uses a classical optimizer to find the optimal parameters for a QAOA circuit. After running the optimized circuit on a quantum backend (or simulator), the script decodes the most probable bitstring to determine the optimal route and its total cost.
      * **Configuration**: Before running, you **must** edit the `cluster_type` and `target_id` variables within the script to specify which cluster you want to solve. You can also adjust the QAOA parameters, such as the number of layers (`QAOA_P`) and the number of shots for measurement (`SHOTS`).
      * **Dependencies**: This script relies on `qiskit-ibm-runtime`, `qiskit`, `numpy`, `scipy`, and `json`.
      * **Usage**:
    <!-- end list -->
    ```bash
    python quantum_tsp_naive.py
    ```
    Be aware that this implementation is designed for demonstration and might require significant computational resources or queue times if run on a real quantum device. The script is configured to fall back to `AerSimulator` if a real backend is unavailable or encounters issues.

-----

## Workflow

1.  **Preparation**: Ensure the `cluster_complete_graphs.json` file is present in the same directory.
2.  **Configure Solver**: Open `quantum_tsp_naive.py` and set the `cluster_type` and `target_id` for the TSP instance you wish to solve.
3.  **Run QAOA Solver**: Execute `python quantum_tsp_naive.py`. The script will output the optimization process, the optimal parameters found, and the final decoded route.

-----

## Notes

  * **QUBO Formulation**: The script uses a QUBO formulation, which maps the TSP's constraints (visiting each city exactly once) and objectives (minimizing total distance) into a binary quadratic model that can be solved by a quantum computer. The number of qubits required for this approach is `N^2`, where `N` is the number of cities, highlighting the current hardware limitations for larger problems.
  * **Approximation**: QAOA is an **approximate** optimization algorithm. While it aims to find high-quality solutions, it does not guarantee the absolute optimal solution, especially with a limited number of QAOA layers (`QAOA_P`) or optimization iterations.
  * **Hybrid Approach**: The solution process is a hybrid of classical and quantum computing. A classical optimizer is used to find the best parameters for the quantum circuit, which then runs on the quantum computer to provide the final solution. This is a common strategy for current quantum algorithms.

-----

  * **Citations:**
      * **[1]** "Traveling Salesman Problem using Quantum Computing" by Anish Bhatt, published in *The Quantastic Journal* on Medium ([https://medium.com/the-quantastic-journal/traveling-salesman-problem-using-quantum-computing-02ae6356544b](https://medium.com/the-quantastic-journal/traveling-salesman-problem-using-quantum-computing-02ae6356544b)).
      * **[2]** The QAOA is a variational quantum algorithm for finding approximate solutions to combinatorial optimization problems. It works by using a classical optimizer to find the best parameters for a quantum circuit to find the ground state of a Hamiltonian representing the problem.
      * **[3]** The `cluster_complete_graphs.json` file contains pre-computed distance matrices for different clusters, which serve as the input data for the VRP/TSP solvers.