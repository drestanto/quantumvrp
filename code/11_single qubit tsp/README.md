# 11\_single\_qubit\_tsp/

This directory contains `quantum_tsp_single.py`, a script that reproduces the research presented in the paper "A single-qubit quantum algorithm for the Traveling Salesman Problem". The script implements a novel approach to the Traveling Salesman Problem (TSP) by mapping it onto a single qubit, in contrast to the more conventional multi-qubit Quadratic Unconstrained Binary Optimization (QUBO) formulations.

-----

## Contents

  * `quantum_tsp_single.py`:
      * **Purpose**: This script solves a TSP instance for a single cluster by using a single-qubit quantum algorithm. It loads a distance matrix and then models the problem as a search for an optimal sequence of single-qubit unitary operations. A classical optimizer (`scipy.optimize.minimize`) is used to find the parameters for the unitary gates that minimize the total route distance, effectively finding a near-optimal route.
      * **Configuration**: Before running, you **must** edit the `cluster_type` and `target_id` variables within the script to specify which cluster from `cluster_complete_graphs.json` you want to solve.
      * **Dependencies**: This script relies on `qiskit`, `numpy`, and `scipy`.
      * **Usage**:
    <!-- end list -->
    ```bash
    python quantum_tsp_single.py
    ```

-----

## Workflow

1.  **Preparation**: Ensure the `cluster_complete_graphs.json` file is in the same directory.
2.  **Configure Solver**: Open `quantum_tsp_single.py` and set the `cluster_type` and `target_id` for the TSP instance you want to solve.
3.  **Run Solver**: Execute `python quantum_tsp_single.py`. The script will use a classical optimizer to find the parameters for the single-qubit circuit and output the best route found.

-----

## Notes

  * **Single-Qubit Approach**: This method is based on the idea of representing the permutations of a TSP tour as rotations on the Bloch sphere. By performing a series of unitary operations, the algorithm evolves a quantum state, and the final state's measurement corresponds to a particular tour. The classical optimizer then tunes the parameters of these operations to minimize the tour's total distance.
  * **Research Reproduction**: This script is a direct implementation of the research paper "A single-qubit quantum algorithm for the Traveling Salesman Problem". This approach is an alternative to the more common QAOA-based methods that require `N^2` qubits for a problem with `N` cities, making it potentially more scalable for current and near-future quantum hardware.
  * **Fake Backend**: The script uses Qiskit's `FakeProvider` (`GenericBackendV2`) to simulate the quantum device, allowing for quick testing without using a real quantum computer. This is a good practice for validating the algorithm's logic before moving to more computationally expensive real hardware runs.