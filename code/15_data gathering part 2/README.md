# 13\_data\_gathering/

This directory contains scripts and data for solving the Vehicle Routing Problem (VRP) using both classical brute-force methods and the Quantum Approximate Optimization Algorithm (QAOA). It also includes a robust system for batch processing and benchmarking these approaches.

-----

## Contents

  * `cluster_complete_graphs.json`:

      * **Purpose**: This JSON file is the **input data source** for all VRP calculations. It contains pre-calculated distance matrices for various VRP instances (clusters), categorized by `waste_facility` and `waste_collection_area` types. Each cluster includes a depot and a set of customer nodes.

  * `data_entries.csv`:

      * **Purpose**: This CSV file acts as the **input configuration for quantum VRP experiments**. Each row defines a specific QAOA run, including the qubit allocation method (`node_visit_time` or `naive_adjacency`), the target cluster, QAOA depth (`conf_qaoa_depth`), number of shots, and a `done` status.
      * **Usage**: Edit this file to configure your quantum experiments. Set `done` to `no` for new runs.

  * `data_entries_classical.csv`:

      * **Purpose**: This CSV file is the **input configuration for classical VRP benchmark calculations**. Each row defines a classical brute-force run, specifying the target cluster and a `done` status.
      * **Usage**: Edit this file to configure your classical benchmarks. Set `done` to `no` for new runs.

  * `vrp_quantum_runner.py`:

      * **Purpose**: This Python script executes **quantum VRP experiments** configured in `data_entries.csv`. It connects to an IBM Quantum backend (or a local simulator), uses a classical optimizer to find QAOA parameters, runs the quantum circuit, and processes the results.
      * **Features**:
          * Supports two qubit allocation methods: **"node-visit-time"** (more scalable) and **"naive adjacency"** (QUBO).
          * Performs classical optimization of QAOA parameters (gamma and beta).
          * Logs detailed output, including **COBYLA's gamma and beta parameter iterations**, to individual `.out` files.
          * Appends results to `out.csv` and creates timestamped checkpoint files (`out_<timestamp>.csv`) after each successful run.
          * Updates the `done` status in `data_entries.csv` upon completion.
      * **Usage**:
        ```bash
        python vrp_quantum_runner.py
        ```

  * `naive_vrp_runner.py`:

      * **Purpose**: This Python script performs **classical brute-force VRP calculations** based on configurations in `data_entries_classical.csv`. It calculates the true minimum, maximum, and average travel costs for each cluster.
      * **Features**:
          * Provides **definitive classical benchmarks** for evaluating quantum algorithm performance.
          * Logs detailed output to individual `.out` files.
          * Appends results (including the number of nodes) to `out_classical.csv`.
          * Updates the `done` status in `data_entries_classical.csv` upon completion.
      * **Usage**:
        ```bash
        python naive_vrp_runner.py
        ```

  * `out.csv`:

      * **Purpose**: The primary output CSV for **quantum VRP experiment results**. This file accumulates data (bitstrings, counts, costs, and classical benchmarks) from all `vrp_quantum_runner.py` runs.

  * `out_classical.csv`:

      * **Purpose**: The primary output CSV for **classical brute-force VRP benchmarks**. This file stores the minimum, average, and maximum travel costs, along with the number of nodes, calculated by `naive_vrp_runner.py`.

  * `out_<timestamp>.csv`:

      * **Purpose**: **Checkpoint files** for quantum experiment results. These are timestamped copies of `out.csv` created after each successful quantum run, allowing for data recovery and progressive data collection.

  * `*.out` files:

      * **Purpose**: Individual **log files** for each quantum (`vrp_quantum_runner.py`) and classical (`naive_vrp_runner.py`) experiment. These capture all print statements during a single run, providing detailed insights into execution, parameter iterations, and results.

-----

## Workflow

1.  **Preparation**:
      * Ensure `cluster_complete_graphs.json` is in the same directory.
      * Prepare your experiment configurations in `data_entries.csv` and `data_entries_classical.csv`, setting the `done` column to `no` for new runs.
2.  **Install Dependencies**: Install necessary Python libraries:
    ```bash
    pip install qiskit-ibm-runtime qiskit numpy scipy
    ```
3.  **Set up IBM Quantum Credentials (Optional)**: If you plan to use IBM Quantum hardware or cloud simulators, save your credentials:
    ```python
    from qiskit_ibm_runtime import QiskitRuntimeService
    QiskitRuntimeService.save_account(channel="ibm_quantum", token="YOUR_IBM_QUANTUM_TOKEN", instance="YOUR_INSTANCE")
    ```
    (Replace placeholders with your actual token and instance.) The `vrp_quantum_runner.py` script will automatically fall back to a local `AerSimulator` if these are not configured.
4.  **Run Classical Benchmarks**: Execute the classical runner first to establish ground truth data:
    ```bash
    python naive_vrp_runner.py
    ```
    This will process entries in `data_entries_classical.csv`, update `out_classical.csv`, and create `classical_<cluster_type>_<target_id>.out` log files.
5.  **Run Quantum Experiments**: Once classical benchmarks are complete, execute the quantum runner:
    ```bash
    python vrp_quantum_runner.py
    ```
    This will process entries in `data_entries.csv`, update `out.csv` (and checkpoint files), and create `quantum_<method>_<cluster_type>_<target_id>_p<depth>_s<shots>.out` log files.

-----

## Notes

  * **Qubit Allocation**: The project uses two distinct qubit allocation strategies:
      * **"node-visit-time"** (in `vrp_quantum_runner.py`): Generally more scalable, requiring fewer qubits, especially as the number of cities increases, with a scaling of $\\mathcal{O}((N-1) \\log\_2(N-1))$.
      * **"naive adjacency"** (QUBO, also in `vrp_quantum_runner.py`): A standard QUBO formulation that requires $N^2$ qubits for $N$ cities, which can be less qubit-efficient for larger problems.
  * **Classical Benchmarking**: The `naive_vrp_runner.py` script performs a full brute-force classical search to find the true minimum, average, and maximum costs for each problem instance. This provides a crucial and reliable baseline for evaluating how close the quantum algorithm's results are to the optimal classical solution.
  * **Measurement Counts**: The core output of the quantum algorithms is a distribution of measurement counts, where the highest count typically corresponds to the most probable (and ideally optimal) solution found by the quantum computer. The classical optimizer guides the QAOA circuit towards maximizing the probability of these optimal solutions.
  * **Batch Processing & Checkpointing**: The runner scripts (`vrp_quantum_runner.py` and `naive_vrp_runner.py`) are designed for robust, long-running experiments. They process configurations one by one, update the `done` status, and provide checkpoint files to mitigate data loss in case of interruptions.