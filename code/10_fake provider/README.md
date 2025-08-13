# 10\_fake\_provider/

This directory contains an updated version of the `quantum_vrp.py` script. The primary change is the replacement of the `QiskitRuntimeService` with a local, in-memory quantum backend provided by Qiskit's `FakeProvider`. This allows for rapid prototyping, debugging, and testing of the Quantum Approximate Optimization Algorithm (QAOA) for the Vehicle Routing Problem (VRP) without consuming real quantum cloud time or incurring costs.

-----

## Contents

  * `quantum_vrp.py`:
      * **Purpose**: This is the main script that implements the QAOA for a specific VRP instance. Unlike the previous version, this script uses `GenericBackendV2` from `qiskit.providers.fake_provider` as the quantum backend. It loads VRP data, allocates qubits, constructs a parameterized QAOA circuit, and uses a classical optimizer (`scipy.optimize.minimize` with `COBYLA`) to find the optimal QAOA parameters. The circuit is then run on the fake backend to obtain measurement results, which are decoded into an optimal VRP route and its total cost.
      * **Configuration**: Before running, you **must** edit the `cluster_type` and `target_id` variables within the script to specify which VRP cluster from `cluster_complete_graphs.json` you want to solve. You can also adjust QAOA parameters like the number of layers (`QAOA_P`) and the number of shots (`SHOTS`).
      * **Dependencies**: This script relies on `qiskit`, `numpy`, and `scipy`.
      * **Usage**:
    <!-- end list -->
    ```bash
    python quantum_vrp.py
    ```

-----

## Workflow

1.  **Preparation**: Ensure the `cluster_complete_graphs.json` file is present in the same directory.
2.  **Configure Solver**: Open `quantum_vrp.py` and set the `cluster_type` and `target_id` for the VRP instance you wish to solve.
3.  **Run QAOA Solver**: Execute `python quantum_vrp.py`. The script will run locally and quickly, outputting the optimization process, the optimal parameters, and the final decoded route.

-----

## Notes

  * **Fake Provider**: Qiskit's `FakeProvider` provides a set of simulators that mimic the behavior of real quantum devices but run on a classical computer. This allows for a fast development and testing cycle, as there are no queue times or resource limitations. It is an excellent tool for verifying algorithm logic before moving to real quantum hardware.
  * **Performance**: The script's performance is significantly faster compared to running on a real quantum computer, making it ideal for iterating on the QAOA implementation and parameter tuning.