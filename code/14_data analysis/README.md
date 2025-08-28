# 14\_data\_analysis/

This directory contains the Jupyter Notebook and output data for analyzing the performance of a **Quantum Approximate Optimization Algorithm (QAOA)** on the Vehicle Routing Problem (VRP). The analysis aims to benchmark the quantum solver against classical solutions, providing critical insights into its efficiency, accuracy, and resilience to noise.

-----

## Contents

  * `Data Analysis.ipynb`:

      * **Purpose**: This Jupyter Notebook contains the complete data analysis workflow. It includes data loading, preprocessing, metric calculations, and visualizations, directly implementing the seven key calculations outlined in the `Data_Analysis_Write_Up.pdf` document.
      * **Sections Covered**:
        1.  **Optimality Ratio**: Calculates how closely the quantum solution approximates the true optimal classical solution.
        2.  **Invalid Bitstring Measurements**: Analyzes the occurrence of invalid solutions from the quantum hardware.
        3.  **Summary of Qubit Allocation Techniques**: Compares the overall performance of `naive_adjacency` and `node_visit_time` qubit allocation methods.
        4.  **Comparison Between Shots**: Investigates the impact of varying the number of shots on the optimality ratio for specific instances.
        5.  **Comparison of QAOA Depth**: Examines how different QAOA depths affect the optimality ratio for specific instances.
        6.  **Cost Distribution Analysis**: Visualizes the distribution of measured costs for the `node_visit_time` qubit allocation technique.
        7.  **Scaling Analysis**: Studies the relationship between the optimality ratio and the number of nodes (problem size) for selected instances.

  * `out.csv`:

      * **Purpose**: This CSV file serves as the primary output for **quantum VRP experiment results**. It consolidates data including qubit allocation, cluster type, QAOA depth, number of shots, measured bitstrings, their counts, associated costs, and merged classical benchmark costs.

  * `out_classical.csv`:

      * **Purpose**: This CSV file contains the **classical brute-force VRP benchmarks**. It stores the minimum, average, and maximum travel costs, along with the number of nodes and processing time, for various VRP instances.

  * `Data_Analysis_Write_Up.pdf`:

      * **Purpose**: This document outlines the theoretical framework and comprehensive plan for the data analysis. It details the methodologies for calculations and the expected insights derived from evaluating the quantum VRP algorithm.

-----

## Workflow

1.  **Preparation**:
      * Ensure that `out.csv` and `out_classical.csv` (the results from the quantum and classical VRP runners) are present in the same directory as `Data Analysis.ipynb`.
2.  **Install Dependencies**: Install the necessary Python libraries for data manipulation and visualization:
    ```bash
    pip install pandas numpy seaborn matplotlib
    ```
3.  **Run Analysis**: Open and execute the Jupyter Notebook to perform the analysis:
    ```bash
    jupyter notebook "Data Analysis.ipynb"
    ```
    The notebook is structured with cells corresponding to each analysis step, which can be run sequentially to reproduce the results and visualizations.

-----

## Notes

  * **Qubit Allocation**: The original VRP experiments utilized two distinct qubit allocation strategies:
      * **"node-visit-time"**: Generally more scalable, requiring fewer qubits as problem size increases, with a scaling of $\\mathcal{O}((N-1) \\log\_2(N-1))$.
      * **"naive adjacency"** (QUBO): A standard QUBO formulation requiring $N^2$ qubits for $N$ cities, which can be less qubit-efficient for larger problems.
  * **Classical Benchmarking**: The `out_classical.csv` data provides **definitive classical benchmarks** obtained from a full brute-force search. This serves as a crucial baseline for evaluating the quantum algorithm's performance against the true optimal solution.
  * **Measurement Counts**: The fundamental output of the quantum algorithms, as reflected in `out.csv`, is a distribution of measurement counts. The highest count generally corresponds to the most probable (and ideally optimal) solution found by the quantum computer, guided by classical optimization of QAOA parameters.