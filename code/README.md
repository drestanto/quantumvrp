# Quantum Optimization for the Vehicle Routing Problem (VRP)

This repository contains the implementation and experimental results for a research project focused on applying quantum optimization techniques, specifically the Quantum Approximate Optimization Algorithm (QAOA), to solve instances of the Vehicle Routing Problem (VRP). The project uses real-world waste collection data from the City of Casey, Australia, as a practical case study.

---

## Project Overview

The Vehicle Routing Problem (VRP) is a fundamental combinatorial optimization challenge in logistics. Given its NP-hard nature, classical algorithms often struggle with large-scale instances. This project explores the potential of quantum computing to find efficient approximate solutions to VRP by leveraging quantum algorithms.

The project pipeline is structured into several distinct phases, each represented by a dedicated directory, ensuring a clear and organized workflow from raw data acquisition to quantum solution implementation.

---

## Directory Structure and Workflow

The project is organized into sequentially numbered directories, reflecting the data processing and experimentation pipeline:

* `00_experiments/`:
    * **Purpose**: Contains initial exploratory data analysis (EDA) and visualization scripts (e.g., Jupyter notebooks using `folium`) to understand the raw spatial data.
    * **Output**: Interactive HTML maps for visual inspection of geographical data.

* `0_raw data/`:
    * **Purpose**: Stores the original, raw datasets (e.g., CSV files from City of Casey) and the `clean.py` script for their initial preprocessing.
    * **Output**: Cleaned and standardized CSV files (e.g., `*-clean.csv`).

* `1_cleaned data/`:
    * **Purpose**: Holds the cleaned data from the previous step and the `isolate.py` script, which extracts and formats only the essential spatial information.
    * **Output**: Spatial-only CSV files (e.g., `*-clean-spatial.csv`), ready for graph construction.

* `2_isolated data/`:
    * **Purpose**: Contains the spatial-only data and the `cluster.py` script, which performs initial clustering operations (e.g., assigning litter bins to waste collection areas, and waste areas to facilities).
    * **Output**: JSON files detailing clustering assignments (e.g., `public_litter_bins_to_waste_collection_areas_clustering.json`).

* `3_clustered data/`:
    * **Purpose**: Stores the results of the clustering and the `complete_grapher.py` script. This script builds a detailed road network graph, maps all relevant points to it, and generates complete distance matrices for each identified cluster.
    * **Output**: A single JSON file (`cluster_complete_graphs.json`) containing all calculated distance matrices.

* `4_completed graph data/`:
    * **Purpose**: Provides utilities for inspecting the generated complete graphs and includes a basic classical (brute-force) VRP solver (`naive_vrp.py`) for small clusters. This serves as a classical benchmark.
    * **Output**: Console output of distance matrices and classical VRP solutions.

* `5_quantum vrp_error/`:
    * **Purpose**: A legacy directory preserving an early, unsuccessful attempt at developing a quantum VRP solver. It documents challenges and lessons learned.
    * **Note**: This folder is kept for historical context and does not contain functional quantum VRP code.

* `6_quantum vrp_initial_approach/`:
    * **Purpose**: Contains the initial successful implementation of a QAOA-based VRP solver. It includes scripts for API testing (`quantum_api_test.py`), the main QAOA solver (`quantum_vrp.py`), and a sample output log (`sample.out`).
    * **Output**: Console output of quantum optimization progress, optimal QAOA parameters, and the decoded quantum VRP route.

---

## Getting Started

To set up and run this project:

1.  **Clone the Repository**:
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```
2.  **Data Acquisition**: Obtain the raw data files from the City of Casey Open Data Portal and other specified sources (as detailed in the `experiments/README.md` or project documentation) and place them in the `0_raw data/` directory.
3.  **Python Environment**: It is highly recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    ```
4.  **Install Dependencies**: Install all required Python libraries. You can find specific dependencies listed in the `README.md` files of individual sub-directories, but common ones include:
    ```bash
    pip install pandas numpy folium shapely networkx scipy qiskit qiskit-ibm-runtime
    ```
5.  **IBM Quantum Account**: For the quantum components (`6_quantum vrp_initial_approach/`), you will need an IBM Quantum account and API token. Configure your token using `quantum_api_test.py` or by setting it up directly in `quantum_vrp.py`.
6.  **Follow Directory Order**: Proceed through the directories sequentially (from `0_raw data/` to `6_quantum vrp_initial_approach/`), running the scripts as instructed in each directory's `README.md` to build the data pipeline step-by-step.

---

## Research Focus

This project highlights:

* The practical challenges of transforming real-world geospatial data into a VRP formulation suitable for quantum algorithms.
* The implementation of QAOA with a "Node-Visit-Time Encoding" strategy for VRP.
* A comparative analysis framework against classical VRP solutions.
* The current capabilities and limitations of NISQ (Noisy Intermediate-Scale Quantum) devices for combinatorial optimization problems.

---

## Contact

For any questions or further information, please refer to the individual `README.md` files within each sub-directory or the main project documentation.
