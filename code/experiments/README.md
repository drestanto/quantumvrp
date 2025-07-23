# experiments/

This directory serves as a repository for experimental code and initial data analysis related to the Quantum Optimization for the Vehicle Routing Problem (VRP) project. It contains Jupyter notebooks for data visualization and the raw data files used in these experiments.

---

## Contents

* `Visual Map.ipynb`: A Jupyter Notebook containing Python code for visualizing the spatial data on interactive maps using `folium`. This notebook generates HTML map files that help in understanding the geographical distribution of waste facilities, collection areas, public litter bins, and road networks in the City of Casey.
* **Data Files**: This folder is expected to contain the following CSV and ZIP files, which are essential for running the `Visual Map.ipynb` notebook:
    * `public-litter-bins.csv`
    * `waste-collection-area.csv`
    * `waste-facility-locations.csv`
    * `road-responsibility.csv`
    * `traffic-volume-survey-copy.csv` (for constraints)
    * `rainfall-data.csv` (for constraints)
    * `caseylga_boundary.zip` (for elevation constraints)
    * `Order_29EYKH.zip` (for elevation constraints)

---

## How to Use

To run the experiments and generate the visualizations:

1.  **Place Data Files**: Ensure all the data files listed above are present in this `experiments/` directory alongside the `Visual Map.ipynb` notebook.
2.  **Install Dependencies**: If you haven't already, install the necessary Python libraries:
    ```bash
    pip install pandas folium
    ```
3.  **Open Jupyter Notebook**: Launch Jupyter Notebook or JupyterLab from your terminal in the parent directory of `experiments/` or directly within `experiments/`:
    ```bash
    jupyter notebook
    ```
    Then, navigate to and open `Visual Map.ipynb`.
4.  **Run Cells**: Execute the cells in the `Visual Map.ipynb` notebook sequentially. This will:
    * Load the spatial data.
    * Generate various interactive HTML maps (e.g., `map.html`, `combined_map.html`, `map_with_roads.html`, `combined_map_with_toggle.html`).
    * These HTML files will be saved directly into this `experiments/` directory.
5.  **View Maps**: Open the generated `.html` files in your web browser to view the interactive maps.

---

## Purpose

The visualizations generated from `Visual Map.ipynb` help in:

* **Data Validation**: Visually confirming the accuracy and completeness of the spatial datasets.
* **Understanding Spatial Relationships**: Gaining insights into the geographical layout of depots, customer nodes, and road networks.
* **Initial Problem Formulation**: Aiding in the conceptualization of the VRP graph by providing a clear visual representation of the nodes and potential edges.

This `experiments/` folder is designed for quick iterations and visual checks during the initial phases of the project.