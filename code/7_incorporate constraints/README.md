# 7\_incorporate\_constraints/

This directory contains the scripts used to preprocess and integrate real-world constraints—such as **traffic volume**, **elevation**, and **rainfall**—into the road network data. The goal is to create a more realistic and accurate representation of travel costs for the Vehicle Routing Problem (VRP). The workflow involves three main steps: splitting the large road network file, modifying the edge weights of each split file based on constraints, and then building complete graphs from these modified networks for VRP solving.

-----

## Contents

  * `split.py`:

      * **Purpose**: This script efficiently divides a large road network CSV file (`road-responsibility-clean-spatial.csv`) into multiple, smaller CSV files. This is a crucial first step, as processing the entire road network at once can be memory-intensive and time-consuming. By splitting the data, subsequent processing steps can be parallelized or handled in smaller, more manageable chunks.
      * **Usage**: The script is configured to split the data into 80 parts by default. You can adjust the `NUM_PARTS` variable if needed.

    <!-- end list -->

    ```bash
    python split.py
    ```

    The output files will be saved in the `../working` directory with a naming convention like `road-network-part1.csv`, `road-network-part2.csv`, etc..

  * `distancify.py`:

      * **Purpose**: This is the core script for integrating constraints. It iterates through each of the smaller road network files created by `split.py`. For each road segment (edge), it calculates a new, "**modified**" weight. This new weight is the original Haversine distance, but it's adjusted (or "penalized") by factors related to **traffic volume**, **elevation changes**, and **rainfall**. This results in a more realistic travel cost, where a 10 km route on a congested, hilly road with heavy rainfall would have a higher modified weight than a 10 km route on a flat, dry, and low-traffic road.
      * **Usage**: This script relies on the split files being present in the `../working` directory. It also requires the constraint data files (`traffic-volume-survey-copy.csv`, `EXTRACT_POLYGON.shp`, and `rainfall-data.csv`) to be in the same directory.

    <!-- end list -->

    ```bash
    python distancify.py
    ```

    The output files, which now include the `modified_weight_km` column, will be saved in the `../working2` directory with a prefix like `preprocessed-road-network-part`.

  * `complete_grapher.py`:

      * **Purpose**: This script serves as the final step in the data preparation pipeline. It reads the pre-processed road network files (from `distancify.py`) and builds a single, weighted `NetworkX` graph from them. It then loads the cluster assignments for the VRP and calculates the all-pairs shortest paths within each cluster using the modified edge weights. The result is a JSON file (`cluster_complete_graphs_with_constraints.json`) containing a complete graph for each cluster, where edge weights represent the most realistic travel costs based on the road network and applied constraints. This output is ready to be used by a VRP solver.
      * **Usage**: This script depends on the output of `distancify.py` and the clustering JSONs from a previous step.

    <!-- end list -->

    ```bash
    python complete_grapher.py
    ```

    The final JSON output file will be saved in the same directory.

-----

## Workflow

1.  **Split the Road Network**: Run `split.py` to break the large road network file into smaller, more manageable parts.
2.  **Incorporate Constraints**: Run `distancify.py` to calculate and add a new, more realistic travel weight (`modified_weight_km`) to each road segment, based on factors like traffic and elevation.
3.  **Build Complete Graphs**: Run `complete_grapher.py` to create a final, constraint-aware distance matrix for each VRP cluster.

-----

## Notes

  * **Dependencies**: The scripts rely on `pandas`, `numpy`, `scipy`, `networkx`, `rasterio`, and `os`. Ensure you have these libraries installed in your environment.
  * **Data Integrity**: The process assumes that all required input files (both the road network and the constraint data) are in their specified locations. Errors will be logged if files are not found.
  * **Sequential Execution**: The three scripts must be run in the correct order (`split.py` -\> `distancify.py` -\> `complete_grapher.py`) as each script's output serves as the input for the next.
  * **Error Handling**: The scripts include basic error handling to gracefully manage situations like missing files or unexpected data formats. Warnings are printed to the console if certain constraint factors cannot be applied.