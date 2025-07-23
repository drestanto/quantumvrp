# 0_raw data/

This directory is dedicated to storing the initial, raw datasets and the Python script (`clean.py`) responsible for their preliminary cleaning and preprocessing. The goal of this step is to standardize data formats, extract relevant geographical coordinates, and handle missing or malformed entries, preparing the data for subsequent analysis and integration into the VRP model.

---

## Contents

* `clean.py`: A Python script designed to clean and preprocess the raw CSV data files. It performs the following key operations:
    * Loads raw CSV files (e.g., `public-litter-bins.csv`, `waste-collection-area.csv`, `waste-facility-locations.csv`, `road-responsibility.csv`).
    * Extracts and converts latitude and longitude coordinates to numeric types.
    * Handles special cases for `waste-collection-area.csv` by calculating centroids from `Feature Extent` GeoJSON strings.
    * Processes `road-responsibility.csv` to extract all vertices from `Feature Extent` (LineString/MultiLineString GeoJSON).
    * Drops rows with missing or invalid coordinate data.
    * Removes duplicate rows.
    * Saves the cleaned data to new CSV files (e.g., `public-litter-bins-clean.csv`) with a semicolon delimiter.

* **Raw Data Files (Expected Inputs)**: This folder should contain the following original, raw CSV files for `clean.py` to process:
    * `public-litter-bins.csv`
    * `waste-collection-area.csv`
    * `waste-facility-locations.csv`
    * `road-responsibility.csv`
    * *(Note: Other constraint-related files like traffic, elevation, rainfall are not directly processed by `clean.py` but are part of the overall raw data collection.)*

* **Cleaned Data Files (Expected Outputs)**: After running `clean.py`, the following cleaned CSV files will be generated in this same directory:
    * `public-litter-bins-clean.csv`
    * `waste-collection-area-clean.csv`
    * `waste-facility-locations-clean.csv`
    * `road-responsibility-clean.csv`

---

## How to Use

To perform the data cleaning step:

1.  **Place Raw Data**: Ensure all the original raw CSV files (listed under "Raw Data Files") are present in this `0_raw data/` directory alongside the `clean.py` script.
2.  **Install Dependencies**: Make sure you have the necessary Python libraries installed. `clean.py` uses `pandas` and `shapely`.
    ```bash
    pip install pandas shapely
    ```
3.  **Run the Cleaning Script**: Open your terminal or command prompt, navigate to the `0_raw data/` directory, and execute the script:
    ```bash
    python clean.py
    ```
4.  **Verify Outputs**: After execution, new cleaned CSV files (e.g., `public-litter-bins-clean.csv`) will appear in this directory. Review these files to ensure the cleaning process was successful.

---

## Purpose

The `clean.py` script is a critical first step in the data pipeline for this project. Its primary purposes are:

* **Standardization**: To bring diverse raw datasets into a consistent and usable format.
* **Geospatial Preparation**: To extract and correctly format geographical coordinates (latitude and longitude) from various input representations, including GeoJSON strings.
* **Data Quality Assurance**: To remove incomplete or erroneous records, ensuring that subsequent analyses and model inputs are based on reliable data.
* **Foundation for VRP Graph**: To prepare the foundational spatial data required for constructing the VRP graph, which is then used by the quantum optimization algorithms.