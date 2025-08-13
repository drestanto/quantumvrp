import pandas as pd
import numpy as np
import json
from scipy.spatial import KDTree
import os
import rasterio

# --- Configuration and File Paths ---
DELIMITER = ';'

# Input directory and file prefix for the split road network data
SPLIT_ROAD_DIR = '../working'
SPLIT_ROAD_PREFIX = 'road-network-part'
NUM_PARTS = 80

# New Input data for constraints
TRAFFIC_VOLUME_FILE = 'traffic-volume-survey-copy.csv'
ELEVATION_DATA_SHP = 'EXTRACT_POLYGON.shp'
RAINFALL_DATA_FILE = 'rainfall-data.csv'

# Output directory and prefix for the pre-processed files
OUTPUT_DIR = '../working2'
OUTPUT_PREFIX = 'preprocessed-road-network-part'

# --- Helper Functions ---

def load_data(file_path):
    """Loads a CSV file into a pandas DataFrame."""
    try:
        return pd.read_csv(file_path, delimiter=DELIMITER)
    except FileNotFoundError:
        print(f"Error: Required file not found: {file_path}. Skipping.")
        return None
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance

def _get_elevation_from_data(lat, lon, elevation_data):
    if elevation_data is None or 'raster_src' not in elevation_data or elevation_data['raster_src'] is None:
        return 0.0
    src = elevation_data['raster_src']
    try:
        for val in src.sample([(lon, lat)]):
            return float(val[0])
    except:
        return 0.0
    return 0.0

def _apply_traffic_factor(current_weight_km, start_lat, start_lon, traffic_data, traffic_kdtree):
    if traffic_data is None or traffic_data.empty or traffic_kdtree is None:
        return current_weight_km
    try:
        dist_to_traffic, traffic_idx = traffic_kdtree.query([start_lat, start_lon], k=1)
        traffic_factor = 1.0
        if traffic_idx < len(traffic_data):
            traffic_volume_24h = traffic_data.iloc[traffic_idx]['Volume in 24Hours']
            if traffic_volume_24h > 10000:
                traffic_factor = 1.30
            elif traffic_volume_24h > 5000:
                traffic_factor = 1.15
            elif traffic_volume_24h > 1000:
                traffic_factor = 1.05
        return current_weight_km * traffic_factor
    except:
        return current_weight_km

def _apply_rainfall_factor(current_weight_km, start_lat, start_lon, rainfall_df_for_kdtree, rainfall_kdtree):
    if rainfall_df_for_kdtree is None or rainfall_df_for_kdtree.empty or rainfall_kdtree is None:
        return current_weight_km
    try:
        dist_to_rainfall, rainfall_idx = rainfall_kdtree.query([start_lat, start_lon], k=1)
        rainfall_factor = 1.0
        if rainfall_idx < len(rainfall_df_for_kdtree):
            daily_rainfall_mm = rainfall_df_for_kdtree.iloc[rainfall_idx]['Value (mm)']
            if daily_rainfall_mm > 20.0:
                rainfall_factor = 1.25
            elif daily_rainfall_mm > 5.0:
                rainfall_factor = 1.10
            elif daily_rainfall_mm > 0.0:
                rainfall_factor = 1.02
        return current_weight_km * rainfall_factor
    except:
        return current_weight_km

def _apply_elevation_factor(current_weight_km, start_lat, start_lon, end_lat, end_lon, elevation_data):
    if elevation_data is None or 'raster_src' not in elevation_data or elevation_data['raster_src'] is None:
        return current_weight_km
    try:
        elev_start = _get_elevation_from_data(start_lat, start_lon, elevation_data)
        elev_end = _get_elevation_from_data(end_lat, end_lon, elevation_data)
        delta_elevation = abs(elev_end - elev_start)
        elevation_factor = 1 + (delta_elevation * 0.001)
        return current_weight_km * elevation_factor
    except:
        return current_weight_km

def calculate_modified_edge_weight(row, traffic_data, traffic_kdtree, rainfall_df_for_kdtree, rainfall_kdtree, elevation_data):
    """Combines all factors into a single weight calculation for a DataFrame row."""
    modified_weight = haversine(row['START_LAT'], row['START_LON'], row['END_LAT'], row['END_LON'])
    modified_weight = _apply_traffic_factor(modified_weight, row['START_LAT'], row['START_LON'], traffic_data, traffic_kdtree)
    modified_weight = _apply_rainfall_factor(modified_weight, row['START_LAT'], row['START_LON'], rainfall_df_for_kdtree, rainfall_kdtree)
    modified_weight = _apply_elevation_factor(modified_weight, row['START_LAT'], row['START_LON'], row['END_LAT'], row['END_LON'], elevation_data)
    return modified_weight

# --- Main Execution ---
def main():
    print("Starting pre-processing of split road network files.")

    # 1. Load constraint data once
    print("Loading constraint data...")
    traffic_data = load_data(TRAFFIC_VOLUME_FILE)
    rainfall_data = load_data(RAINFALL_DATA_FILE)

    elevation_data = {'raster_src': None}
    try:
        elevation_data['raster_src'] = rasterio.open(ELEVATION_DATA_SHP)
    except Exception as e:
        print(f"Warning: Could not open elevation data file: {e}. Elevation factor will not be applied.")

    traffic_kdtree = None
    if traffic_data is not None and not traffic_data.empty:
        traffic_kdtree = KDTree(traffic_data[['Latitude Coordinate', 'Longitude Coordinate']].values)

    rainfall_df_for_kdtree = None
    rainfall_kdtree = None
    if rainfall_data is not None and not rainfall_data.empty:
        rainfall_df_for_kdtree = rainfall_data.copy()
        if 'geolocation' in rainfall_df_for_kdtree.columns and 'Latitude' not in rainfall_df_for_kdtree.columns:
            rainfall_df_for_kdtree[['Latitude', 'Longitude']] = rainfall_df_for_kdtree['geolocation'].str.split(', ', expand=True).astype(float)
        if 'Latitude' in rainfall_df_for_kdtree.columns and 'Longitude' in rainfall_df_for_kdtree.columns:
            rainfall_kdtree = KDTree(rainfall_df_for_kdtree[['Latitude', 'Longitude']].values)
    
    # 2. Process each split file individually
    print(f"\nIterating through {NUM_PARTS} road network parts...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for i in range(1, NUM_PARTS + 1):
        input_file = os.path.join(SPLIT_ROAD_DIR, f"{SPLIT_ROAD_PREFIX}{i}.csv")
        output_file = os.path.join(OUTPUT_DIR, f"{OUTPUT_PREFIX}{i}.csv")
        
        road_part_df = load_data(input_file)
        
        if road_part_df is not None:
            print(f"  - Processing part {i}/{NUM_PARTS} with {len(road_part_df)} records...")
            
            # Apply the combined weight calculation to all rows
            road_part_df['modified_weight_km'] = road_part_df.apply(
                lambda row: calculate_modified_edge_weight(
                    row, traffic_data, traffic_kdtree, rainfall_df_for_kdtree, rainfall_kdtree, elevation_data
                ), axis=1
            )
            
            # Save the result
            road_part_df.to_csv(output_file, index=False, sep=DELIMITER)
            print(f"    -> Saved pre-processed data to '{output_file}'")

    print("\nAll pre-processing complete.")

    # Close the elevation raster source
    if elevation_data['raster_src']:
        elevation_data['raster_src'].close()

if __name__ == "__main__":
    main()