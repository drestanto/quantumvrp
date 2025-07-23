import pandas as pd
import numpy as np
import json
import os

# Haversine distance function (in kilometers)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in kilometers

    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance

# --- Part 1: Assign Public Litter Bins to Waste Collection Areas ---

print("Assigning Public Litter Bins to Waste Collection Areas...")

# Load data
try:
    bins_df = pd.read_csv('public-litter-bins-clean-spatial.csv', delimiter=';')
    waste_areas_df = pd.read_csv('waste-collection-area-clean-spatial.csv', delimiter=';')
except FileNotFoundError as e:
    print(f"Error: Missing one or more input spatial CSV files. Please ensure 'public-litter-bins-clean-spatial.csv' and 'waste-collection-area-clean-spatial.csv' exist. Error: {e}")
    bins_df = pd.DataFrame() # Create empty DataFrames to avoid further errors
    waste_areas_df = pd.DataFrame()


if not bins_df.empty and not waste_areas_df.empty:
    # Prepare lists to store assignments
    bin_to_waste_area_assignments = []

    # Iterate through each litter bin
    for idx_bin, row_bin in bins_df.iterrows():
        min_distance = float('inf')
        closest_waste_area_id = None

        # Calculate distance to each waste collection area
        for idx_area, row_area in waste_areas_df.iterrows():
            dist = haversine(row_bin['Latitude'], row_bin['Longitude'],
                             row_area['Latitude'], row_area['Longitude'])
            
            if dist < min_distance:
                min_distance = dist
                closest_waste_area_id = row_area['ID']
        
        if closest_waste_area_id is not None:
            bin_to_waste_area_assignments.append({
                'bin_id': row_bin['ID'],
                'assigned_waste_area_id': closest_waste_area_id,
                'distance_km': min_distance
            })

    # Group assignments by waste_area_id
    grouped_bins = {}
    for assignment in bin_to_waste_area_assignments:
        area_id = assignment['assigned_waste_area_id']
        if area_id not in grouped_bins:
            grouped_bins[area_id] = []
        grouped_bins[area_id].append(assignment['bin_id'])

    # Format output for JSON
    litter_bins_clustering_output = {
        "description": "Clustering of Public Litter Bins to the nearest Waste Collection Area centroid.",
        "assignments": [
            {
                "waste_collection_area_id": area_id,
                "public_litter_bin_ids": bins
            } for area_id, bins in grouped_bins.items()
        ]
    }

    # Save to JSON
    output_litter_bins_json_path = 'public_litter_bins_to_waste_collection_areas_clustering.json'
    with open(output_litter_bins_json_path, 'w') as f:
        json.dump(litter_bins_clustering_output, f, indent=4)
    print(f"Public Litter Bins clustering saved to '{output_litter_bins_json_path}'")
else:
    print("Skipping Public Litter Bins clustering due to missing input files or empty data.")


# --- Part 2: Assign Waste Collection Areas to Waste Facilities ---

print("\nAssigning Waste Collection Areas to Waste Facilities...")

# Load data
try:
    waste_areas_df = pd.read_csv('waste-collection-area-clean-spatial.csv', delimiter=';')
    facilities_df = pd.read_csv('waste-facility-locations-clean-spatial.csv', delimiter=';')
except FileNotFoundError as e:
    print(f"Error: Missing one or more input spatial CSV files. Please ensure 'waste-collection-area-clean-spatial.csv' and 'waste-facility-locations-clean-spatial.csv' exist. Error: {e}")
    waste_areas_df = pd.DataFrame() # Create empty DataFrames to avoid further errors
    facilities_df = pd.DataFrame()

if not waste_areas_df.empty and not facilities_df.empty:
    # Prepare lists to store assignments
    waste_area_to_facility_assignments = []

    # Iterate through each waste collection area
    for idx_area, row_area in waste_areas_df.iterrows():
        min_distance = float('inf')
        closest_facility_id = None

        # Calculate distance to each waste facility
        for idx_fac, row_fac in facilities_df.iterrows():
            dist = haversine(row_area['Latitude'], row_area['Longitude'],
                             row_fac['Latitude'], row_fac['Longitude'])
            
            if dist < min_distance:
                min_distance = dist
                closest_facility_id = row_fac['ID']
        
        if closest_facility_id is not None:
            waste_area_to_facility_assignments.append({
                'waste_area_id': row_area['ID'],
                'assigned_facility_id': closest_facility_id,
                'distance_km': min_distance
            })

    # Group assignments by facility_id
    grouped_areas = {}
    for assignment in waste_area_to_facility_assignments:
        facility_id = assignment['assigned_facility_id']
        if facility_id not in grouped_areas:
            grouped_areas[facility_id] = []
        grouped_areas[facility_id].append(assignment['waste_area_id'])

    # Format output for JSON
    waste_collection_areas_clustering_output = {
        "description": "Clustering of Waste Collection Areas to the nearest Waste Facility location.",
        "assignments": [
            {
                "waste_facility_id": facility_id,
                "waste_collection_area_ids": areas
            } for facility_id, areas in grouped_areas.items()
        ]
    }

    # Save to JSON
    output_waste_areas_json_path = 'waste_collection_areas_to_waste_facilities_clustering.json'
    with open(output_waste_areas_json_path, 'w') as f:
        json.dump(waste_collection_areas_clustering_output, f, indent=4)
    print(f"Waste Collection Areas clustering saved to '{output_waste_areas_json_path}'")
else:
    print("Skipping Waste Collection Areas clustering due to missing input files or empty data.")

print("\nClustering process completed.")