import pandas as pd
import os

def digest_to_spatial(input_filename, output_filename, is_road_network=False, delimiter=';'):
    """
    Digests a cleaned CSV file into a spatial-only version.
    For point data, it extracts ID, Latitude, Longitude.
    For road network data, it extracts road segments (edges) with start/end node coordinates.

    Args:
        input_filename (str): The path to the input cleaned CSV file.
        output_filename (str): The path to save the new spatial CSV file.
        is_road_network (bool, optional): True if the input is the road network file,
                                          False for point data files. Defaults to False.
        delimiter (str, optional): The delimiter used for reading and writing CSVs. Defaults to ';'.
    """
    print(f"Digesting '{input_filename}' to spatial data...")
    try:
        df = pd.read_csv(input_filename, delimiter=delimiter) # Read with semicolon delimiter
    except FileNotFoundError:
        print(f"Error: Input file '{input_filename}' not found. Please ensure the '*-clean.csv' files exist. Skipping.")
        return

    if is_road_network:
        # Process road network data to create segments (edges)
        if 'LINE_ID' not in df.columns or 'POINT_ORDER' not in df.columns or \
           'Latitude' not in df.columns or 'Longitude' not in df.columns:
            print(f"Error: Required columns (LINE_ID, POINT_ORDER, Latitude, Longitude) not found in '{input_filename}'. Skipping road network digestion.")
            return

        road_segments = []
        segment_id_counter = 0

        # Group by LINE_ID to process each original road line sequentially
        for line_id, group in df.groupby('LINE_ID'):
            # Sort points by POINT_ORDER to ensure correct segment creation
            group = group.sort_values(by='POINT_ORDER').reset_index(drop=True)

            # Create segments from consecutive points within each LINE_ID
            for i in range(len(group) - 1):
                segment_id_counter += 1
                start_point = group.iloc[i]
                end_point = group.iloc[i+1]
                road_segments.append({
                    'ROAD_SEGMENT_ID': segment_id_counter,
                    'LINE_ID': line_id,
                    'START_LAT': start_point['Latitude'],
                    'START_LON': start_point['Longitude'],
                    'END_LAT': end_point['Latitude'],
                    'END_LON': end_point['Longitude']
                })
        
        spatial_df = pd.DataFrame(road_segments)
        
    else:
        # Process point data (litter bins, waste collection areas, waste facilities)
        if 'Latitude' not in df.columns or 'Longitude' not in df.columns:
            print(f"Error: 'Latitude' or 'Longitude' column not found in '{input_filename}'. Skipping point data digestion.")
            return

        spatial_df = df[['Latitude', 'Longitude']].copy()
        # Add a simple sequential ID for each point
        spatial_df.insert(0, 'ID', range(1, 1 + len(spatial_df)))

    # Save the digested spatial DataFrame using semicolon as delimiter
    spatial_df.to_csv(output_filename, index=False, sep=delimiter)
    print(f"  Spatial data saved to '{output_filename}' with {len(spatial_df)} rows.")

# Define the cleaned input files and their corresponding spatial output files
# and whether they represent a road network.
files_to_digest = {
    "public-litter-bins-clean.csv": {"output": "public-litter-bins-clean-spatial.csv", "is_road_network": False},
    "road-responsibility-clean.csv": {"output": "road-responsibility-clean-spatial.csv", "is_road_network": True},
    "waste-collection-area-clean.csv": {"output": "waste-collection-area-clean-spatial.csv", "is_road_network": False},
    "waste-facility-locations-clean.csv": {"output": "waste-facility-locations-clean-spatial.csv", "is_road_network": False},
}

# Run the digestion process for each file
print("Starting digestion of cleaned data to spatial-only versions...")
for input_file, config in files_to_digest.items():
    digest_to_spatial(input_file, config["output"], is_road_network=config["is_road_network"])

print("\nSpatial data digestion completed for all specified files.")
print("You now have the spatial-only files ready for your research.")