import pandas as pd
import json
from shapely.geometry import shape, MultiPolygon, Polygon, LineString, MultiLineString
import os

def clean_and_save_csv(input_filename, output_filename, lat_col, lon_col, delimiter=';',
                       is_waste_collection_area=False, is_road_responsibility_file=False):
    """
    Cleans a CSV file by:
    1. Loading it into a pandas DataFrame.
    2. Extracting/converting latitude and longitude columns to numeric types.
    3. Dropping rows with missing latitude or longitude.
    4. Saving the cleaned data to a new CSV.

    Args:
        input_filename (str): The path to the input CSV file.
        output_filename (str): The path to save the cleaned CSV file.
        lat_col (str): The name of the latitude column in the input file.
        lon_col (str): The name of the longitude column in the input file.
        delimiter (str, optional): The delimiter used in the CSV. Defaults to ';'.
        is_waste_collection_area (bool, optional): Special handling for waste-collection-area.csv
                                                   to extract centroid from Feature Extent. Defaults to False.
        is_road_responsibility_file (bool, optional): Special handling for road-responsibility.csv
                                                      to extract all vertices from Feature Extent. Defaults to False.
    """
    print(f"Processing '{input_filename}'...")
    try:
        df = pd.read_csv(input_filename, delimiter=delimiter) # Reads with specified delimiter (default ';')
    except FileNotFoundError:
        print(f"Error: File '{input_filename}' not found. Skipping.")
        return

    # Standardize column names for consistency by stripping whitespace and replacing spaces with underscores
    df.columns = [col.strip().replace(' ', '_') for col in df.columns]

    df_cleaned = pd.DataFrame() # Initialize an empty DataFrame for the cleaned data

    if is_road_responsibility_file:
        # Special handling for 'road-responsibility.csv' to extract all vertices from Feature_Extent
        all_road_points = []
        # LINE_ID will now refer to the original row's index to group all segments of a MultiLineString
        for original_row_index, row in df.iterrows():
            try:
                geojson_str = row['Feature_Extent']
                # Ensure it's a string before attempting string operations
                if not isinstance(geojson_str, str):
                    print(f"Warning: 'Feature_Extent' at row {original_row_index} in {input_filename} is not a string. Value: '{geojson_str}'. Skipping.")
                    continue

                # Replace single quotes with double quotes for valid JSON parsing
                geojson_str = geojson_str.replace("'", "\"")
                geo_json_data = json.loads(geojson_str)

                geometry = shape(geo_json_data)

                # Prepare a dictionary of the original row's data to be carried over
                # Exclude Feature_Extent, Location_Coordinate, and original Latitude/Longitude Coordinates
                # as these will be regenerated or are redundant.
                cols_to_drop = ['Feature_Extent', 'Location_Coordinate', 'Latitude_Coordinate', 'Longitude_Coordinate']
                original_row_data = row.drop(labels=[col for col in cols_to_drop if col in row.index], errors='ignore').to_dict()

                # Helper function to process coordinates from a single LineString
                def add_line_string_coords(line_geom, current_line_id, current_point_order):
                    points_added = 0
                    for lon, lat in line_geom.coords: # Shapely coords are (longitude, latitude)
                        current_point_order += 1
                        point_data = {
                            **original_row_data, # Include all original data from the row
                            'LINE_ID': current_line_id,
                            'POINT_ORDER': current_point_order,
                            'Latitude': lat,
                            'Longitude': lon
                        }
                        all_road_points.append(point_data)
                        points_added += 1
                    return current_point_order, points_added

                current_line_id = original_row_index + 1 # Use 1-based index for LINE_ID
                point_order_in_line = 0 # To preserve the order of points within the current line/multiline

                if isinstance(geometry, LineString):
                    point_order_in_line, _ = add_line_string_coords(geometry, current_line_id, point_order_in_line)
                elif isinstance(geometry, MultiLineString):
                    for line_geom in geometry.geoms: # Iterate through each LineString in the MultiLineString
                        point_order_in_line, _ = add_line_string_coords(line_geom, current_line_id, point_order_in_line)
                else:
                    print(f"Warning: Unexpected geometry type '{geometry.geom_type}' for road segment at row {original_row_index} in {input_filename}. Expected LineString or MultiLineString. Skipping this segment.")

            except (json.JSONDecodeError, KeyError, IndexError, AttributeError) as e:
                print(f"Warning: Could not parse geometry for a road segment at row {original_row_index} in {input_filename}. Error: {e}. Problematic Feature_Extent (first 200 chars): '{str(row.get('Feature_Extent', 'N/A'))[:200]}...'. Skipping this segment.")

        df_cleaned = pd.DataFrame(all_road_points)
        # Ensure Latitude and Longitude are numeric and drop NaNs that might result from parsing errors
        df_cleaned['Latitude'] = pd.to_numeric(df_cleaned['Latitude'], errors='coerce')
        df_cleaned['Longitude'] = pd.to_numeric(df_cleaned['Longitude'], errors='coerce')
        original_points_count = len(df_cleaned)
        df_cleaned.dropna(subset=['Latitude', 'Longitude'], inplace=True)
        points_after_dropping_na = len(df_cleaned)
        print(f"  Removed {original_points_count - points_after_dropping_na} points with missing coordinates from road data.")

    elif is_waste_collection_area:
        # Special handling for 'waste-collection-area.csv' to extract centroid from Feature_Extent
        clean_coords = []
        for index, row in df.iterrows():
            current_feature_extent = row.get('Feature_Extent') # Use .get to avoid KeyError if column is missing
            try:
                if not isinstance(current_feature_extent, str) or pd.isna(current_feature_extent):
                    print(f"Warning: 'Feature_Extent' at row {index} in {input_filename} is not a valid string/is NaN. Value: '{current_feature_extent}'. Setting coordinates to None.")
                    clean_coords.append({'Latitude': None, 'Longitude': None})
                    continue

                geojson_str = str(current_feature_extent).replace("'", "\"")
                
                # Check if it looks like a coordinate array instead of a full GeoJSON object
                if not geojson_str.strip().startswith("{"):
                    # Attempt to wrap it as a Polygon. This assumes the string is just the coordinates array.
                    # This might be an oversimplification if it's not strictly an array.
                    geojson_str = f'{{"coordinates": {geojson_str}, "type": "Polygon"}}'
                
                geo_json_data = json.loads(geojson_str)
                geometry = shape(geo_json_data)

                if isinstance(geometry, MultiPolygon):
                    # For MultiPolygon, take the centroid of the first polygon for simplicity
                    polygon = geometry.geoms[0]
                elif isinstance(geometry, Polygon):
                    polygon = geometry
                else:
                    print(f"Warning: Unexpected geometry type '{geometry.geom_type}' at row {index} in {input_filename}. Expected Polygon or MultiPolygon. Setting coordinates to None.")
                    clean_coords.append({'Latitude': None, 'Longitude': None})
                    continue

                centroid = polygon.centroid # Centroid (x, y) where x is longitude, y is latitude
                clean_coords.append({'Latitude': centroid.y, 'Longitude': centroid.x})

            except (json.JSONDecodeError, KeyError, IndexError, AttributeError) as e:
                print(f"Warning: Could not parse geometry for row {index} in {input_filename}. Error: {e}. Problematic Feature_Extent (first 200 chars): '{str(current_feature_extent)[:200]}...'. Setting coordinates to None.")
                clean_coords.append({'Latitude': None, 'Longitude': None})

        coords_df = pd.DataFrame(clean_coords)
        
        # Ensure Latitude and Longitude columns are ready for assignment
        # Standardize column names before assigning
        lat_col_cleaned = lat_col.strip().replace(' ', '_')
        lon_col_cleaned = lon_col.strip().replace(' ', '_')

        if lat_col_cleaned in df.columns and lon_col_cleaned in df.columns:
            df.rename(columns={lat_col_cleaned: 'Original_Latitude', lon_col_cleaned: 'Original_Longitude'}, inplace=True)
        
        df['Latitude'] = coords_df['Latitude'] # Add new Latitude column (from centroid)
        df['Longitude'] = coords_df['Longitude'] # Add new Longitude column (from centroid)


        # Convert to numeric and drop NaNs
        df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
        df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')

        original_rows = len(df)
        df.dropna(subset=['Latitude', 'Longitude'], inplace=True)
        rows_after_dropping_na = len(df)
        print(f"  Removed {original_rows - rows_after_dropping_na} rows with missing coordinates.")
        df_cleaned = df # Keep all other original columns for this file
        
    else:
        # Generic handling for other files (public-litter-bins, waste-facility-locations)
        lat_col_cleaned = lat_col.strip().replace(' ', '_')
        lon_col_cleaned = lon_col.strip().replace(' ', '_')

        if lat_col_cleaned not in df.columns or lon_col_cleaned not in df.columns:
            print(f"Error: Required coordinate columns ('{lat_col}' and '{lon_col}') not found in '{input_filename}'. Found columns: {df.columns.tolist()}. Skipping.")
            return

        # Convert to numeric and rename for consistency
        df[lat_col_cleaned] = pd.to_numeric(df[lat_col_cleaned], errors='coerce')
        df[lon_col_cleaned] = pd.to_numeric(df[lon_col_cleaned], errors='coerce')
        df.rename(columns={lat_col_cleaned: 'Latitude', lon_col_cleaned: 'Longitude'}, inplace=True)
        
        original_rows = len(df)
        df.dropna(subset=['Latitude', 'Longitude'], inplace=True)
        rows_after_dropping_na = len(df)
        print(f"  Removed {original_rows - rows_after_dropping_na} rows with missing coordinates.")
        df_cleaned = df # Keep all original columns for these files

    # Remove duplicate rows based on all current columns in df_cleaned
    original_rows_cleaned = len(df_cleaned)
    df_cleaned.drop_duplicates(inplace=True)
    rows_after_dropping_duplicates = len(df_cleaned)
    print(f"  Removed {original_rows_cleaned - rows_after_dropping_duplicates} duplicate rows from cleaned data.")

    # Reorder columns to have Latitude and Longitude first for consistency where applicable
    # For road data, LINE_ID and POINT_ORDER are also important at the beginning
    if 'Latitude' in df_cleaned.columns and 'Longitude' in df_cleaned.columns:
        if is_road_responsibility_file:
            # Ensure LINE_ID and POINT_ORDER are at the front for road data
            cols = ['LINE_ID', 'POINT_ORDER', 'Latitude', 'Longitude'] + \
                   [col for col in df_cleaned.columns if col not in ['LINE_ID', 'POINT_ORDER', 'Latitude', 'Longitude']]
        else:
            # For other files, Latitude and Longitude at the front
            cols = ['Latitude', 'Longitude'] + \
                   [col for col in df_cleaned.columns if col not in ['Latitude', 'Longitude']]
        
        # Ensure all columns exist before reordering
        existing_cols = [col for col in cols if col in df_cleaned.columns]
        df_cleaned = df_cleaned[existing_cols]

    # Save the cleaned DataFrame to CSV using a SEMICOLON as delimiter
    df_cleaned.to_csv(output_filename, index=False, sep=';')
    print(f"  Cleaned data saved to '{output_filename}' with {len(df_cleaned)} rows.")


# Define the files and their specific cleaning configurations
# Ensure your original CSV files are in the same directory as this script.
files_to_clean = {
    "public-litter-bins.csv": {
        "output": "public-litter-bins-clean.csv",
        "lat_col": "Latitude Coordinate",
        "lon_col": "Longitude Coordinate",
        "is_waste_collection_area": False,
        "is_road_responsibility_file": False
    },
    "road-responsibility.csv": {
        "output": "road-responsibility-clean.csv",
        "lat_col": "Latitude Coordinate", # This specific column is not directly used for roads anymore
        "lon_col": "Longitude Coordinate", # as Feature_Extent is parsed, but kept for function signature consistency.
        "is_waste_collection_area": False,
        "is_road_responsibility_file": True # Activates LineString/MultiLineString parsing from Feature_Extent
    },
    "waste-collection-area.csv": {
        "output": "waste-collection-area-clean.csv",
        "lat_col": "Latitude Coordinate", # These will be newly generated from centroid calculation
        "lon_col": "Longitude Coordinate",
        "is_waste_collection_area": True, # Activates centroid calculation from Feature_Extent
        "is_road_responsibility_file": False
    },
    "waste-facility-locations.csv": {
        "output": "waste-facility-locations-clean.csv",
        "lat_col": "Latitude Coordinate",
        "lon_col": "Longitude Coordinate",
        "is_waste_collection_area": False,
        "is_road_responsibility_file": False
    },
}

# Run the cleaning process for each file
print("Starting the initial data cleaning process...")
for input_file, config in files_to_clean.items():
    clean_and_save_csv(
        input_file,
        config["output"],
        config["lat_col"],
        config["lon_col"],
        is_waste_collection_area=config["is_waste_collection_area"],
        is_road_responsibility_file=config["is_road_responsibility_file"]
    )

print("\nInitial data cleaning process completed. The cleaned files are ready.")