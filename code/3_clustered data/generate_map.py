import pandas as pd
import folium
import json
import os

# --- File Paths (Ensure these files are in the same directory as this script) ---
PUBLIC_LITTER_BINS_SPATIAL_FILE = 'public-litter-bins-clean-spatial.csv'
WASTE_COLLECTION_AREAS_SPATIAL_FILE = 'waste-collection-area-clean-spatial.csv'
WASTE_FACILITY_LOCATIONS_SPATIAL_FILE = 'waste-facility-locations-clean-spatial.csv'
ROAD_RESPONSIBILITY_SPATIAL_FILE = 'road-responsibility-clean-spatial.csv' # This file has START/END LAT/LON

LITTER_BINS_CLUSTERING_JSON = 'public_litter_bins_to_waste_collection_areas_clustering.json'
WASTE_AREAS_CLUSTERING_JSON = 'waste_collection_areas_to_waste_facilities_clustering.json'

OUTPUT_HTML_FILE = 'combined_map_with_clustering.html'

# --- Helper Functions for Data Loading ---

def load_csv_data(file_path):
    """Loads CSV data with semicolon delimiter."""
    try:
        return pd.read_csv(file_path, delimiter=';')
    except FileNotFoundError:
        print(f"Error: CSV file not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error loading CSV {file_path}: {e}")
        return None

def load_json_data(file_path):
    """Loads JSON data."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file not found: {file_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {file_path}: {e}")
        return None
    except Exception as e:
        print(f"Error loading JSON {file_path}: {e}")
        return None

# --- Main Script Execution ---
print("Starting map generation script...")

# 1. Load all necessary data
print("Loading spatial data...")
bins_df = load_csv_data(PUBLIC_LITTER_BINS_SPATIAL_FILE)
waste_areas_df = load_csv_data(WASTE_COLLECTION_AREAS_SPATIAL_FILE)
facilities_df = load_csv_data(WASTE_FACILITY_LOCATIONS_SPATIAL_FILE)
roads_df = load_csv_data(ROAD_RESPONSIBILITY_SPATIAL_FILE)

print("Loading clustering data...")
bins_to_areas_clustering = load_json_data(LITTER_BINS_CLUSTERING_JSON)
areas_to_facilities_clustering = load_json_data(WASTE_AREAS_CLUSTERING_JSON)

# Check if all essential data loaded successfully
if any(df is None for df in [bins_df, waste_areas_df, facilities_df, roads_df]) or \
   any(json_data is None for json_data in [bins_to_areas_clustering, areas_to_facilities_clustering]):
    print("Aborting map generation due to missing or unreadable input files.")
    exit()

# 2. Pre-process data for quick lookups and clustering info
print("Processing data for map rendering...")

# Create dictionaries for quick spatial coordinate lookup by ID
public_litter_bins_coords = {row['ID']: {'lat': row['Latitude'], 'lon': row['Longitude']} for _, row in bins_df.iterrows()}
waste_collection_areas_coords = {row['ID']: {'lat': row['Latitude'], 'lon': row['Longitude']} for _, row in waste_areas_df.iterrows()}
waste_facility_locations_coords = {row['ID']: {'lat': row['Latitude'], 'lon': row['Longitude']} for _, row in facilities_df.iterrows()}

# Create dictionaries to easily lookup clustered items for popups
waste_area_to_bins_map = {}
if bins_to_areas_clustering and 'assignments' in bins_to_areas_clustering:
    for assignment in bins_to_areas_clustering['assignments']:
        area_id = int(assignment['waste_collection_area_id'])
        bin_ids = [int(bid) for bid in assignment['public_litter_bin_ids']]
        waste_area_to_bins_map[area_id] = bin_ids

facility_to_areas_map = {}
if areas_to_facilities_clustering and 'assignments' in areas_to_facilities_clustering:
    for assignment in areas_to_facilities_clustering['assignments']:
        facility_id = int(assignment['waste_facility_id'])
        area_ids = [int(aid) for aid in assignment['waste_collection_area_ids']]
        facility_to_areas_map[facility_id] = area_ids

# 3. Initialize the map centered around Casey
m = folium.Map(location=[-38.0, 145.0], zoom_start=10)

# 4. Create FeatureGroups for toggling all layers
bins_group = folium.FeatureGroup(name='Public Litter Bins (Lightgreen)')
collection_areas_group = folium.FeatureGroup(name='Waste Collection Areas (Orange)')
facility_locations_group = folium.FeatureGroup(name='Waste Facility Locations (Red)')
roads_group = folium.FeatureGroup(name='Roads (Blue)')
cluster_bins_to_areas_group = folium.FeatureGroup(name='Bin to Area Clusters (Purple)')
cluster_areas_to_facilities_group = folium.FeatureGroup(name='Area to Facility Clusters (Dark Red)')

# 5. Add markers for Public Litter Bins
print("Adding Public Litter Bin markers...")
for _, row in bins_df.iterrows():
    bin_id = int(row['ID'])
    lat = row['Latitude']
    lon = row['Longitude']
    
    popup_content = f"<b>Public Litter Bin ID: {bin_id}</b>"
    
    # Check if this bin is assigned to a waste collection area
    assigned_area_id = None
    for area_id, bin_ids in waste_area_to_bins_map.items():
        if bin_id in bin_ids:
            assigned_area_id = area_id
            break
    if assigned_area_id is not None:
        popup_content += f"<br>Assigned to Waste Area ID: {assigned_area_id}"

    folium.Marker(
        location=[lat, lon],
        icon=folium.Icon(color='lightgreen', icon='trash', prefix='fa'), # Using Font Awesome icon
        popup=popup_content
    ).add_to(bins_group)

# 6. Add markers for Waste Collection Areas
print("Adding Waste Collection Area markers...")
for _, row in waste_areas_df.iterrows():
    area_id = int(row['ID'])
    lat = row['Latitude']
    lon = row['Longitude']

    popup_content = f"<b>Waste Collection Area ID: {area_id}</b>"
    # List bins assigned to this area
    if area_id in waste_area_to_bins_map and waste_area_to_bins_map[area_id]:
        popup_content += f"<br>Assigned Bins: {', '.join(map(str, waste_area_to_bins_map[area_id]))}"
    else:
        popup_content += "<br>No bins assigned to this area."
    
    # Check if this area is assigned to a waste facility
    assigned_facility_id = None
    for facility_id, area_ids in facility_to_areas_map.items():
        if area_id in area_ids:
            assigned_facility_id = facility_id
            break
    if assigned_facility_id is not None:
        popup_content += f"<br>Assigned to Facility ID: {assigned_facility_id}"

    folium.Marker(
        location=[lat, lon],
        icon=folium.Icon(color='orange', icon='cube', prefix='fa'), # Using Font Awesome icon
        popup=popup_content
    ).add_to(collection_areas_group)

# 7. Add markers for Waste Facility Locations
print("Adding Waste Facility Location markers...")
for _, row in facilities_df.iterrows():
    facility_id = int(row['ID'])
    lat = row['Latitude']
    lon = row['Longitude']

    popup_content = f"<b>Waste Facility ID: {facility_id}</b>"
    # List areas assigned to this facility
    if facility_id in facility_to_areas_map and facility_to_areas_map[facility_id]:
        popup_content += f"<br>Assigned Areas: {', '.join(map(str, facility_to_areas_map[facility_id]))}"
    else:
        popup_content += "<br>No areas assigned to this facility."

    folium.Marker(
        location=[lat, lon],
        icon=folium.Icon(color='red', icon='industry', prefix='fa'), # Using Font Awesome icon
        popup=popup_content
    ).add_to(facility_locations_group)

# 8. Add Road Polylines
print("Adding road polylines...")
for _, row in roads_df.iterrows():
    start_lat = row['START_LAT']
    start_lon = row['START_LON']
    end_lat = row['END_LAT']
    end_lon = row['END_LON']
    
    # Ensure coordinates are valid floats
    if pd.isna(start_lat) or pd.isna(start_lon) or pd.isna(end_lat) or pd.isna(end_lon):
        print(f"Warning: Skipping road segment due to invalid coordinates: {row.to_dict()}")
        continue

    folium.PolyLine(
        locations=[[start_lat, start_lon], [end_lat, end_lon]],
        color='blue',
        weight=3,
        opacity=0.7
    ).add_to(roads_group)

# 9. Add Clustering Polylines (Bin to Area)
print("Adding Bin to Area clustering lines...")
if bins_to_areas_clustering and 'assignments' in bins_to_areas_clustering:
    for assignment in bins_to_areas_clustering['assignments']:
        area_id = int(assignment['waste_collection_area_id'])
        bin_ids = [int(bid) for bid in assignment['public_litter_bin_ids']]
        
        area_coords = waste_collection_areas_coords.get(area_id)
        
        if area_coords:
            for bin_id in bin_ids:
                bin_coords = public_litter_bins_coords.get(bin_id)
                if bin_coords:
                    folium.PolyLine(
                        locations=[[bin_coords['lat'], bin_coords['lon']], [area_coords['lat'], area_coords['lon']]],
                        color='purple',
                        weight=1.5,
                        opacity=0.6,
                        dash_array='5, 5' # Dashed line
                    ).add_to(cluster_bins_to_areas_group)
                else:
                    print(f"Warning: Bin ID {bin_id} not found in spatial data for clustering line.")
        else:
            print(f"Warning: Waste Area ID {area_id} not found in spatial data for clustering line.")
else:
    print("No Bin to Area clustering data found or assignments missing.")

# 10. Add Clustering Polylines (Area to Facility)
print("Adding Area to Facility clustering lines...")
if areas_to_facilities_clustering and 'assignments' in areas_to_facilities_clustering:
    for assignment in areas_to_facilities_clustering['assignments']:
        facility_id = int(assignment['waste_facility_id'])
        area_ids = [int(aid) for aid in assignment['waste_collection_area_ids']]
        
        facility_coords = waste_facility_locations_coords.get(facility_id)
        
        if facility_coords:
            for area_id in area_ids:
                area_coords = waste_collection_areas_coords.get(area_id)
                if area_coords:
                    folium.PolyLine(
                        locations=[[area_coords['lat'], area_coords['lon']], [facility_coords['lat'], facility_coords['lon']]],
                        color='darkred',
                        weight=2,
                        opacity=0.7,
                        dash_array='3, 3' # Slightly different dashed line
                    ).add_to(cluster_areas_to_facilities_group)
                else:
                    print(f"Warning: Waste Area ID {area_id} not found in spatial data for clustering line.")
        else:
            print(f"Warning: Facility ID {facility_id} not found in spatial data for clustering line.")
else:
    print("No Area to Facility clustering data found or assignments missing.")

# 11. Add all feature groups to the map
bins_group.add_to(m)
collection_areas_group.add_to(m)
facility_locations_group.add_to(m)
roads_group.add_to(m)
cluster_bins_to_areas_group.add_to(m) # Add by default
cluster_areas_to_facilities_group.add_to(m) # Add by default

# 12. Add layer control to toggle groups on/off
folium.LayerControl().add_to(m)

# 13. Save combined map to one HTML file
m.save(OUTPUT_HTML_FILE)
print(f"Map saved to {OUTPUT_HTML_FILE}")
print("Script finished successfully!")
