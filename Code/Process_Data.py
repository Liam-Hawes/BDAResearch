import geopandas as gpd
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import os
import subprocess

def get_git_root():
    try:
        git_root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], universal_newlines=True).strip()
        return git_root
    except subprocess.CalledProcessError:
        raise RuntimeError("This script must be run inside a Git repository.")

# Set the working directory to the Git repository root
os.chdir(get_git_root())

# Now, all file paths will be relative to the Git repository root
print("Current working directory set to:", os.getcwd())
# Load River Mask Shapefiles
years = [2017, 2022, 2023]
river_masks = {
    2017: gpd.read_file('Raw_Data/2017RiverMask/2017RiverMask.shp'),
    2022: pd.concat([
        gpd.read_file('Raw_Data/2022South_Section_RiverMask/2022SouthRiverMask.shp'),
        gpd.read_file('Raw_Data/2022North_Section_RiverMask/2022HighResMask.shp')
    ], ignore_index=True),
    2023: pd.concat([
        gpd.read_file('Raw_Data/2023South_Section_RiverMask/2023_riverMask_south_Fixed_geometries.shp'),
        gpd.read_file('Raw_Data/2023North_Section_RiverMask/2023_riverMask.shp')
    ], ignore_index=True)
}

# Load Line Divider Shapefile
line_dividers = gpd.read_file('Raw_Data/2023Line_dividers/BDA_Line_dividers_5m_upstream_and_downstream.shp')
# Ensure CRS matches between all shapefiles
for year, mask in river_masks.items():
    if mask.crs != line_dividers.crs:
        river_masks[year] = mask.to_crs(line_dividers.crs)

# Prepare an empty list to collect data
data = []

# Iterate through each year and calculate intersections
for year, river_mask in river_masks.items():
    # Spatial join to find intersections between line dividers and river mask
    intersections = gpd.overlay(line_dividers, river_mask, how='intersection')
    
    for _, row in intersections.iterrows():
        # Extract relevant attributes
        bda_number = row.get('BDANum', None)  # Assuming this field exists in line_dividers
        reference_number = row.get('Reference', None)  # Assuming this field exists in line_dividers
        downstream = row.get('Downstream', None)  # Assuming this field exists in line_dividers
        comments = row.get('Comments', None)  # Assuming this field exists in line_dividers
        line_length = row.geometry.length  # Length of the overlapping segment
        geometry = row.geometry  # Include geometry information
        
        # Set BDAPresent based on conditions
        if year == 2017:
            bda_present = 0
        elif year in [2022, 2023]:
            bda_present = 0 if pd.notna(reference_number) and reference_number != '' else 1
        
        # Append data for the dataset
        data.append([year, bda_number, reference_number, downstream, comments, bda_present, line_length, geometry])

# Create GeoDataFrame from collected data
df = gpd.GeoDataFrame(data, columns=['Year', 'BDANumber', 'ReferenceNumber', 'Downstream', 'Comments', 'BDAPresent', 'LineLength', 'geometry'], geometry='geometry')

# Remove duplicate rows for BDANumber 30 in 2023, keeping only the first two occurrences
df_2023 = df[(df['Year'] == 2023) & (df['BDANumber'] == 30)]
if len(df_2023) > 2:
    indices_to_drop = df_2023.index[2:]
    df = df.drop(indices_to_drop)

# Save to CSV
df.to_csv('Analysis_Data/analysis_ready_dataset.csv', index=False)

print("Dataset creation complete. Geometry information included in the GeoDataFrame.")
