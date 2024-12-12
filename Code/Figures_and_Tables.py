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

# File path to save the plot
file_path = 'Figures_and_Tables'

# Paths to the CSV files
df_path = 'Code/Temp_Data/df.csv'
df_upstream_path = 'Code/Model_Data/df_upstream.csv'

# Load CSVs as DataFrames
df = pd.read_csv(df_path)
df_upstream = pd.read_csv(df_upstream_path)

# Convert to GeoDataFrames if 'geometry' column exists
if 'geometry' in df.columns:
    df['geometry'] = gpd.GeoSeries.from_wkt(df['geometry'])  # Convert WKT to Geometry
    df = gpd.GeoDataFrame(df, geometry='geometry')

if 'geometry' in df_upstream.columns:
    df_upstream['geometry'] = gpd.GeoSeries.from_wkt(df_upstream['geometry'])  # Convert WKT to Geometry
    df_upstream = gpd.GeoDataFrame(df_upstream, geometry='geometry')

# Confirm successful loading
print("Loaded GeoDataFrame 'df':")
print(df.head())
print("\nLoaded GeoDataFrame 'df_upstream':")
print(df_upstream.head())

##################################
# Plots not related to residuals #
##################################

# Boxplot of LineLength by BDAPresent
plt.figure(figsize=(8, 6))
sns.boxplot(data=df_upstream, x='BDAPresent', y='LineLength')
plt.title('Boxplot of Line Length by BDAPresent (Upstream)')
plt.xlabel('BDA Present')
plt.ylabel('Line Length (meters)')
plt.grid(True)
# Replace x-axis labels (0 -> No, 1 -> Yes)
plt.xticks(ticks=[0, 1], labels=['No', 'Yes'])
# Save the plot as a PNG with retina resolution
plt.savefig(f"{file_path}/boxplot_line_length_by_bdapresent.png", dpi=300, format='png', bbox_inches='tight')
plt.show()


# # Filter data to include only rows for the year 2023
# df_2023_upstream = df_upstream[df_upstream['Year'] == 2023]

# # Create the plot
# plt.figure(figsize=(12, 8))

# # Plot actual LineLength
# plt.scatter(
#     df_2023_upstream['BDANumber'],
#     df_2023_upstream['LineLength'],
#     color='blue',
#     label='Actual Line Length'
# )

# # Plot predicted LineLength
# plt.scatter(
#     df_2023_upstream['BDANumber'],
#     df_2023_upstream['predicted'],
#     color='orange',
#     label='Predicted Line Length'
# )

# # Customize the plot
# plt.title('Actual vs Predicted Line Lengths for Upstream Sites (2023) by BDA Number')
# plt.xlabel('BDA Number')
# plt.ylabel('Line Length')
# plt.xlim(0, 47)  # Set x-axis range
# plt.legend()
# plt.grid(True)

# # Show the plot
# plt.show()


# # Calculate mean and standard deviation for the entire dataset
# mean_line_length = df['LineLength'].mean()
# std_line_length = df['LineLength'].std()

# print(f"Overall Line Length: Mean = {mean_line_length:.4f}, Std = {std_line_length:.4f}")

# # Calculate mean and standard deviation for BDAPresent = 1
# mean_line_length_present = df[df['BDAPresent'] == 1]['LineLength'].mean()
# std_line_length_present = df[df['BDAPresent'] == 1]['LineLength'].std()

# print(f"BDAPresent = 1: Mean = {mean_line_length_present:.4f}, Std = {std_line_length_present:.4f}")

# # Calculate mean and standard deviation for BDAPresent = 0
# mean_line_length_absent = df[df['BDAPresent'] == 0]['LineLength'].mean()
# std_line_length_absent = df[df['BDAPresent'] == 0]['LineLength'].std()

# print(f"BDAPresent = 0: Mean = {mean_line_length_absent:.4f}, Std = {std_line_length_absent:.4f}")


# Filter data to include only rows for the year 2023
df_2023_upstream = df_upstream[df_upstream['Year'] == 2023]

# Sort the DataFrame by 'LineLength' in descending order
df_2023_upstream_sorted = df_2023_upstream.sort_values(by='LineLength', ascending=False)

# Filter the DataFrame to exclude BDANumber > 100
df_2023_upstream_filtered = df_2023_upstream_sorted[df_2023_upstream_sorted['BDANumber'] <= 100]

# Sort the data by 'LineLength' in descending order (or ensure it's already sorted as desired)
df_2023_upstream_sorted = df_2023_upstream_filtered.sort_values(by='LineLength', ascending=False)


# Create the plot
plt.figure(figsize=(12, 8))

# Plot actual LineLength
plt.scatter(
    range(len(df_2023_upstream_sorted)),
    df_2023_upstream_sorted['LineLength'].values,
    color='blue',
    label='2023 Actual Line Length'
)

# Plot predicted LineLength
plt.scatter(
    range(len(df_2023_upstream_sorted)),
    df_2023_upstream_sorted['predicted'].values,
    color='orange',
    label='Predicted Line Length'
)

# Customize the plot
plt.title('Actual vs Predicted Line Lengths for Upstream Sites (2023) by BDA Number')
plt.xlabel('BDA Number')
plt.ylabel('Line Length (meters)')
plt.xticks(
    ticks=range(len(df_2023_upstream_sorted)),
    labels=df_2023_upstream_sorted['BDANumber'].astype(int),
    rotation=45
)
plt.legend()
plt.grid(True)
plt.savefig(f"{file_path}/actual_vs_predicted_line_lengths_2023.png", dpi=300, format='png', bbox_inches='tight')
plt.show()





# Filter the data for all years with Downstream == 0 and BDANumber <= 100
df_filtered = df[(df['Downstream'] == 0) & (df['BDANumber'] <= 100)]

# Sort the data for 2023 by LineLength in descending order
df_2023_sorted = df_filtered[df_filtered['Year'] == 2023].sort_values(by='LineLength', ascending=False)

# Get the sorted BDA numbers based on 2023 data (unique values only)
sorted_bda_numbers = df_2023_sorted['BDANumber'].drop_duplicates()

# Reorder the entire dataset by the sorted BDA numbers
df_filtered['BDAOrder'] = pd.Categorical(df_filtered['BDANumber'], categories=sorted_bda_numbers, ordered=True)
df_sorted = df_filtered.sort_values('BDAOrder')

# Create a color map for the years
color_map = {
    2017: 'red',
    2022: 'green',
    2023: 'blue'
}

# Create the plot
plt.figure(figsize=(12, 8))

# Loop through each year and plot
for year, group in df_sorted.groupby('Year'):
    plt.scatter(
        group['BDAOrder'].cat.codes,  # x-axis based on sorted BDA numbers
        group['LineLength'].values,  # y-axis is Line Length
        color=color_map.get(year, 'gray'),  # Default color to gray if year is missing from color_map
        label=f"Year {year}",
        alpha=0.7
    )

# Customize the plot
plt.title('Upstream Line Length by BDA Number for Multiple Years')
plt.xlabel('BDA Number')
plt.ylabel('Line Length (meters)')
plt.xticks(
    ticks=range(len(sorted_bda_numbers)),
    labels=sorted_bda_numbers.astype(int),
    rotation=45
)
plt.legend(title="Year")
plt.grid(True)
plt.savefig(f"{file_path}/upstream_line_length_multiple_years.png", dpi=300, format='png', bbox_inches='tight')
# Show the plot
plt.show()




# Filter the data for all years with Downstream == 1 and BDANumber <= 100
df_filtered = df[(df['Downstream'] == 1) & (df['BDANumber'] <= 100)]

# Sort the data for 2023 by LineLength in descending order
df_2023_sorted = df_filtered[df_filtered['Year'] == 2023].sort_values(by='LineLength', ascending=False)

# Get the sorted BDA numbers based on 2023 data (unique values only)
sorted_bda_numbers = df_2023_sorted['BDANumber'].drop_duplicates()

# Reorder the entire dataset by the sorted BDA numbers
df_filtered['BDAOrder'] = pd.Categorical(df_filtered['BDANumber'], categories=sorted_bda_numbers, ordered=True)
df_sorted = df_filtered.sort_values('BDAOrder')

# Create a color map for the years
color_map = {
    2017: 'red',
    2022: 'green',
    2023: 'blue'
}

# Create the plot
plt.figure(figsize=(12, 8))

# Loop through each year and plot
for year, group in df_sorted.groupby('Year'):
    plt.scatter(
        group['BDAOrder'].cat.codes,  # x-axis based on sorted BDA numbers
        group['LineLength'].values,  # y-axis is Line Length
        color=color_map.get(year, 'gray'),  # Default color to gray if year is missing from color_map
        label=f"Year {year}",
        alpha=0.7
    )

# Customize the plot
plt.title('Downstream Line Length by BDA Number for Multiple Years')
plt.xlabel('BDA Number')
plt.ylabel('Line Length (meters)')
plt.xticks(
    ticks=range(len(sorted_bda_numbers)),
    labels=sorted_bda_numbers.astype(int),
    rotation=45
)
plt.legend(title="Year")
plt.grid(True)

plt.savefig(f"{file_path}/downstream_line_length_multiple_years.png", dpi=300, format='png', bbox_inches='tight')
# Show the plot
plt.show()

