import geopandas as gpd
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

# Load River Mask Shapefiles
years = [2017, 2022, 2023]
river_masks = {
    2017: gpd.read_file('file:///Volumes/Samsung_T5/BDA Research/BDA QGIS Work/Try 3 Manual Shapefiles of the river/2017/RiverMask/2017RiverMask.shp'),
    2022: pd.concat([
        gpd.read_file('file:///Volumes/Samsung_T5/BDA Research/BDA QGIS Work/Try 3 Manual Shapefiles of the river/2022/South_Section/2022SouthRiverMask.shp'),
        gpd.read_file('file:///Volumes/Samsung_T5/BDA Research/BDA QGIS Work/Try 3 Manual Shapefiles of the river/Shapefiles/2022HighResMask.shp')
    ], ignore_index=True),
    2023: pd.concat([
        gpd.read_file('file:///Volumes/Samsung_T5/BDA Research/BDA QGIS Work/Try 3 Manual Shapefiles of the river/2023/2023_riverMask_south_Fixed_geometries.shp'),
        gpd.read_file('file:///Volumes/Samsung_T5/BDA Research/BDA QGIS Work/Try 3 Manual Shapefiles of the river/2023/2023_riverMask.shp')
    ], ignore_index=True)
}

# Load Line Divider Shapefile
line_dividers = gpd.read_file('file:///Volumes/Samsung_T5/BDA Research/BDA QGIS Work/Try 3 Manual Shapefiles of the river/2023/BDA Divider Lines 5m/BDA_Line_dividers_5m_upstream_and_downstream.shp')

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

# Save to CSV (Commented Out)
# df.to_csv('/Users/liamhawes/Documents/college/Grad/2 Envi Data Science/SemesterProject/analysis_ready_dataset.csv', index=False)

print("Dataset creation complete. Geometry information included in the GeoDataFrame.")


# Model Code

# Add an offset of 100 to ReferenceNumber to distinguish it from BDANumber
df['ReferenceNumber'] = df['ReferenceNumber'].apply(lambda x: x + 100 if pd.notna(x) else x)

# Create a SiteID column by combining BDANumber and ReferenceNumber (with offset applied)
df['SiteID'] = df['BDANumber'].combine_first(df['ReferenceNumber']).astype(int)

print("Offset added to ReferenceNumber and SiteID updated.")

# Data Preparation
df['SiteID'] = df['BDANumber'].combine_first(df['ReferenceNumber']).astype(int)

# Convert SiteID, BDAPresent, and Downstream columns to categorical data types
df['SiteID'] = df['SiteID'].astype('category')
df['BDAPresent'] = df['BDAPresent'].astype('category')
df['Downstream'] = df['Downstream'].astype('category')

print("Converted SiteID, BDAPresent, and Downstream to categorical types.")

# # Fit the Linear Mixed Effects Model Includes year downstream and upstream

# # W_{i,j} = LineLength
# # BDA_{i,j} = BDAPresent
# # Random intercepts: SiteID (location) and Year
# model = smf.mixedlm("LineLength ~ BDAPresent", df, groups=df["SiteID"], re_formula="~Year")
# result = model.fit()

# # Display Model Summary
# print(result.summary())
# # This model returned a value of zero for the year to year variation


# # No Year, Both Upstream and Downstream

# # Fit the Linear Mixed Effects Model without Year as a random effect
# # W_{i,j} = LineLength
# # BDA_{i,j} = BDAPresent
# # Random intercepts: SiteID (location)
# model_no_year = smf.mixedlm("LineLength ~ BDAPresent", df, groups=df["SiteID"])
# result_no_year = model_no_year.fit()

# # Display Model Summary
# print(result_no_year.summary())
# # --------------------------------------------------------
# #               Coef.  Std.Err.   z    P>|z| [0.025 0.975]
# # --------------------------------------------------------
# # Intercept      3.478    0.118 29.375 0.000  3.246  3.710
# # BDAPresent    -0.271    0.131 -2.063 0.039 -0.528 -0.014
# # Group Var      0.280    0.117                           
# # ========================================================

# # Downstream Analysis

# # Filter data to include only rows where Downstream is 1
# df_downstream = df[(df['Downstream'] == 1) | (df['ReferenceNumber'].notna())]

# # Fit the Linear Mixed Effects Model without Year as a random effect
# # Random intercepts: SiteID (location)
# model_downstream = smf.mixedlm("LineLength ~ BDAPresent", df_downstream, groups=df_downstream["SiteID"])
# result_downstream = model_downstream.fit()

# # Display Model Summary
# print(result_downstream.summary())
# #                   :                Coef.  Std.Err.   z    P>|z| [0.025 0.975]
# # This model returns: BDAPresent    -0.096    0.142 -0.673 0.501 -0.375  0.183
# # It appears that there is no corolation for the downstream size 5 meters down


# Filter data to include only rows where Downstream is 0
df_upstream = df[(df['Downstream'] == 0) | (df['ReferenceNumber'].notna())]


# Model 2: LineLength ~ BDAPresent + (1 + BDAPresent | groups)
# This model adds a random slope for BDAPresent within groups
model_2 = smf.mixedlm("LineLength ~ BDAPresent", df_upstream, groups=df_upstream["SiteID"], re_formula="~BDAPresent")
result_2 = model_2.fit()

# Display summary for Model 2
print("\nModel 2: LineLength ~ BDAPresent + (1 + BDAPresent | groups)")
print(result_2.summary())




# Fit the Linear Mixed Effects Model without Year as a random effect
# Random intercepts: SiteID (location)
model_upstream = smf.mixedlm("LineLength ~ BDAPresent", df_upstream, groups=df_upstream["SiteID"])
result_upstream = model_upstream.fit()

# Display Model Summary
print(result_upstream.summary())
# --------------------------------------------------------
#               Coef.  Std.Err.   z    P>|z| [0.025 0.975]
# --------------------------------------------------------
# Intercept      3.438    0.129 26.692 0.000  3.185  3.690
# BDAPresent    -0.374    0.158 -2.362 0.018 -0.684 -0.064
# Group Var      0.278    0.172                           
# ========================================================

# Extract and display coefficients from the model
coefficients = result_upstream.params
std_errors = result_upstream.bse  # Standard errors for the coefficients

# Print coefficients with standard errors
print("Model Coefficients for df_upstream:")
for param, coef, se in zip(coefficients.index, coefficients.values, std_errors.values):
    print(f"{param}: Coefficient = {coef:.4f}, Std. Error = {se:.4f}")

# Optionally, convert to a DataFrame for a more structured display
coef_df = pd.DataFrame({
    'Coefficient': coefficients.values,
    'Std. Error': std_errors.values
}, index=coefficients.index)

print("\nCoefficients DataFrame:")
print(coef_df)


# Add predicted values to the DataFrame
df_upstream['predicted'] = result_upstream.fittedvalues.values

# Calculate residuals
df_upstream['residuals'] = df_upstream['LineLength'] - df_upstream['predicted']

# Calculate standardized residuals
df_upstream['std_residuals'] = df_upstream['residuals'] / np.std(df_upstream['residuals'])


# File path to save the plot
file_path = '/Volumes/Samsung_T5/BDA Research/BDA QGIS Work/Try 3 Manual Shapefiles of the river/Latex/Images/2024_CEE_609'

# Save the plot as a PNG with retina resolution
plt.savefig(f"{file_path}/predicted_vs_actual_and_residuals.png", dpi=300, format='png', bbox_inches='tight')

# Create a figure with two subplots
plt.figure(figsize=(12, 10))

# First plot: Predicted (x) vs Actual (y)
plt.subplot(2, 1, 1)
plt.scatter(df_upstream['predicted'], df_upstream['LineLength'], alpha=0.7)
plt.plot(
    [df_upstream['LineLength'].min(), df_upstream['LineLength'].max()],
    [df_upstream['LineLength'].min(), df_upstream['LineLength'].max()],
    color='red', linestyle='--', linewidth=1, label='Perfect Prediction'
)
plt.title('Predicted vs. Actual Line Lengths (Upstream Analysis)')
plt.xlabel('Predicted Line Length')
plt.ylabel('Actual Line Length')
plt.legend()
plt.grid(True)

# Second plot: Predicted (x) vs Standardized Residuals (y)
plt.subplot(2, 1, 2)
plt.scatter(df_upstream['predicted'], df_upstream['std_residuals'], alpha=0.7)
plt.axhline(0, color='red', linestyle='--', linewidth=1)
plt.title('Predicted vs. Standardized Residuals (Upstream Analysis)')
plt.xlabel('Predicted Line Length')
plt.ylabel('Standardized Residuals')
plt.ylim(-6, 6)  # Set y-axis limits
plt.grid(True)

# Show the plots
plt.tight_layout()

# Save the plot as a PNG with retina resolution
plt.savefig(f"{file_path}/predicted_vs_actual_and_residuals.png", dpi=300, format='png', bbox_inches='tight')

# Show Plot
plt.show()




# # 1. Predicted vs Actual
# plt.figure(figsize=(8, 6))
# plt.scatter(df_upstream['predicted'], df_upstream['LineLength'], alpha=0.7)
# plt.plot([df_upstream['LineLength'].min(), df_upstream['LineLength'].max()],
#          [df_upstream['LineLength'].min(), df_upstream['LineLength'].max()],
#          color='red', linestyle='--', linewidth=1)
# plt.title('Predicted vs. Actual Line Lengths')
# plt.xlabel('Predicted Line Length')
# plt.ylabel('Actual Line Length')
# plt.grid(True)
# plt.show()

# # 2. Standardized Residuals vs Predicted
# plt.figure(figsize=(8, 6))
# plt.scatter(df_upstream['predicted'], df_upstream['std_residuals'], alpha=0.7)
# plt.axhline(0, color='red', linestyle='--', linewidth=1)
# plt.title('Standardized Residuals vs. Predicted Line Lengths')
# plt.xlabel('Predicted Line Length')
# plt.ylabel('Standardized Residuals')
# plt.ylim(-6, 6)
# plt.grid(True)
# plt.show()

# 3. Histogram of Residuals
plt.figure(figsize=(8, 6))
sns.histplot(df_upstream['residuals'], kde=True, bins=30)
plt.title('Histogram of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.grid(True)
# Save the plot as a PNG with retina resolution
plt.savefig(f"{file_path}/histogram_of_residuals.png", dpi=300, format='png', bbox_inches='tight')
plt.show()


# Calculate mean and standard deviation for the entire dataset
mean_line_length = df['LineLength'].mean()
std_line_length = df['LineLength'].std()

print(f"Overall Line Length: Mean = {mean_line_length:.4f}, Std = {std_line_length:.4f}")

# Calculate mean and standard deviation for BDAPresent = 1
mean_line_length_present = df[df['BDAPresent'] == 1]['LineLength'].mean()
std_line_length_present = df[df['BDAPresent'] == 1]['LineLength'].std()

print(f"BDAPresent = 1: Mean = {mean_line_length_present:.4f}, Std = {std_line_length_present:.4f}")

# Calculate mean and standard deviation for BDAPresent = 0
mean_line_length_absent = df[df['BDAPresent'] == 0]['LineLength'].mean()
std_line_length_absent = df[df['BDAPresent'] == 0]['LineLength'].std()

print(f"BDAPresent = 0: Mean = {mean_line_length_absent:.4f}, Std = {std_line_length_absent:.4f}")


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

