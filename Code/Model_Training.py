import geopandas as gpd
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import os
import subprocess
from scipy.spatial import distance_matrix
from esda.moran import Moran
from libpysal.weights import Kernel

def get_git_root():
    try:
        git_root = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], universal_newlines=True).strip()
        return git_root
    except subprocess.CalledProcessError:
        raise RuntimeError("This script must be run inside a Git repository.")

# Set the working directory to the Git repository root
os.chdir(get_git_root())


# Load the CSV file into a Pandas DataFrame
csv_path = "Analysis_Data/analysis_ready_dataset.csv"
df = pd.read_csv(csv_path)

# If the CSV has a 'geometry' column, convert it to a GeoDataFrame
if 'geometry' in df.columns:
    df['geometry'] = gpd.GeoSeries.from_wkt(df['geometry'])  # Ensure the geometry column is WKT
    df = gpd.GeoDataFrame(df, geometry='geometry')

print("GeoDataFrame loaded successfully:")
print(df.head())


################
## Model Code ##
################
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
file_path = 'Model_Outputs'

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


# Save df to 'Code/Temp_Data'
temp_data_path = 'Code/Temp_Data/df.csv'
df.to_csv(temp_data_path, index=False)
print(f"Saved df to {temp_data_path}")

# Save df_upstream to 'Code/Model_Data'
model_data_path = 'Code/Model_Data/df_upstream.csv'
df_upstream.to_csv(model_data_path, index=False)
print(f"Saved df_upstream to {model_data_path}")


# Geodataframe used for next analysis
gdf = df_upstream


# Check for missing data in the variable of interest and drop rows if necessary
variable_of_interest = 'LineLength'  # Replace with your column name
gdf = gdf.dropna(subset=[variable_of_interest])

# Create kernel-based spatial weights matrix
# Set bandwidth to determine the influence of nearby lines
bandwidth = 40  # meters
w_kernel = Kernel.from_dataframe(gdf, bandwidth=bandwidth)

# Ensure the weights matrix is row-standardized
w_kernel.transform = 'r'

# Extract the variable of interest as a NumPy array
y = gdf[variable_of_interest].values

# Calculate Moran's I using kernel weights
moran_kernel = Moran(y, w_kernel)

# Print Moran's I statistic and p-value
print(f"Moran's I (Kernel): {moran_kernel.I:.4f}")
print(f"P-value (Kernel): {moran_kernel.p_sim:.4f}")



# Check for missing data in the variable of interest and drop rows if necessary
variable_of_interest = 'LineLength'  # Replace with your column name
gdf = gdf.dropna(subset=[variable_of_interest])

# Extract coordinates and variable of interest
coords = np.array([(geom.centroid.x, geom.centroid.y) for geom in gdf.geometry])
values = gdf[variable_of_interest].values

# Calculate distance matrix
dist_matrix = distance_matrix(coords, coords)

# Calculate semivariance for distance bins
max_distance = np.percentile(dist_matrix, 95)  # Use 95th percentile as max distance
num_bins = 20  # Number of distance bins
bins = np.linspace(0, max_distance, num_bins + 1)

# Initialize arrays to store results
semivariances = []
bin_centers = []

for i in range(len(bins) - 1):
    # Get the indices of distances within the current bin
    bin_mask = (dist_matrix >= bins[i]) & (dist_matrix < bins[i + 1])
    
    # Calculate semivariance for the bin
    diff = values[:, None] - values[None, :]
    semivariance = 0.5 * np.mean(diff[bin_mask] ** 2)
    
    # Store the results
    semivariances.append(semivariance)
    bin_centers.append((bins[i] + bins[i + 1]) / 2)

# Plot the spatial variogram
plt.figure(figsize=(8, 6))
plt.plot(bin_centers, semivariances, marker='o', linestyle='-', color='blue')
plt.title('Spatial Variogram of LineLength')
plt.xlabel('Distance')
plt.ylabel('Semivariance')
plt.grid()
plt.savefig(f"{file_path}/Spatial_Variogram_of_LineLength.png", dpi=300, format='png', bbox_inches='tight')
plt.show()



# Determine the symmetric range around zero
max_abs_residual = max(abs(gdf['residuals'].min()), abs(gdf['residuals'].max()))

# Updated plot with symmetric color scale
plt.figure(figsize=(12, 8))
ax = gdf.plot(
    column='residuals',       # Column to visualize
    cmap='coolwarm',          # Diverging colormap
    legend=True,              # Add a color legend
    legend_kwds={'shrink': 0.6, 'label': 'Residuals'},  # Legend customization
    vmin=-max_abs_residual,   # Minimum value of the scale
    vmax=max_abs_residual,    # Maximum value of the scale
    linewidth=7,            # Increase line width for visibility
    alpha=1.0                 # Make lines fully opaque
)


# Add title and labels
plt.title('Spatial Plot of Model Residuals (Zero-Centered)', fontsize=16)
plt.axis('off')  # Turn off the axis for a cleaner map
plt.savefig(f"{file_path}/Spatial_plot_of_residuals.png", dpi=300, format='png', bbox_inches='tight')
# Show plot
plt.show()


