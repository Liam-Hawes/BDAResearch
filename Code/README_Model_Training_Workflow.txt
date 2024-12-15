
# README: Workflow for Model Training Code

This document provides an overview of the workflow for the model training code, including the processing steps, inputs, and outputs. 

---

## **Overview**

The provided code is designed to analyze river morphology data and train a **Linear Mixed Effects Model (LMM)** to understand the impact of **Beaver Dam Analog (BDA)** structures on river line lengths. The workflow involves reading geospatial data, processing intersections, and training predictive models.

---

## **Workflow Steps**

### **1. Input Data**
#### **Files and Locations**
- **River Mask Shapefiles**:
  - 2017: `file:///Volumes/Samsung_T5/BDA Research/BDA QGIS Work/Try 3 Manual Shapefiles of the river/2017/RiverMask/2017RiverMask.shp`
  - 2022: 
    - `file:///Volumes/Samsung_T5/BDA Research/BDA QGIS Work/Try 3 Manual Shapefiles of the river/2022/South_Section/2022SouthRiverMask.shp`
    - `file:///Volumes/Samsung_T5/BDA Research/BDA QGIS Work/Try 3 Manual Shapefiles of the river/Shapefiles/2022HighResMask.shp`
  - 2023: 
    - `file:///Volumes/Samsung_T5/BDA Research/BDA QGIS Work/Try 3 Manual Shapefiles of the river/2023/2023_riverMask_south_Fixed_geometries.shp`
    - `file:///Volumes/Samsung_T5/BDA Research/BDA QGIS Work/Try 3 Manual Shapefiles of the river/2023/2023_riverMask.shp`

- **Line Divider Shapefile**:
  - `file:///Volumes/Samsung_T5/BDA Research/BDA QGIS Work/Try 3 Manual Shapefiles of the river/2023/BDA Divider Lines 5m/BDA_Line_dividers_5m_upstream_and_downstream.shp`

#### **Description**
- The **River Mask Shapefiles** represent spatial polygons for river segments in different years.
- The **Line Divider Shapefile** contains lines at 5-meter intervals to measure river features.

---

### **2. Data Processing**
1. **Reading and CRS Alignment**:
   - Geospatial data from the river masks and line divider shapefiles are read using `geopandas`.
   - CRS (Coordinate Reference Systems) are aligned between datasets.

2. **Spatial Intersection**:
   - Line dividers are intersected with river masks to calculate overlap lengths (`LineLength`) for each year.
   - Attributes such as `BDANumber`, `ReferenceNumber`, and `Downstream` are extracted.

3. **BDA Presence Calculation**:
   - A binary variable (`BDAPresent`) is created:
     - `0`: No BDA within 5 meters.
     - `1`: BDA present within 5 meters.

4. **Site ID Creation**:
   - A unique `SiteID` column combines `BDANumber` and `ReferenceNumber` (with an offset of 100 for references).

5. **Filtering for Specific Conditions**:
   - Separate datasets are created for:
     - **Upstream Analysis**: Rows where `Downstream == 0`.
     - **Downstream Analysis**: Rows where `Downstream == 1`.

---

### **3. Model Training**
#### **a. Linear Mixed Effects Model**
- A **Linear Mixed Effects Model** is trained using `statsmodels` for `LineLength` as the dependent variable.
- Formula: 
  \[
  W_{i,j} = 	ext{LineLength}_{i,j} = eta_0 + eta_1 \cdot 	ext{BDAPresent}_{i,j} + u_i + \epsilon_{i,j}
  \]
  - Fixed Effects:
    - `BDAPresent`: Indicates the presence of a BDA.
  - Random Effects:
    - `u_i`: Variance attributed to `SiteID`.
  - Error Term:
    - \(\epsilon_{i,j}\): Residual variance.

#### **b. Model Variants**
1. **Full Dataset (No Year Random Effects)**:
   - Includes all data, but only random intercepts for `SiteID`.
2. **Upstream Data**:
   - Filters `Downstream == 0`.
3. **Downstream Data**:
   - Filters `Downstream == 1`.

---

### **4. Outputs**
#### **Data Outputs**
- A **GeoDataFrame** (`df`) containing:
  - Columns: `Year`, `BDANumber`, `ReferenceNumber`, `Downstream`, `Comments`, `BDAPresent`, `LineLength`, `geometry`, `SiteID`.
- **Filtered Datasets**:
  - `df_upstream`: Rows with `Downstream == 0`.
  - `df_downstream`: Rows with `Downstream == 1`.

#### **Model Outputs**
1. **Coefficients Summary**:
   - Intercept (\(eta_0\)), BDAPresent effect (\(eta_1\)).
   - Random effect variance (`Group Var`).
2. **Residual Analysis**:
   - Residuals and standardized residuals are calculated and visualized.

#### **Visualization Outputs**
1. **Predicted vs Actual Line Lengths**:
   - Scatter plot comparing predicted and actual values.
2. **Residual Plots**:
   - Scatter plot of standardized residuals vs. predicted values.
   - Histogram of residuals.
3. **Boxplot of Line Lengths by BDAPresent**:
   - Visualizes differences in `LineLength` for `BDAPresent = 0` and `BDAPresent = 1`.
4. **2023 Upstream Actual vs Predicted**:
   - Scatter plot showing actual and predicted line lengths for upstream data in 2023.

---

### **5. Key Observations**
- The impact of BDAs on line length is significant for upstream sites (`p < 0.05`).
- No significant impact for downstream sites.

---

### **6. Running the Code**
1. **Prerequisites**:
   - Python packages: `geopandas`, `numpy`, `pandas`, `statsmodels`, `matplotlib`, `seaborn`.
   - Ensure input shapefiles are accessible at the specified paths.
2. **Execution**:
   - Run the script from start to finish in an environment like Jupyter Notebook or a Python IDE.
3. **Customizations**:
   - Modify filtering conditions or plotting ranges as needed.

---

### **7. Notes**
- Missing CRS alignment or shapefile availability may lead to errors.
- Outputs can be saved to CSV by uncommenting the relevant line (`df.to_csv`).

For further assistance, please contact Liam Hawes, lphawes@g.syr.edu.
