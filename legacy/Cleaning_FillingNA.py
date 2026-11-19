import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from statsmodels.stats.outliers_influence import variance_inflation_factor   # added

# Load the 'CBS Squares' worksheet
print("Loading 'CBS Squares' data...")
file_path = 'data_Maastricht_2024.xlsx'
df = pd.read_excel(file_path, sheet_name='CBS Squares')

# Display initial information
print(f"Initial dataset shape: {df.shape}")
print("\nMissing values per column before cleaning:")
missing_values = df.isna().sum()
missing_percentages = (missing_values / len(df)) * 100
missing_info = pd.DataFrame({
    'Missing Values': missing_values,
    'Percentage': missing_percentages.round(2)
})
print(missing_info)

# Count missing values per row
missing_per_row = df.isna().sum(axis=1)
print("\nMissing values per row before cleaning:")
print(f"Min missing values in a row: {missing_per_row.min()}")
print(f"Max missing values in a row: {missing_per_row.max()}")
print(f"Mean missing values per row: {missing_per_row.mean().round(2)}")
print(f"Median missing values per row: {missing_per_row.median()}")

# Delete rows where population is N/A
print("\nRemoving rows where population is N/A...")
pop_columns = [col for col in df.columns if 'population' in col.lower()]

if pop_columns:
    print(f"Population column(s) found: {pop_columns}")
    df_cleaned = df.dropna(subset=pop_columns)
    print(f"Rows removed: {len(df) - len(df_cleaned)}")
else:
    # If no column has 'population' in its name, look for likely population columns
    # Common column names might include 'Pop', 'Total_Pop', 'Residents', etc.
    print("No column with 'population' in the name found.")
    print("Column names in the dataset:")
    print(df.columns.tolist())
    
    # For now, we'll assume the first column might be related to population
    # You can modify this based on the actual column name
    potential_pop_column = df.columns[1]  # Adjust index as needed
    print(f"Using '{potential_pop_column}' as the population column")
    df_cleaned = df.dropna(subset=[potential_pop_column])
    print(f"Rows removed: {len(df) - len(df_cleaned)}")

# Count missing values per row in the cleaned dataset
missing_per_row_after_pop = df_cleaned.isna().sum(axis=1)
print("\nMissing values per row after removing rows with missing population:")
print(f"Number of rows: {len(df_cleaned)}")
print(f"Min missing values in a row: {missing_per_row_after_pop.min()}")
print(f"Max missing values in a row: {missing_per_row_after_pop.max()}")
print(f"Mean missing values per row: {missing_per_row_after_pop.mean().round(2)}")

# Delete rows with more than 4 missing values
print("\nRemoving rows with more than 4 missing values...")
rows_before = len(df_cleaned)
df_cleaned = df_cleaned[missing_per_row_after_pop <= 6]
print(f"Additional rows removed: {rows_before - len(df_cleaned)}")

# Display information about the cleaned dataset
print(f"\nFinal cleaned dataset shape: {df_cleaned.shape}")
print("\nMissing values per column after cleaning:")
missing_after = df_cleaned.isna().sum()
missing_pct_after = (missing_after / len(df_cleaned)) * 100
missing_info_after = pd.DataFrame({
    'Missing Values': missing_after,
    'Percentage': missing_pct_after.round(2)
})
print(missing_info_after)

# Impute remaining missing values using median of nearby rows
print("\nImputing remaining missing values using median of nearby rows...")

# First, check if we have spatial coordinates in the data
spatial_cols = []
for col in df_cleaned.columns:
    if col.lower() in ['x', 'y', 'longitude', 'latitude', 'lon', 'lat', 'easting', 'northing', 'x_coord', 'y_coord']:
        spatial_cols.append(col)

if not spatial_cols and 'X' in df_cleaned.columns and 'Y' in df_cleaned.columns:
    spatial_cols = ['X', 'Y']  # Common naming in spatial datasets

if spatial_cols and len(spatial_cols) >= 2:
    print(f"Found spatial coordinates: {spatial_cols}")
    
    # Make a copy of the cleaned dataframe before imputation
    df_before_imputation = df_cleaned.copy()
    
    # Choose number of neighbors to consider
    n_neighbors = 5
    print(f"Using {n_neighbors} nearest neighbors for imputation")
    
    # Get the coordinates for finding nearest neighbors
    coords = df_cleaned[spatial_cols].values
    
    # Create a nearest neighbors model
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1)  # +1 because the point itself is included
    nn.fit(coords)
    
    # Find the nearest neighbors for each point
    distances, indices = nn.kneighbors(coords)
    
    # For each column with missing values
    columns_with_missing = df_cleaned.columns[df_cleaned.isna().any()].tolist()
    for col in columns_with_missing:
        # Skip spatial columns for imputation
        if col in spatial_cols:
            continue
        
        # Get indices of rows with missing values in this column
        missing_indices = df_cleaned[df_cleaned[col].isna()].index
        
        for idx in missing_indices:
            # Find the position of this index in the cleaned dataframe
            pos = df_cleaned.index.get_loc(idx)
            
            # Get the indices of the nearest neighbors (excluding the point itself)
            neighbor_indices = indices[pos][1:]
            
            # Get the values of the neighbors for this column
            neighbor_values = df_cleaned.iloc[neighbor_indices][col].dropna()
            
            # Calculate the median of the neighbor values if any exist
            if len(neighbor_values) > 0:
                df_cleaned.at[idx, col] = neighbor_values.median()
            else:
                # If all neighbors also have missing values, use column median
                df_cleaned.at[idx, col] = df_cleaned[col].median()
    
    # Display the number of values imputed
    imputed_count = {}
    for col in columns_with_missing:
        imputed_count[col] = df_before_imputation[col].isna().sum() - df_cleaned[col].isna().sum()
    
    imputed_df = pd.DataFrame({'Column': imputed_count.keys(), 'Values Imputed': imputed_count.values()})
    print("\nNumber of values imputed by column:")
    print(imputed_df)
    
    # Check if there are still missing values after imputation
    still_missing = df_cleaned.isna().sum()
    if still_missing.sum() > 0:
        print("\nRemaining missing values after spatial imputation:")
        print(still_missing[still_missing > 0])
        
        # For any remaining missing values, use column medians
        print("\nImputing remaining missing values with column medians...")
        df_cleaned = df_cleaned.fillna(df_cleaned.mean())
else:
    print("No spatial coordinates found for nearest-neighbor imputation.")
    print("Using column medians for imputation instead.")
    
    # Impute missing values with column medians
    df_cleaned = df_cleaned.fillna(df_cleaned.mean())

# Verify all missing values have been imputed
final_missing = df_cleaned.isna().sum().sum()
print(f"\nTotal missing values after all imputation: {final_missing}")

# --- Save the imputed dataset to Excel ---
imputed_output_path = 'cleaned_imputed_cbs_squares_after_na.xlsx'
df_cleaned.to_excel(imputed_output_path, index=False)
print(f"Imputed dataset saved to '{imputed_output_path}'")

# --- Additional Processing: Drop Gender Columns, Extra Columns, Age Proportions, and Household Proportions ---

# Drop gender columns if they exist
gender_cols = [col for col in df_cleaned.columns if col.lower() in ['female', 'male']]
if gender_cols:
    df_cleaned.drop(columns=gender_cols, inplace=True)
    print(f"Dropped gender column(s): {gender_cols}")
else:
    print("No gender columns ('female' or 'male') found to drop.")

# Drop extra columns 'Households' and 'Houses' if they exist
extra_drop_cols = ['Households', 'Houses']
existing_extra_drop_cols = [col for col in extra_drop_cols if col in df_cleaned.columns]
if existing_extra_drop_cols:
    df_cleaned.drop(columns=existing_extra_drop_cols, inplace=True)
    print(f"Dropped extra column(s): {existing_extra_drop_cols}")
else:
    print("No extra columns ('Households' or 'Houses') found to drop.")

# Define the expected age columns explicitly
expected_age_cols = ['Age0-14', 'Age15-24', 'Age25-44', 'Age45-64', 'Age65+']
# Use only the age columns that exist in the dataframe
age_cols = [col for col in expected_age_cols if col in df_cleaned.columns]
if age_cols:
    # Calculate the total age count per row
    df_cleaned['total_age'] = df_cleaned[age_cols].sum(axis=1)
    for col in age_cols:
        # Avoid division by zero
        df_cleaned[col + '_perc'] = np.where(df_cleaned['total_age'] > 0,
                                              df_cleaned[col] / df_cleaned['total_age'] * 100,
                                              0)
    print(f"Calculated age proportions for columns: {age_cols}")
else:
    print("No age-related columns found from the expected list:", expected_age_cols)

# Define the expected household type columns explicitly
expected_household_cols = [
    'Single-person households',
    'Multi-person households w/o kids',
    'Single parent households',
    'Two-parent households'
]
# Use only the household columns that exist in the dataframe
household_cols = [col for col in expected_household_cols if col in df_cleaned.columns]
if household_cols:
    # Calculate the total household count per row
    df_cleaned['total_household'] = df_cleaned[household_cols].sum(axis=1)
    for col in household_cols:
        df_cleaned[col + '_perc'] = np.where(df_cleaned['total_household'] > 0,
                                              df_cleaned[col] / df_cleaned['total_household'] * 100,
                                              0)
    print(f"Calculated household proportions for columns: {household_cols}")
else:
    print("No household-related columns found from the expected list:", expected_household_cols)

# (Optionally, drop the temporary sum columns if they are no longer needed)
for tmp_col in ['total_age', 'total_household']:
    if tmp_col in df_cleaned.columns:
        df_cleaned.drop(columns=tmp_col, inplace=True)


# --- Drop raw age & household columns before VIF analysis ---
drop_cols = [
    'Age0-14', 'Age15-24', 'Age25-44', 'Age45-64', 'Age65+',
    'Single-person households',
    'Multi-person households w/o kids',
    'Single parent households',
    'Two-parent households',
    'Rental %', 
    'Percentage high income households',
    'Age65+_perc',
    'Two-parent households_perc'
]
existing_drop = [col for col in drop_cols if col in df_cleaned.columns]
if existing_drop:
    df_cleaned.drop(columns=existing_drop, inplace=True)
    print(f"Dropped raw demographic columns before VIF: {existing_drop}")

# --- VIF Analysis: compute variance inflation factors for numeric features ---
# Place this block after all cleaning/imputation and before saving the dataset

# Select all numeric columns
numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()

# Exclude spatial coordinate columns if present
features = df_cleaned[numeric_cols].drop(columns=spatial_cols, errors='ignore')

# Build VIF dataframe
vif_df = pd.DataFrame({
    'feature': features.columns,
    'VIF': [variance_inflation_factor(features.values, i) for i in range(features.shape[1])]
})

print("\nVariance Inflation Factor (VIF) for numeric features:")
print(vif_df)

# Save the cleaned and imputed dataset (with additional processing)
output_path = 'cleaned_imputed_cbs_squares.xlsx'
df_cleaned.to_excel(output_path, index=False)
print(f"\nCleaned and imputed dataset saved to '{output_path}'")

# Save VIF results to an Excel file
vif_results_path = 'vif_results.xlsx'
vif_df.to_excel(vif_results_path, index=False)
print(f"VIF results saved to '{vif_results_path}'")

# Create comparison histograms for a few columns to show before/after imputation
# numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
# if len(numeric_cols) > 0:
#     sample_cols = numeric_cols[:min(20, len(numeric_cols))]
    
#     for col in sample_cols:
#         plt.figure(figsize=(12, 5))
        
#         plt.subplot(1, 2, 1)
#         plt.hist(df[col].dropna(), bins=20, color='skyblue', edgecolor='black')
#         plt.title(f'{col} - Before Imputation')
#         plt.xlabel('Value')
#         plt.ylabel('Frequency')
        
#         plt.subplot(1, 2, 2)
#         plt.hist(df_cleaned[col], bins=20, color='lightgreen', edgecolor='black')
#         plt.title(f'{col} - After Imputation')
#         plt.xlabel('Value')
#         plt.ylabel('Frequency')
        
#         plt.tight_layout()
#         plt.savefig(f'imputation_comparison_{col}.png')
#         print(f"Saved comparison histogram for {col}")

# # Create histogram of missing values per row in final dataset
# missing_per_row_final = df_cleaned.isna().sum(axis=1)
# plt.figure(figsize=(10, 6))
# plt.hist(missing_per_row_final, bins=range(0, 6), color='green', edgecolor='black', align='left')
# plt.xticks(range(0, 5))
# plt.xlabel('Number of Missing Values')
# plt.ylabel('Number of Rows')
# plt.title('Distribution of Missing Values Per Row (Final Dataset)')
# plt.grid(axis='y', alpha=0.3)
# plt.savefig('missing_values_per_row_final.png')
# print("Missing values per row histogram saved to 'missing_values_per_row_final.png'")