# ==============================================================================
# PART 1: Import Pandas
# ==============================================================================
# What is Pandas?
# Pandas is a Python library used to read, organize, analyze, and manipulate
# tabular data (data arranged in rows and columns, like a table or spreadsheet).
import pandas as pd

# Set display options so tables print clearly in the console without truncation
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

print("=" * 70)
print("SMART AGRICULTURE: PANDAS & DATAFRAME DEMONSTRATION")
print("=" * 70)

# ==============================================================================
# PART 2: Read the CSV File
# ==============================================================================
# pd.read_csv() reads a CSV (comma-separated values) file from disk into memory
# and converts it into a tabular DataFrame object.
df = pd.read_csv("data/class_analysis.csv")

print("\nData successfully loaded.")
print("\n--- Entire DataFrame ---")
print(df)

# ==============================================================================
# PART 3: Understand the DataFrame
# ==============================================================================
# What is a DataFrame?
# - DataFrame = table of rows and columns
# - Rows = records/classes (in our project, each row is a plant class)
# - Columns = attributes/features describing each record (crop, category, counts, etc.)
# - shape = number of rows and columns (rows, columns)
# - columns = column names (header of each column)
# - index = row labels (numerical positions 0 to 37 by default)
# - dtypes = data type of each column (e.g., int64 for integers, object for text)

print("\n" + "=" * 70)
print("PART 3: UNDERSTAND THE DATAFRAME")
print("=" * 70)

print("\n--- 1. Entire DataFrame (print(df)) ---")
print(df)

print("\n--- 2. First 5 Rows (print(df.head())) ---")
print(df.head())

print("\n--- 3. Last 5 Rows (print(df.tail())) ---")
print(df.tail())

print("\n--- 4. DataFrame Shape (print(df.shape)) ---")
print(f"Shape: {df.shape} -> ({df.shape[0]} rows, {df.shape[1]} columns)")

print("\n--- 5. Column Names (print(df.columns)) ---")
print(list(df.columns))

print("\n--- 6. Row Index Labels (print(df.index)) ---")
print(df.index)

print("\n--- 7. Column Data Types (print(df.dtypes)) ---")
print(df.dtypes)

print("\n--- 8. DataFrame Summary Information (df.info()) ---")
df.info()

# ==============================================================================
# PART 4: Access Columns
# ==============================================================================
# Difference between single and multiple column access:
# - df["crop"] (single bracket): selects a single column and returns a 1D Pandas Series.
# - df[["crop", "category", "total_image_count"]] (double brackets): selects multiple
#   columns and returns a 2D Pandas DataFrame.

print("\n" + "=" * 70)
print("PART 4: ACCESS COLUMNS")
print("=" * 70)

print("\n--- 1. Single Column Access (df['crop']) -> Pandas Series ---")
print(df["crop"].head())

print("\n--- 2. Multiple Column Access (df[['crop', 'category', 'total_image_count']]) -> DataFrame ---")
print(df[["crop", "category", "total_image_count"]].head())

print("\n--- 3. Accessing Specific Columns in Our Dataset ---")
print("\n[class_name column]:")
print(df["class_name"].head(5))

print("\n[crop column]:")
print(df["crop"].head(5))

print("\n[category column]:")
print(df["category"].head(5))

print("\n[total_image_count column]:")
print(df["total_image_count"].head(5))

# ==============================================================================
# PART 5: Access Rows
# ==============================================================================
# Difference between iloc and loc:
# - iloc (integer-location): accesses rows using their integer position (0, 1, 2, ...).
# - loc (label-location): accesses rows using their index label (in our default index, labels are also 0, 1, 2, ...).

print("\n" + "=" * 70)
print("PART 5: ACCESS ROWS")
print("=" * 70)

print("\n--- 1. First Row Using Integer Position (df.iloc[0]) ---")
print(df.iloc[0])

print("\n--- 2. Rows 0 to 4 Using Slicing (df.iloc[0:5]) ---")
print(df.iloc[0:5])

print("\n--- 3. First Row Using Index Label (df.loc[0]) ---")
print(df.loc[0])

# ==============================================================================
# PART 6: Filtering Data
# ==============================================================================
# What is filtering?
# Filtering selects rows that satisfy a specific condition (boolean condition).
# Rows where the condition evaluates to True are kept; False rows are excluded.

print("\n" + "=" * 70)
print("PART 6: FILTERING")
print("=" * 70)

print("\n--- 1. Healthy Classes (df[df['category'] == 'Healthy']) ---")
healthy = df[df["category"] == "Healthy"]
print(healthy)

print("\n--- 2. Diseased Classes (df[df['category'] == 'Diseased']) ---")
diseased = df[df["category"] == "Diseased"]
print(diseased)

print("\n--- 3. Tomato Classes (df[df['crop'] == 'Tomato']) ---")
tomato = df[df["crop"] == "Tomato"]
print(tomato)

print("\n--- 4. Classes with > 2400 Images (df[df['total_image_count'] > 2400]) ---")
large_classes = df[df["total_image_count"] > 2400]
print(large_classes)

# ==============================================================================
# PART 7: Basic DataFrame Statistics
# ==============================================================================
# Explanation of operations:
# - .sum() -> calculates total sum of numeric values in a column
# - .mean() -> calculates arithmetic average of numeric values in a column
# - .min() -> finds minimum value in a column
# - .max() -> finds maximum value in a column
# - .nunique() -> counts number of unique/distinct values in a column
# - .unique() -> returns array of all unique/distinct values in a column
# - .value_counts() -> counts how many times each distinct value occurs

print("\n" + "=" * 70)
print("PART 7: BASIC DATAFRAME STATISTICS")
print("=" * 70)

print(f"Total image count sum: {df['total_image_count'].sum():,}")
print(f"Average images per class: {df['total_image_count'].mean():.2f}")
print(f"Minimum images in a class: {df['total_image_count'].min():,}")
print(f"Maximum images in a class: {df['total_image_count'].max():,}")
print(f"Number of unique crops: {df['crop'].nunique()}")
print(f"List of unique crops: {list(df['crop'].unique())}")

print("\nCategory counts (df['category'].value_counts()):")
print(df["category"].value_counts())

print("\nCrop counts (df['crop'].value_counts()):")
print(df["crop"].value_counts())

# ==============================================================================
# PART 8: Basic GroupBy
# ==============================================================================
# What does groupby() do?
# groupby() groups rows having the same value in a specified column.
# In this project, grouping by "crop" and calling .size() tells us how many
# classes belong to each crop.

print("\n" + "=" * 70)
print("PART 8: BASIC GROUPBY")
print("=" * 70)

print("\nNumber of classes per crop (df.groupby('crop').size()):")
print(df.groupby("crop").size())

# ==============================================================================
# PART 9: Project Verification
# ==============================================================================
# Programmatic validation to confirm project numbers match ground truth.

rows_count = len(df)
cols_count = len(df.columns)
crops_count = df["crop"].nunique()
healthy_count = len(df[df["category"] == "Healthy"])
diseased_count = len(df[df["category"] == "Diseased"])
total_images = int(df["total_image_count"].sum())

# Assertions to ensure expected values
assert rows_count == 38, f"Expected 38 rows, got {rows_count}"
assert cols_count == 10, f"Expected 10 columns, got {cols_count}"
assert crops_count == 14, f"Expected 14 crops, got {crops_count}"
assert healthy_count == 12, f"Expected 12 healthy classes, got {healthy_count}"
assert diseased_count == 26, f"Expected 26 diseased classes, got {diseased_count}"
assert total_images == 87867, f"Expected 87867 total images, got {total_images}"

print("\n========================================")
print("PANDAS DATAFRAME VERIFICATION")
print("========================================")
print()
print(f"Rows: {rows_count}")
print(f"Columns: {cols_count}")
print(f"Crops: {crops_count}")
print(f"Healthy Classes: {healthy_count}")
print(f"Diseased Classes: {diseased_count}")
print(f"Total Images: {total_images}")
print()
print("========================================")
