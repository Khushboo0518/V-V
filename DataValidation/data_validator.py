import pandas as pd
import numpy as np

def check_missing_values(df):
    print("\n--- Missing Values Check ---")
    missing = df.isnull().sum()
    percent = (missing / len(df)) * 100
    result = pd.DataFrame({'Missing Count': missing, 'Percent %': percent})
    print(result)
    if missing.sum() == 0:
        print("PASS - No missing values found!")
    else:
        print("FAIL - Missing values detected!")

def check_data_balance(df, target_column):
    print("\n--- Data Balance Check ---")
    balance = df[target_column].value_counts()
    print(balance)
    min_class = balance.min()
    max_class = balance.max()
    ratio = min_class / max_class
    if ratio > 0.4:
        print("PASS - Data is reasonably balanced!")
    else:
        print("FAIL - Data is imbalanced! Bias risk detected!")

def check_duplicates(df):
    print("\n--- Duplicate Rows Check ---")
    duplicates = df.duplicated().sum()
    print(f"Duplicate rows found: {duplicates}")
    if duplicates == 0:
        print("PASS - No duplicates!")
    else:
        print("FAIL - Duplicates detected!")

def check_data_types(df):
    print("\n--- Data Types Check ---")
    print(df.dtypes)
    print("PASS - Data types listed above, check manually!")

def run_data_validation(filepath, target_column):
    print("=============================")
    print("   AI V&V DATA VALIDATOR     ")
    print("=============================")
    df = pd.read_csv(filepath)
    print(f"\nDataset loaded! Shape: {df.shape}")
    check_missing_values(df)
    check_duplicates(df)
    check_data_types(df)
    check_data_balance(df, target_column)
    print("\n=============================")
    print("   VALIDATION COMPLETE!")
    print("=============================")

# --- RUN IT ---
# ============================================================
# HOW TO USE THIS FUNCTION:
#
# run_data_validation("your_file.csv", "target_column")
#
#
# NOTE:
#   → Your CSV file and this script must be in the SAME folder
#   → Target column must exist in your CSV file
# ============================================================

run_data_validation("your_data.csv", "target")
run_data_validation("your_data.csv", "target")
