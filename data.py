import pandas as pd
import os

def load_and_clean_data(filepath):
    """
    Loads DASS-21 data from CSV or Excel, renames columns,
    removes duplicates and 'straight-liners'.
    """
    # 1. Load Data
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    else:
        df = pd.read_excel(filepath)
        
    print(f"Initial shape: {df.shape}")

    # 2. Rename Columns
    # Assuming the first two are Gender/Age and rest are Questions
    # Logic copied from original notebook:
    new_column_names = ['Gender', 'Age'] + [f'Q{i}' for i in range(1, 22)]
    
    # Verify column count matches 
    if len(df.columns) == len(new_column_names):
        df.columns = new_column_names
    else:
        # Fallback if strict count doesn't match, try to rename by index up to 23
        print("Warning: Column count mismatch, attempting slice rename...")
        df.columns = new_column_names + list(df.columns[len(new_column_names):])

    # 3. Standardize Categorical Data
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].astype(str).str.strip().str.title()

    initial_count = len(df)

    # 4. Remove Duplicates
    df = df.drop_duplicates()
    duplicates_removed = initial_count - len(df)

    # 5. Remove "Straight-Liners" (Response Bias)
    # Logic: Drop if Standard Deviation is 0 AND Mean is not 0
    question_cols = [f'Q{i}' for i in range(1, 22)]
    
    # Ensure Q columns are numeric
    for col in question_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows where Questions are NaN
    df = df.dropna(subset=question_cols)

    row_std = df[question_cols].std(axis=1)
    row_mean = df[question_cols].mean(axis=1)

    mask_invalid = (row_std == 0) & (row_mean > 0)
    straight_liners_removed = mask_invalid.sum()
    
    df_clean = df[~mask_invalid].copy()

    print(f"--- Data Cleaning Report ---")
    print(f"Duplicates Dropped: {duplicates_removed}")
    print(f"Straight-Liners Dropped: {straight_liners_removed}")
    print(f"Final Shape: {df_clean.shape}")

    return df_clean
