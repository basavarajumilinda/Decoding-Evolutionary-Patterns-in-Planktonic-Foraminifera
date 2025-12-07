import numpy as np
import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor
import re

def format_column_names(df):
    """Format column names by removing special characters, spaces, and converting to lowercase."""
    df.columns = df.columns.str.replace('.', '', regex=False)
    df.columns = [re.sub(r'[\(\)²\s]', '', col).lower() for col in df.columns]
    return df

def clean_file(df: pd.DataFrame) -> pd.DataFrame:
    """
    If the first column contains 'Count in filter ranges', retain only the preceding rows.
    """
    mask = df.iloc[:, 0].astype(str).str.contains('Count in filter ranges', na=False)
    if mask.any():
        df = df.iloc[:mask.idxmax()]
    return df

def read_file(file_path):
    """
    Read the file, remove the first two columns, and fill missing values.
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None

    file_type = os.path.splitext(file_path)[1].lower()
    df = pd.read_excel(file_path) if file_type in ['.xls', '.xlsx'] else pd.read_csv(file_path)
    df = clean_file(df)

    # Remove the first two columns
    df = df.iloc[:, 2:]

    # Handle NaN values
    df = df.fillna(df.mean())  # Fill with mean values
    df = df.fillna(0)  # Fill remaining NaNs with 0

    return df

def process_age_group(age_group, group, folder_path, output_folder, spilt):
    """Process a single age group and save the result."""
    dataframes = []

    for file_name in group['FileName']:
        file_path = os.path.join(folder_path, file_name)
        df = read_file(file_path)
        if df is not None:
            df = format_column_names(df)
            dataframes.append(df)

    if dataframes:
        result = pd.concat(dataframes, axis=0, ignore_index=True)
        output_file = os.path.join(output_folder, f"age_{age_group}_{age_group + spilt}.xlsx")
        result.to_excel(output_file, index=False)
        print(f"Saved: {output_file}, total rows: {len(result)}")
    else:
        print(f"No data for age range {age_group} - {age_group + spilt}")

def concate(file_name, folder_path, output_folder, spilt):
    """Main function: group and merge data by age range."""
    age_df = pd.read_excel(file_name)
    age_df['age_group'] = (age_df['Age(Ma)'] // spilt) * spilt
    grouped = age_df.groupby('age_group')

    os.makedirs(output_folder, exist_ok=True)

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_age_group, age_group, group, folder_path, output_folder, spilt)
                   for age_group, group in grouped]

if __name__ == '__main__':
    folder_path = r'C:/Users/ROG/Desktop/DSMP/Data Earth Sciences/Ceara Rise data'
    file_name = r'C:/Users/ROG/Desktop/DSMP/Data Earth Sciences/combined_metrics.xlsx'
    output_folder = r'C:/Users/ROG/Desktop/DSMP/Data Earth Sciences/Age_Grouped_Output'

    concate(file_name, folder_path, output_folder, spilt=1)
    print("Processing completed")