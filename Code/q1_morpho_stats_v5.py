"""
Author: Kaifeng Luo & Tengxiao Luo - M6
Date: 2025-02-26
Version: 5.0

Description:
    This script processes foraminiferal morphometric data, performing 
    statistical analysis on shape parameters to study species variation 
    over time.

Key Updates in v5:
    - Added "FileName" field in the output to track the original file source.
    - Improved output structure for easier reference.

Features:
    - Standardizes file/column names
    - Cleans and maps sample data to geological ages
    - Computes key statistics: Mean, SD, 95th Percentile, Skewness, Kurtosis
    - Outputs results in an Excel file with file tracking
"""

import os
import pandas as pd
from scipy.stats import skew, kurtosis

# -------------------------------
# Formatting Functions: Standardizing file and column names
# -------------------------------
def format_file_or_column(name: str) -> str:
    """
    Removes all non-alphanumeric characters and unnecessary substrings 
    to standardize file and column names.
    """
    formatted = ''.join(c for c in str(name) if c.isalnum())
    # Remove specific substrings (adjust as needed)
    formatted = formatted.replace('CountandMeasureof', '').replace('csv', '').replace('xlsx', '')
    return formatted

# -------------------------------
# File Mapping: Establish a mapping between sample keys and filenames (with ages)
# -------------------------------
def build_mapping(excel_files: list, keys: pd.Series, ages: pd.Series) -> dict:
    """
    Matches formatted sample keys with filenames and returns a dictionary:
    {formatted_sample_key: (filename, age)}
    """
    file_map = {}
    missing = 0
    for file in excel_files:
        formatted_file = format_file_or_column(file)
        matched_key = False
        for key, age in zip(keys, ages):
            formatted_key = format_file_or_column(key)
            if formatted_key == formatted_file:
                file_map[formatted_key] = (file, age)
                matched_key = True
                break  # Ensure one file matches only one key
        if not matched_key:
            missing += 1
    print(f"Missing files for {missing} keys")
    return file_map

# -------------------------------
# Data Cleaning: Remove invalid rows in Excel files
# -------------------------------
def clean_file(df: pd.DataFrame) -> pd.DataFrame:
    """
    If the first column contains 'Count in filter ranges',
    retain only the rows before this line.
    """
    mask = df.iloc[:, 0].astype(str).str.contains('Count in filter ranges', na=False)
    if mask.any():
        df = df.iloc[:mask.idxmax()]
    return df

# -------------------------------
# Independent Statistical Functions
# -------------------------------
def calculate_mean(series: pd.Series) -> float:
    """Calculate mean"""
    return series.mean()

def calculate_std(series: pd.Series) -> float:
    """Calculate standard deviation"""
    return series.std()

def calculate_95(series: pd.Series) -> float:
    """Calculate 95th percentile"""
    return series.quantile(0.95)

def calculate_skewness(series: pd.Series) -> float:
    """Calculate skewness"""
    return skew(series)

def calculate_kurtosis(series: pd.Series) -> float:
    """
    Calculate kurtosis (Pearson type, without subtracting 3)
    """
    return kurtosis(series, fisher=False)

# -------------------------------
# Compute statistics for all numeric columns in the file
# -------------------------------
def calculate_all_statistics(file_path: str) -> dict:
    """
    Reads data (Excel or CSV), standardizes column names,
    and computes statistics for all columns that can be converted to numeric:
      - Mean
      - Standard deviation
      - 95th percentile
      - Skewness
      - Kurtosis
    Returns a dictionary:
      { column_name: {'mean': ..., 'std': ..., '95': ..., 'skew': ..., 'kurt': ...}, ... }
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ['.xls', '.xlsx']:
        df = pd.read_excel(file_path)
        df = clean_file(df)
    elif ext == '.csv':
        df = pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    
    # Standardize all column names
    df.columns = [format_file_or_column(str(col)) for col in df.columns]
    
    stats = {}
    for col in df.columns:
        series = pd.to_numeric(df[col], errors='coerce').dropna()
        if series.empty:
            continue
        stats[col] = {
            'mean': calculate_mean(series),
            'std': calculate_std(series),
            '95': calculate_95(series),
            'skew': calculate_skewness(series),
            'kurt': calculate_kurtosis(series)
        }
    return stats

# -------------------------------
# Extract specific statistics from the dictionary using candidate substrings
# -------------------------------
def get_metric_value(stat_dict: dict, possible_names: list, stat_key: str) -> str:
    """
    Iterates through candidate substrings and returns the corresponding 
    value from the statistics dictionary if any key contains the candidate substring.
    If not found, returns "N/A".
    """
    for candidate in possible_names:
        for key in stat_dict.keys():
            if candidate.lower() in key.lower():
                return stat_dict[key][stat_key]
    return "N/A"

# -------------------------------
# Main Process: Process all files, compute statistics, and save results to Excel
# -------------------------------
def process_files(folder_path: str, excel_files: list, keys: pd.Series, ages: pd.Series, output_file: str) -> None:
    """
    Computes statistics for each matched file and writes results to an Excel file.
    
    For each target metric, candidate substrings are defined to match the (possibly non-uniform) column names.
    For each target metric, the following statistics are extracted:
      - Size.Mean.<target>
      - Size.sd.<target>      (standard deviation)
      - Size.95.<target>
      - Size.skewness.<target>
      - Size.kurtosis.<target>
    Also includes sample keys and age information.
    """
    file_map = build_mapping(excel_files, keys, ages)

    # Define candidate substrings for each target metric.
    # Adjust these candidate lists based on your actual data.
    target_columns = {
        "Area":              ["Area"],
        "GrayIntensity":     ["GrayIntensity", "GrayIntensityValue"],
        "ShapeFactor":       ["ShapeFactor", "Shape"],
        "DiameterMin":       ["MinDiameter"],
        "DiameterMax":       ["MaxDiameter"],
        "DiameterMean":      ["MeanDiameter"],
        "Elongation":        ["Elongation"],
        "Sphericity":        ["Sphericity"],
        "Perimeter":         ["Perimeter"]
    }
    # Define the desired order of target metrics
    target_order = ["Area", "GrayIntensity", "ShapeFactor", 
                    "DiameterMin", "DiameterMax", "DiameterMean", 
                    "Elongation", "Sphericity", "Perimeter"]
    
    results = []
    for formatted_key, (file, age) in file_map.items():
        file_path = os.path.join(folder_path, file)
        out_dict = {"Key": formatted_key, "Age(Ma)": age, "FileName": file}

        try:
            stats = calculate_all_statistics(file_path)
            # For each target metric, iterate over candidate substrings to find a matching column
            for target in target_order:
                candidates = target_columns[target]
                found = False
                selected = None
                for candidate in candidates:
                    for stat_key in stats.keys():
                        if candidate.lower() in stat_key.lower():
                            selected = stats[stat_key]
                            found = True
                            break
                    if found:
                        break
                if found:
                    out_dict[f"Size.Mean.{target}"] = selected["mean"]
                    out_dict[f"Size.sd.{target}"]   = selected["std"]
                    out_dict[f"Size.95.{target}"]    = selected["95"]
                    out_dict[f"Size.skewness.{target}"] = selected["skew"]
                    out_dict[f"Size.kurtosis.{target}"] = selected["kurt"]
                else:
                    for stat in ["Mean", "sd", "95", "skewness", "kurtosis"]:
                        out_dict[f"Size.{stat}.{target}"] = "N/A"
        except Exception as e:
            print(f"Error processing {file}: {e}")
            for target in target_order:
                for stat in ["Mean", "sd", "95", "skewness", "kurtosis"]:
                    out_dict[f"Size.{stat}.{target}"] = "N/A"
        print(out_dict)
        results.append(out_dict)
    
    df_results = pd.DataFrame(results)
    # Define the final column order: Key, Age, then for each target metric the five statistics

    # Desired_order section before modification:
    # desired_order = ["Key", "Age(Ma)"]
    # for target in target_order:
    #     desired_order.extend([
    #         f"Size.Mean.{target}",
    #         f"Size.sd.{target}",
    #         f"Size.95.{target}",
    #         f"Size.skewness.{target}",
    #         f"Size.kurtosis.{target}"
    #     ])
    
    # Modified to show all Mean first, then sd, 95, skewness, kurtosis
    stat_order = ["Mean", "sd", "95", "skewness", "kurtosis"]
    desired_order = ["Key", "Age(Ma)", "FileName"]
    for stat in stat_order:
        for target in target_order:
            desired_order.append(f"Size.{stat}.{target}")

    df_results = df_results.reindex(columns=[col for col in desired_order if col in df_results.columns])
    df_results = df_results.sort_values(by="Age(Ma)", ascending=True)

    df_results.to_excel(output_file, index=False)
    print(df_results)
    print(f"Results written to {output_file}")

# -------------------------------
# Main Execution
# -------------------------------
if __name__ == "__main__":
    folder_path = r'C:/Users/ROG/Desktop/DSMP/Data Earth Sciences/Ceara Rise data'
    # Get all Excel/CSV files in the folder
    excel_files = [f for f in os.listdir(folder_path) if f.endswith(('.xls', '.xlsx', '.csv'))]
    
    keys_path = r'C:/Users/ROG/Desktop/DSMP/Data Earth Sciences/925_Mastersheet.xlsx'
    df_keys = pd.read_excel(keys_path, skiprows=1)
    
    output_file = r'C:/Users/ROG/Desktop/DSMP/Data Earth Sciences/combined_metrics.xlsx'
    process_files(folder_path, excel_files, df_keys['Sample _IDS'], df_keys['Age (Ma)'], output_file)
