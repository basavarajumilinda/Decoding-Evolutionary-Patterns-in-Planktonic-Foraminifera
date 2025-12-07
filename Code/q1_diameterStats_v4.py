"""
Author: Kaifeng Luo & Tengxiao Luo - M6
Date: 2025-02-05
Description:
- Further improved modularization.
- Added standard deviation computation to diameter statistics.
- Implemented column ordering and sorting before exporting results.
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
# Compute statistics for all "Diameter" columns
# -------------------------------
def calculate_diameter_statistics(file_path: str) -> dict:
    """
    Reads data (Excel or CSV), filters columns containing "Diameter",
    and computes:
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
    
    # Standardize column names
    df.columns = [format_file_or_column(str(col)) for col in df.columns]
    
    # Select columns containing "Diameter"
    diameter_columns = [col for col in df.columns if 'Diameter' in col]
    if not diameter_columns:
        raise ValueError(f"No column containing 'Diameter' found in {file_path}")
    
    stats = {}
    for col in diameter_columns:
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
# Extract specific statistics from the dictionary (supporting multiple candidates)
# -------------------------------
def get_diameter_value(stat_dict: dict, possible_names: list, stat_key: str) -> str:
    """
    Iterates through candidate column names and returns the corresponding 
    value from the statistics dictionary. If not found, return "N/A".
    """
    for name in possible_names:
        if name in stat_dict:
            return stat_dict[name][stat_key]
    return "N/A"

# -------------------------------
# Main Process: Process all files, compute statistics, and save to Excel
# -------------------------------
def process_files(folder_path: str, excel_files: list, keys: pd.Series, ages: pd.Series, output_file: str) -> None:
    """
    Computes diameter statistics for each matched file and writes results to an Excel file.
    
    Extracts values for the target columns (DiameterMean, DiameterMin, DiameterMax)
    using candidate names and outputs the following metrics:
      - Size.Mean.<target_column>
      - Size.sd.<target_column>      (standard deviation)
      - Size.95.<target_column>
      - Size.skewness.<target_column>
      - Size.kurtosis.<target_column>
    Also includes sample keys and age information.
    """
    file_map = build_mapping(excel_files, keys, ages)
    
    # Define candidate names for target columns
    target_columns = {
        "DiameterMean": ["MeanDiameterµm", "Mean(Diameter)(µm)", "Mean(Diameter)"],
        "DiameterMin":  ["MinDiameterµm", "Min(Diameter)(µm)", "Min(Diameter)"],
        "DiameterMax":  ["MaxDiameterµm", "Max(Diameter)(µm)", "Max(Diameter)"]
    }
    # Define output order: Mean → Min → Max
    target_order = ["DiameterMean", "DiameterMin", "DiameterMax"]
    
    results = []
    for formatted_key, (file, age) in file_map.items():
        file_path = os.path.join(folder_path, file)
        out_dict = {"Key": formatted_key, "Age(Ma)": age}
        try:
            stats = calculate_diameter_statistics(file_path)
            for target in target_order:
                candidates = target_columns[target]
                found = False
                for candidate in candidates:
                    if candidate in stats:
                        selected = stats[candidate]
                        found = True
                        break
                if found:
                    out_dict[f"Size.Mean.{target}"] = selected["mean"]
                    out_dict[f"Size.sd.{target}"]   = selected["std"]
                    out_dict[f"Size.95.{target}"] = selected["95"]
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
        results.append(out_dict)
    
    df_results = pd.DataFrame(results)
    # Define the order of columns in the final output
    desired_order = [
        "Key", "Age(Ma)",
        "Size.Mean.DiameterMean", "Size.Mean.DiameterMin", "Size.Mean.DiameterMax",
        "Size.sd.DiameterMean", "Size.sd.DiameterMin", "Size.sd.DiameterMax",
        "Size.95.DiameterMean", "Size.95.DiameterMin", "Size.95.DiameterMax",
        "Size.skewness.DiameterMean", "Size.skewness.DiameterMin", "Size.skewness.DiameterMax",
        "Size.kurtosis.DiameterMean", "Size.kurtosis.DiameterMin", "Size.kurtosis.DiameterMax"
    ]
    df_results = df_results.reindex(columns=[col for col in desired_order if col in df_results.columns])
    df_results = df_results.sort_values(by="Age(Ma)", ascending=True)

    df_results.to_excel(output_file, index=False)
    print(f"Results written to {output_file}")

# -------------------------------
# Main Execution
# -------------------------------
if __name__ == "__main__":
    folder_path = r'C:/Users/ROG/Desktop/DSMP/Data Earth Sciences/Ceara Rise data'
    excel_files = [f for f in os.listdir(folder_path) if f.endswith(('.xls', '.xlsx', '.csv'))]

    df_keys = pd.read_excel(r'C:/Users/ROG/Desktop/DSMP/Data Earth Sciences/925_Mastersheet.xlsx', skiprows=1)
    
    process_files(folder_path, excel_files, df_keys['Sample _IDS'], df_keys['Age (Ma)'], r'C:/Users/ROG/Desktop/DSMP/Data Earth Sciences/mapping_combined.xlsx')
