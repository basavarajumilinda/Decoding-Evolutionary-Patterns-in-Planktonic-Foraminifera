import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns
import matplotlib.pyplot as plt
import re

import re

'''
格式化列名
'''
def format_column_names(df):
    df.columns = df.columns.str.replace('.', '', regex=False)
    df.columns = [re.sub(r'[\(\)²\s]', '', col).lower() for col in df.columns]
    return df


def concate(file_name, folder_path, output_folder, spilt):
    """
    按照年龄段分组并合
    """
    # 读取包含文件名和年龄信息的 Excel
    age_df = pd.read_excel(file_name)

    # 分组 split = 1; 0-1 1-2...
    age_df['age_group'] = (age_df['Age(Ma)'] // spilt) * spilt
    grouped = age_df.groupby('age_group')
    # 输出目录
    os.makedirs(output_folder, exist_ok=True)

    for age_group, group in grouped:
        print(f"处理年份: {age_group} - {age_group + spilt}")
        dataframes = []

        # 遍历每个文件名，读取并拼接
        for file_name in group['FileName']:
            file_path = os.path.join(folder_path, file_name)
            print(file_name)
            # 读取数据
            df = read_file(file_path)
            if df is not None:
                # 标准化列名
                df = format_column_names(df)
                dataframes.append(df)

        # 按列名拼接
        if dataframes:
            # 按列拼接 填充na
            result = pd.concat(dataframes, axis=0, ignore_index=True, join="outer")

            # 保存拼接后的数据
            output_file = os.path.join(output_folder, f"age_{age_group}_{age_group + spilt}.xlsx")
            result.to_excel(output_file, index=False)
            print(f"保存: {output_file}, 共 {len(result)} 行")
        else:
            print(f"区间 {age_group} - {age_group + spilt} 没有数据")



def clean_file(df: pd.DataFrame) -> pd.DataFrame:
    """
    If the first column contains 'Count in filter ranges',
    retain only the rows before this line.
    """
    mask = df.iloc[:, 0].astype(str).str.contains('Count in filter ranges', na=False)
    if mask.any():
        df = df.iloc[:mask.idxmax()]
    return df

def read_file(file_path):
    """
    读取文件并去掉前两列
    """
    if not os.path.exists(file_path):
        print(f"文件未找到: {file_path}")
        return None

    # 读取文件
    file_type = os.path.splitext(file_path)[1].lower()
    df = pd.read_excel(file_path) if file_type in ['.xls', '.xlsx'] else pd.read_csv(file_path)
    df = clean_file(df)

    # 删除前两列
    df = df.iloc[:, 2:]

    # 处理 NaN
    df = df.fillna(df.mean())  # 用均值填充
    df = df.fillna(0)  # 如果依然有 NaN 则填充为 0

    return df


folder_path = 'C:/Users/罗腾霄/Desktop/研究生课程/Data Earth Sciences/Ceara Rise data'
file_name = 'C:/Users/罗腾霄/Desktop/研究生课程/combined_metrics.xlsx'
output_folder = 'C:/Users/罗腾霄/Desktop/研究生课程/Data Earth Sciences/Age_Grouped_Output'


concate(file_name, folder_path, output_folder, spilt=1)

print("处理完成")
