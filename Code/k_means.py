import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from tqdm import tqdm


def read_age_group_file(file_path):
    """
    Read the Excel file for each age group and return the DataFrame.
Select only the four columns you need as features
    """
    try:
        df = pd.read_excel(file_path)
        # 只保留需要的四列
        columns_needed = ['shapefactor', 'meandiameterµm', 'elongation', 'sphericity']
        if not all(col in df.columns for col in columns_needed):
            print(f"File {os.path.basename(file_path)} Skip this file because necessary columns are missing.")
            return None
        return df[columns_needed]
    except Exception as e:
        print(f"Error when reading {os.path.basename(file_path)} {e}")
        return None


def plot_elbow_method(df_scaled):
    """
    Draw a diagram of the elbow rule
    """
    sse = []
    k_range = range(2, 25)  # 尝试 2 到 15 个簇

    for k in k_range:
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=4096, n_init=10)
        kmeans.fit(df_scaled)
        sse.append(kmeans.inertia_)  # 误差平方和 (SSE)

    # 绘制肘部法则图
    plt.figure(figsize=(8, 6))
    plt.plot(k_range, sse, marker='o', label='SSE')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Sum of Squared Errors (SSE)')
    plt.title('Elbow Method for Optimal K')
    plt.legend()
    plt.show()


def plot_silhouette_scores(df_scaled):
    """
    Plot the contour coefficients of different K values
    """
    silhouette_scores = []
    k_range = range(2, 25)  # 尝试 2 到 15 个簇

    for k in k_range:
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=4096, n_init=10)
        kmeans.fit(df_scaled)
        score = silhouette_score(df_scaled, kmeans.labels_)
        silhouette_scores.append(score)

    # 绘制轮廓系数图
    plt.figure(figsize=(8, 6))
    plt.plot(k_range, silhouette_scores, marker='o', label='Silhouette Score', color='red')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Different K')
    plt.legend()
    plt.show()

    # 返回最佳 K（最大轮廓系数的 K）
    best_k = k_range[np.argmax(silhouette_scores)]
    print(f"The best K value recommended based on the profile coefficient: {best_k}")
    return best_k


def apply_minibatch_kmeans(df, n_clusters):
    """
    MiniBatchKMeans Cluster
    """
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    # 使用 MiniBatchKMeans 进行聚类
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=4096, n_init=10)
    df['Cluster'] = kmeans.fit_predict(df_scaled)

    # 最终K
    n_clusters = df['Cluster'].nunique()
    print(f"Number of final clustering categories: {n_clusters}")

    # 每个sample数量
    cluster_counts = df['Cluster'].value_counts().sort_index()
    print("Sample of each cluster:")
    print(cluster_counts)

    # 可视化聚类结果（不同特征组合）
    features = df.columns[:-1]
    n_features = len(features)

    plt.figure(figsize=(15, 10))
    plot_index = 1
    for i in range(n_features):
        for j in range(i + 1, n_features):
            plt.subplot(2, 3, plot_index)  # 2 行 3 列
            plt.scatter(df[features[i]], df[features[j]], c=df['Cluster'], cmap='viridis', alpha=0.6, s=20)
            plt.xlabel(features[i])
            plt.ylabel(features[j])
            plt.title(f"{features[i]} vs {features[j]}")
            plot_index += 1
    plt.tight_layout()
    plt.show()

    return df, kmeans


def process_age_group_files(folder_path, output_folder):
    """
    Process Excel files of all ages and perform MiniBatchKMeans clustering
    """
    os.makedirs(output_folder, exist_ok=True)

    # 获取文件夹中的所有 Excel 文件
    files = [file for file in os.listdir(folder_path) if file.endswith('.xlsx')]

    # 处理每个文件
    for file_name in files:
        file_path = os.path.join(folder_path, file_name)

        # 读取每个年龄段的文件
        df = read_age_group_file(file_path)
        if df is None:  # 如果缺少必要列 跳过该文件
            continue

        # 显示文件行数
        file_rows = df.shape[0]
        print(f"File {file_name} Row: {file_rows}")

        # 使用 tqdm 显示数据行数进度条
        with tqdm(total=file_rows, desc=f"处理 {file_name} 数据", unit="row") as pbar:
            for _ in range(file_rows):  # 遍历数据行
                pbar.update(1)

        # 标准化数据
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df)

        # 绘制肘部法则图
        plot_elbow_method(df_scaled)

        #best_k_silhouette = plot_silhouette_scores(df_scaled)

        # 手动选择最佳K
        n_clusters = int(input(f"The clustering results are saved: "))


        df, kmeans = apply_minibatch_kmeans(df, n_clusters)

        # 保存结果
        output_file = os.path.join(output_folder, f"clustered_{file_name}")
        df.to_excel(output_file, index=False)
        print(f"The clustering results are saved: {output_file}")


# 读取储存路径
folder_path = 'C:/Users/罗腾霄/Desktop/研究生课程/Data Earth Sciences/Age_Grouped_Output'
output_folder = 'C:/Users/罗腾霄/Desktop/研究生课程/Data Earth Sciences/Clustered_Age_Groups'

# 处理文件
process_age_group_files(folder_path, output_folder)

print("聚类处理完成！")
