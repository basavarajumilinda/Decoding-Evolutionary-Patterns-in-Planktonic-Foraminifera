import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, silhouette_samples

# CPU Optimization
CPU_COUNT = max(1, os.cpu_count() - 3)
print(f"Detected {CPU_COUNT} cores, optimizing CPU usage.")

# Parallel computation for silhouette scores
def parallel_silhouette_score(df_scaled, labels, batch_size=None):
    n_samples = len(df_scaled)
    if batch_size is None:
        batch_size = min(10000, n_samples // CPU_COUNT)
    indices = np.array_split(np.arange(n_samples), max(1, n_samples // batch_size))
    silhouette_values = Parallel(n_jobs=CPU_COUNT, backend="loky")(
        delayed(silhouette_samples)(df_scaled[idx], labels[idx], metric='euclidean')
        for idx in indices
    )
    return np.concatenate(silhouette_values)

class ClusteringAnalysis:
    def __init__(self, file_path, output_dir):
        self.file_path = file_path
        self.output_dir = output_dir
        self.data = None
        self.features = None
        self.scaled_data = None

    def load_data(self):
        """Load and preprocess the data"""
        try:
            self.data = pd.read_csv(self.file_path)
            self.data.dropna(inplace=True)
            self.features = self.data.select_dtypes(include=[np.number])
            if self.features.empty:
                raise ValueError("No numeric columns found in the dataset.")
            self.scaled_data = StandardScaler().fit_transform(self.features)
        except Exception as e:
            print(f"Error loading file {self.file_path}: {e}")

    def evaluate_kmeans(self):
        """Determine the best number of clusters using the Elbow Method and Silhouette Score"""
        distortions = []
        silhouette_scores = []
        cluster_range = range(10, 30)
        # model_path = os.path.join(self.output_dir, "kmeans_model.pkl")

        for k in cluster_range:
            kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=100)
            labels = kmeans.fit_predict(self.scaled_data)
            distortions.append(kmeans.inertia_)
            silhouette_values = parallel_silhouette_score(self.scaled_data, labels)
            silhouette_scores.append(silhouette_values.mean())
            # joblib.dump(kmeans, model_path)
            
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)  # Ensure the directory exists

        # Save Elbow and Silhouette Score plots
        plot_path = os.path.join(self.output_dir, f"elbow_silhouette_{os.path.basename(self.file_path)}.png")
        plt.figure(figsize=(10, 4))

        # Elbow Method
        plt.subplot(1, 2, 1)
        plt.plot(cluster_range, distortions, marker='o', linestyle='--')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Inertia (Distortion)')
        plt.title('Elbow Method')

        # Silhouette Scores
        plt.subplot(1, 2, 2)
        plt.plot(cluster_range, silhouette_scores, marker='o', linestyle='--')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Score Analysis')

        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

        print(f"Elbow and Silhouette Score plot saved: {plot_path}")

        best_k = cluster_range[np.argmax(silhouette_scores)]
        return best_k

    def final_clustering(self, best_k, method='kmeans'):
        """Perform final clustering using the best K value"""
        # model_path = os.path.join(self.output_dir, f"{method}_final_model.pkl")
        if method == 'kmeans':
            model = MiniBatchKMeans(n_clusters=best_k, random_state=42, batch_size=100)
        elif method == 'gmm':
            model = GaussianMixture(n_components=best_k, random_state=42)
        else:
            print("Invalid method. Use 'kmeans' or 'gmm'.")
            return

        labels = model.fit_predict(self.scaled_data)
        self.data['Cluster'] = labels
        # joblib.dump(model, model_path)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        output_file = os.path.join(self.output_dir, f"clustered_{os.path.basename(self.file_path)}.xlsx")
        self.data.to_excel(output_file, index=False)
        print(f"Clustered data saved: {output_file}")

        # Save clustering plot
        plt.figure(figsize=(6, 4))
        sns.scatterplot(x=self.features.iloc[:, 0], y=self.features.iloc[:, 1], hue=labels, palette="viridis")
        plt.xlabel(self.features.columns[0])
        plt.ylabel(self.features.columns[1])
        plt.title(f"Clustering Results ({method.upper()})")
        plt.legend(title="Cluster")

        plot_path = os.path.join(self.output_dir, f"clustering_plot_{os.path.basename(self.file_path)}.png")
        plt.savefig(plot_path)
        plt.close()

        print(f"Clustering plot saved: {plot_path}")

def apply_clustering_to_all_bins(input_folder, output_folder):
    """Processes all bin CSV files from input_folder and saves results to output_folder"""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    bin_files = [f for f in os.listdir(input_folder) if f.endswith(".csv")]

    if not bin_files:
        print("No CSV files found in the input folder.")
        return

    for bin_file in bin_files:
        bin_path = os.path.join(input_folder, bin_file)
        bin_name = os.path.splitext(bin_file)[0]  # Extract bin name (e.g., bin_4.90-5.00)
        bin_output_dir = os.path.join(output_folder, bin_name)

        print(f"\nProcessing bin: {bin_name}")

        clustering = ClusteringAnalysis(bin_path, bin_output_dir)
        clustering.load_data()
        best_k = clustering.evaluate_kmeans()
        clustering.final_clustering(best_k, method='kmeans')

if __name__ == "__main__":
    INPUT_FOLDER = "Bins_Final"  # Change this to the actual folder containing your CSV files
    OUTPUT_FOLDER = "bins_clusters_output"

    apply_clustering_to_all_bins(INPUT_FOLDER, OUTPUT_FOLDER)
