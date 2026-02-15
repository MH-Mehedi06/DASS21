import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Ensure we can import from src
sys.path.append(os.getcwd())

from src.data import load_and_clean_data
from src.features import calculate_scores
from src.clustering import preprocess_for_clustering, run_kmeans, run_gmm, visualize_clusters_tsne
from src.visualization import plot_correlation_matrix, plot_severity_distribution

def main():
    print("--- Starting Cluster Validation ---")
    
    # 1. Load Data
    data_path = "DASS 21 Dataset.xlsx" # Or .csv depending on what's available
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    df = load_and_clean_data(data_path)
    df = calculate_scores(df)
    
    # 2. Preprocess
    X_scaled = preprocess_for_clustering(df)
    
    # 3. K-Means (Original approach)
    print("\n--- Testing K-Means (k=3) ---")
    kmeans, km_labels, km_score = run_kmeans(X_scaled, k=3)
    print(f"K-Means Silhouette Score: {km_score:.4f}")
    
    # 4. GMM (New approach)
    print("\n--- Testing GMM (k=3) ---")
    gmm, gmm_labels, gmm_bic, gmm_sil = run_gmm(X_scaled, n_components=3)
    print(f"GMM BIC: {gmm_bic:.4f}")
    print(f"GMM Silhouette Score: {gmm_sil:.4f}")

    # 5. Visualize with t-SNE
    print("\n--- Generating t-SNE Plots ---")
    
    if not os.path.exists("plots"):
        os.makedirs("plots")

    # K-Means Plot
    visualize_clusters_tsne(X_scaled, km_labels, 
                            title=f"t-SNE with K-Means (Score: {km_score:.4f})", 
                            save_path="plots/tsne_kmeans.png")

    # GMM Plot
    visualize_clusters_tsne(X_scaled, gmm_labels, 
                            title=f"t-SNE with GMM (Score: {gmm_sil:.4f})", 
                            save_path="plots/tsne_gmm.png")

    # Severity Plot
    plot_severity_distribution(df, save_path="plots/severity_dist.png")

    print("\n--- Validation Complete ---")
    print(f"Plots saved to {os.getcwd()}/plots/")

if __name__ == "__main__":
    main()
