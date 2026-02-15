import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_for_clustering(df):
    """
    Selects Question columns and scales them for clustering.
    Returns scaled numpy array.
    """
    question_cols = [f'Q{i}' for i in range(1, 22)]
    X = df[question_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def run_kmeans(X, k=3, random_state=42):
    """
    Runs K-Means clustering.
    Returns: model, labels, silhouette_score
    """
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    return kmeans, labels, score

def run_gmm(X, n_components=3, random_state=42):
    """
    Runs Gaussian Mixture Model.
    Returns: model, labels, bic (Bayesian Information Criterion)
    """
    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    gmm.fit(X)
    labels = gmm.predict(X)
    bic = gmm.bic(X)
    # Silhouette score can also be calculated for GMM labels
    sil_score = silhouette_score(X, labels)
    return gmm, labels, bic, sil_score

def visualize_clusters_tsne(X, labels, title="t-SNE Visualization", save_path=None):
    """
    Runs t-SNE validation and plots the result.
    """
    tsne = TSNE(n_components=2, verbose=1, random_state=42)
    z = tsne.fit_transform(X)
    
    df_tsne = pd.DataFrame()
    df_tsne["y"] = labels
    df_tsne["comp-1"] = z[:,0]
    df_tsne["comp-2"] = z[:,1]

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x="comp-1", y="comp-2", hue=df_tsne.y.tolist(),
                    palette=sns.color_palette("hls", len(np.unique(labels))),
                    data=df_tsne).set(title=title)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show() # In case we run interactively later
