import logging
from .base_clusterer import BaseClusterer
from sklearn.cluster import SpectralClustering

logging.basicConfig(level=logging.INFO)


class SpectralClusterer(BaseClusterer):
    def __init__(self, n_clusters: int, n_neighbors: int = 10):
        super().__init__(n_clusters)
        self.n_neighbors = n_neighbors

    def fit_predict(self, features):
        if features.ndim != 2:
            raise ValueError("Input features must be a 2D array (n_samples, n_features).")
               
        num_samples = features.shape[0]
        k = self.n_clusters
        if k >= num_samples:
            k = max(2, num_samples - 1)
            logging.warning(f"n_clusters adjusted from {self.n_clusters} to {k} (n_samples={num_samples}).")
        
        logging.info(f"Clustering with SpectralClustering (n_clusters={k})...")
        model = SpectralClustering(
            n_clusters=k,
            affinity="nearest_neighbors",
            n_neighbors=min(self.n_neighbors, num_samples - 1),
            random_state=42
        )
        
        labels = model.fit_predict(features)
        return labels
    