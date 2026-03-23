import logging
from .base_clusterer import BaseClusterer
from sklearn.cluster import KMeans


class KMeansClusterer(BaseClusterer):
    def __init__(self, n_clusters: int):
        super().__init__(n_clusters)
        logging.info(f"KMeansClusterer initialized with n_clusters={self.n_clusters}")

    def fit_predict(self, features):
        if features.ndim != 2:
            raise ValueError("Input features for KMeans must be a 2D array (n_samples, n_features).")
        
        model = KMeans(n_clusters=self.n_clusters, random_state=42, n_init='auto')
        labels = model.fit_predict(features)
        
        return labels
    