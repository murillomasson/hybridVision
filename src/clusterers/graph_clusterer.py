import logging
import numpy as np
from .base_clusterer import BaseClusterer
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans

logging.basicConfig(level=logging.INFO)

class GraphClusterer(BaseClusterer):
    def __init__(self, n_clusters: int, n_neighbors: int = None):
        super().__init__(n_clusters)
        self.n_neighbors = n_neighbors

    def _calculate_dynamic_neighbors(self, num_patches: int) -> int:
        n_neighbors = int(round(np.sqrt(num_patches)))
        clamped_n = np.clip(n_neighbors, 6, 30)
        logging.info(f"Using dynamic heuristic for n_neighbors: sqrt({num_patches}) -> {n_neighbors} -> clamped to {clamped_n}")
        return clamped_n

    def fit_predict(self, features: np.ndarray):
        if features.ndim != 2:
            raise ValueError("Input features must be a 2D array (n_samples, n_features).")

        n_samples = features.shape[0]
        k = self.n_clusters
        if k >= n_samples:
            k = max(2, n_samples - 1)
            logging.warning(f"n_clusters adjusted from {self.n_clusters} to {k} (n_samples={n_samples}).")

        if self.n_neighbors is None:
            n_neighbors_cut = self._calculate_dynamic_neighbors(n_samples)
        else:
            logging.info(f"Using fixed n_neighbors provided in config: {self.n_neighbors}")
            n_neighbors_cut = min(self.n_neighbors, n_samples - 1)
        
        logging.info(f"Applying Normalized Cut with n_clusters={k}, n_neighbors={n_neighbors_cut}...")
        
        graph = kneighbors_graph(features, n_neighbors=n_neighbors_cut, mode='connectivity', metric='cosine', include_self=False)
        laplacian = csgraph.laplacian(graph, normed=True)
        
        try:
            _, eigenvectors = eigsh(laplacian, k=k, which='SM', tol=1e-4)
        except Exception:
            logging.warning("Eigenvector computation failed, likely due to a disconnected graph. Retrying with a denser graph.")
            graph = kneighbors_graph(features, n_neighbors=n_neighbors_cut * 2, mode='connectivity', metric='cosine', include_self=False)
            laplacian = csgraph.laplacian(graph, normed=True)
            _, eigenvectors = eigsh(laplacian, k=k, which='SM', tol=1e-4)

        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(eigenvectors)
        
        return labels
    