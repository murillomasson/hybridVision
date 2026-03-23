import logging
from .kmeans_clusterer import KMeansClusterer
from .spectral_clusterer import SpectralClusterer
from .graph_clusterer import GraphClusterer

def create_clusterer(clusterer_config: dict, n_clusters: int):
    method = clusterer_config.get('method', 'kmeans').lower()
    logging.info(f"Creating clusterer of type: {method}")

    if method == 'kmeans':
        return KMeansClusterer(n_clusters=n_clusters)
    
    elif method == 'spectral':
        n_neighbors = clusterer_config.get('n_neighbors', None)
        return SpectralClusterer(n_clusters=n_clusters, n_neighbors=n_neighbors)
    
    elif method == 'graph':
        n_neighbors = clusterer_config.get('n_neighbors', None)
        return GraphClusterer(n_clusters=n_clusters, n_neighbors=n_neighbors)
    
    else:
        raise ValueError(f"Unknown clusterer method specified in config: {method}")