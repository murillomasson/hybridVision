import logging
from sklearn.decomposition import PCA, IncrementalPCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap

logging.basicConfig(level=logging.INFO)

def get_reducer(method: str, n_components: int, **kwargs):
    method = method.lower()
    logging.info(f"Initializing dimensionality reducer: {method} with n_components={n_components}")

    if method == "pca":
        return PCA(n_components=n_components, random_state=42)
    elif method == "umap":
        return umap.UMAP(
            n_components=n_components,
            n_neighbors=kwargs.get('n_neighbors', 15),
            metric=kwargs.get('metric', 'cosine'),
            random_state=42,
            init='random'
        )
    elif method == "incremental_pca":
        return IncrementalPCA(n_components=n_components, batch_size=kwargs.get('batch_size', 256))
    elif method == "svd":
        return TruncatedSVD(n_components=n_components, random_state=42)
    else:
        raise ValueError(f"Unknown dimensionality reduction method: {method}")
    