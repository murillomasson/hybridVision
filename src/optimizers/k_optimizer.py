import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from src.clusterers.utils_clusterer import create_clusterer


class KValueOptimizer:
    def __init__(self, clusterer_config: dict):
        self.clusterer_config = clusterer_config

    def find_best_k(self, features: np.ndarray, k_range: tuple, plot_path: str = None) -> int:
        scores = []
        k_values = range(k_range[0], k_range[1] + 1)
        
        logging.info(f"Starting optimal k search in range {k_range} using Silhouette Score...")

        for k in k_values:
            if k >= features.shape[0]:
                logging.warning(f"k={k} is >= number of samples ({features.shape[0]}). Stopping search.")
                break
            try:
                clusterer = create_clusterer(self.clusterer_config, n_clusters=k)
                labels = clusterer.fit_predict(features)
                
                if len(np.unique(labels)) < 2:
                    score = -1
                else:
                    metric = 'cosine' if np.allclose(np.linalg.norm(features, axis=1), 1.0) else 'euclidean'
                    score = silhouette_score(features, labels, metric=metric)
      
                scores.append(score)
                logging.debug(f"  k={k}, Silhouette Score = {score:.4f}")

            except Exception as e:
                logging.error(f"  Failed to cluster with k={k}: {e}", exc_info=True)
                scores.append(-1)

        if not scores:
            logging.error("No silhouette scores were calculated. Returning default k=8.")
            return 8
            
        best_k_index = np.argmax(scores)
        best_k = k_values[best_k_index]
        best_score = scores[best_k_index]
        
        logging.info(f"Search complete. Best k = {best_k} with Silhouette Score = {best_score:.4f}")

        if plot_path:
            self._plot_scores(k_values[:len(scores)], scores, best_k, plot_path)

        return best_k

    def _plot_scores(self, k_values, scores, best_k, save_path):
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, scores, marker='o', linestyle='--')
        plt.axvline(x=best_k, color='r', linestyle=':', label=f'Best k = {best_k}')
        plt.title('Silhouette Score vs. Number of Clusters (k)')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Average Silhouette Score')
        plt.xticks(k_values)
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
        logging.info(f"k-optimization plot saved to: {save_path}")
        