from abc import ABC, abstractmethod


class BaseClusterer(ABC):
    def __init__(self, n_clusters: int):
        self.n_clusters = n_clusters

    @abstractmethod
    def fit_predict(self, features):
        pass
    