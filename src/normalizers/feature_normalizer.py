import logging
import numpy as np
from sklearn.preprocessing import normalize, StandardScaler, PowerTransformer

logging.basicConfig(level=logging.INFO)


class FeatureNormalizer:
    def __init__(self, method: str = 'l2'):
        self.method = method.lower()
        self.scaler = self._get_scaler()
        logging.info(f"FeatureNormalizer initialized with method: {self.method}")

    def _get_scaler(self):
        if self.method == 'l2':
            return None
        elif self.method == 'z-score':
            return StandardScaler()
        elif self.method == 'root-norm':
            return PowerTransformer(method='yeo-johnson', standardize=True)
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        if self.method == 'l2':
            return normalize(features, norm='l2', axis=1)
        
        return self.scaler.fit_transform(features)

def get_normalizer(method: str, **kwargs):
    return FeatureNormalizer(method=method)
