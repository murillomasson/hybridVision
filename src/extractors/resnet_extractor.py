import torch
import torchvision.models as models
import os
import logging
import numpy as np
import math
from .base_extractor import BaseExtractor
from ..utils.io import load_pickle, save_pickle

logging.basicConfig(level=logging.INFO)


class ResNetExtractor(BaseExtractor):
    def __init__(
        self,
        model_name: str = "resnet18",
        layer_name: str = "conv1",
        output_dir: str = "./output",
        device: str = "cuda"
    ):
        super().__init__(device, output_dir)
        
        self.model_name = model_name
        self.layer_name = layer_name
        self.target_features = {}
        self.cache_dir = os.path.join(self.output_dir, 'features_cache')
        os.makedirs(self.cache_dir, exist_ok=True)

        self._load_model()
        self._register_hook()

    def _load_model(self):
        try:
            logging.info(f"Loading TorchVision model: {self.model_name}...")
            weights = models.get_model_weights(self.model_name).DEFAULT
            self.model = getattr(models, self.model_name)(weights=weights).to(self.device).eval()
            self.preprocess = weights.transforms()
        except AttributeError:
            raise ValueError(f"Model {self.model_name} not found in torchvision.models.")
        except Exception as e:
            logging.error(f"Failed to load TorchVision model {self.model_name}: {e}")
            raise

    def _get_target_module(self, model, layer_name):
        path_parts = layer_name.split('.')
        current_module = model
        for part in path_parts:
            current_module = getattr(current_module, part, None)
            if current_module is None:
                return None
        return current_module

    def _register_hook(self):
        def hook_fn(module, input, output):
            self.target_features['output'] = output.cpu()

        target_module = self._get_target_module(self.model, self.layer_name)
        if target_module is None:
            raise ValueError(f"Layer '{self.layer_name}' not found in model {self.model_name}.")

        self.hook = target_module.register_forward_hook(hook_fn)
        logging.info(f"Extraction hook registered on layer: {self.layer_name}")

    def extract(self, image_paths, cache_key: str = "features"):
        unique_key = f"{cache_key}_{self.model_name.replace('/', '-')}_{self.layer_name.replace('.', '-')}"
        cache_filename = f"{unique_key}.pkl"
        cache_path = os.path.join(self.cache_dir, cache_filename)
        
        cached_data = load_pickle(cache_path)
        if cached_data is not None:
            features_np, paths = cached_data
            return features_np, paths

        image_batch, _ = self._load_images_and_prepare_batch(image_paths)
        if image_batch is None:
            logging.error("Image batch is None. Cannot proceed with ResNet extraction.")
            return None, None
        
        logging.info(f"Starting ResNet feature extraction from layer '{self.layer_name}'...")
        self.target_features = {}
        with torch.no_grad():
            self.model(image_batch)
        
        if 'output' not in self.target_features:
            raise RuntimeError(f"Hook failed to capture output from layer: {self.layer_name}")
            
        raw_output = self.target_features['output']
        features_list_np = [t.numpy() for t in raw_output]
        features_np = np.array(features_list_np)
        
        save_pickle((features_np, image_paths), cache_path)
        return features_np, cache_path
    