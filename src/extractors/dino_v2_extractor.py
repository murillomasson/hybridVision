import os
import logging
import torch
import numpy as np
import math
from torchvision import transforms
from PIL import Image
from .base_extractor import BaseExtractor
from ..utils.io import load_pickle, save_pickle


class DINOv2Extractor(BaseExtractor):
    def __init__(self, model_name='dinov2_vitb14', layer_name='blocks.11', output_dir='./output', device='cuda'):
        super().__init__(device, output_dir)
        self.model_name = model_name
        self.layer_name = layer_name
        self.target_features = {}
        self.cache_dir = os.path.join(self.output_dir, 'features_cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        logging.info(f"Loading DINOv2 model: {self.model_name}...")
        self.model = torch.hub.load('facebookresearch/dinov2', self.model_name)
        self.model.to(self.device)
        self.model.eval()
        
        self.preprocess = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        
        self._register_hook()

    def _get_features_hook(self, module, input, output):
        if isinstance(output, torch.Tensor):
            self.target_features['output'] = output
        elif isinstance(output, tuple):
            self.target_features['output'] = output[0]

    def _register_hook(self):
        layer_found = False
        for name, module in self.model.named_modules():
            if name == self.layer_name:
                module.register_forward_hook(self._get_features_hook)
                layer_found = True
                logging.info(f"Extraction hook registered on layer: {self.layer_name}")
                break
        if not layer_found:
            raise ValueError(f"Layer '{self.layer_name}' not found in model '{self.model_name}'")

    def extract(self, image_paths, cache_key: str = "features"):
        unique_key = f"{cache_key}_{self.model_name.replace('/', '-')}_{self.layer_name.replace('.', '-')}"
        cache_filename = f"{unique_key}.pkl"
        cache_path = os.path.join(self.cache_dir, cache_filename)
        
        cached_data = load_pickle(cache_path) 
        if cached_data is not None:
            logging.info(f"Loaded features from cache: {cache_path}")
            return cached_data

        image_batch, _ = self._load_images_and_prepare_batch(image_paths)
        if image_batch is None:
            logging.error("Image batch is None. Cannot proceed with DINOv2 extraction.")
            return None, None

        logging.info(f"Starting DINOv2 feature extraction from layer '{self.layer_name}'...")
        self.target_features = {} 
        
        with torch.no_grad():
            self.model(image_batch)
        
        if 'output' not in self.target_features:
            raise RuntimeError(f"Hook failed to capture output from layer: {self.layer_name}")
            
        raw_output = self.target_features['output']
        
        features_list_np = []
        for i in range(raw_output.shape[0]):
            feature_map = raw_output[i].squeeze()
            if feature_map.ndim == 2:
                if "attn" not in self.layer_name:
                    feature_map = feature_map[1:, :]
                
                N_patches, E_dim = feature_map.shape
                grid_size = int(math.sqrt(N_patches))
                if grid_size * grid_size != N_patches:
                    logging.error(f"Cannot form a square grid with {N_patches} patches. Aborting.")
                    return None, None
                
                reshaped_map = feature_map.reshape(grid_size, grid_size, E_dim).permute(2, 0, 1)
                features_list_np.append(reshaped_map.cpu().numpy())
            elif feature_map.ndim == 3:
                features_list_np.append(feature_map.cpu().numpy())
            else:
                logging.warning(f"Feature map has unexpected dimension {feature_map.ndim}, skipping.")

        features_np = np.array(features_list_np)
        
        save_pickle((features_np, image_paths), cache_path)
        return features_np, cache_path
    