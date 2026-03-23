import os
import logging
import math
import numpy as np
import torch
import clip
from PIL import Image
from .base_extractor import BaseExtractor
from ..utils.io import load_pickle, save_pickle

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CLIPExtractor(BaseExtractor):
    """
    Extracts feature maps from an intermediate layer of CLIP's visual encoder.
    """
    def __init__(
        self,
        model_name: str = "ViT-B/32",
        layer_name: str = "transformer.resblocks.11",
        output_dir: str = "./output",
        device: str = "cuda"
    ):
        super().__init__(device=device, output_dir=output_dir)
        
        self.model_name = model_name
        self.layer_name = layer_name
        self.captured_features = None

        self.cache_dir = os.path.join(self.output_dir, 'features_cache')
        os.makedirs(self.cache_dir, exist_ok=True)

        self._load_model()
        self._register_hook()

    def _load_model(self):
        """ Loads the full CLIP model and ensures it uses 32-bit precision. """
        try:
            logging.info(f"Loading CLIP model: {self.model_name}...")
            self.full_model, self.preprocess = clip.load(self.model_name, device=self.device)
            self.full_model.float() 
            self.visual_model = self.full_model.visual
            self.visual_model.eval()
        except Exception as e:
            logging.error(f"Failed to load CLIP model: {e}")
            raise

    def _get_target_module(self, model, layer_path_str):
        """ Navigates the model architecture to find the target layer for the hook. """
        current_module = model
        for part in layer_path_str.split('.'):
            current_module = getattr(current_module, part, None)
            if current_module is None:
                raise ValueError(f"Layer '{layer_path_str}' not found in the model.")
        return current_module

    def _hook_fn(self, module, input, output):
        """ The hook function, which captures the output of the target layer. """
        self.captured_features = output

    def _register_hook(self):
        """ Registers the hook on the target visual layer. """
        try:
            target_module = self._get_target_module(self.visual_model, self.layer_name)
            target_module.register_forward_hook(self._hook_fn)
            logging.info(f"Hook successfully registered on layer: visual.{self.layer_name}")
        except ValueError as e:
            logging.error(e)
            raise

    def extract(self, image_paths_or_data, cache_key: str = "features"):
        """
        Main method to extract features. Includes caching logic.
        """
        unique_key = f"{cache_key}_{self.model_name.replace('/', '-')}_{self.layer_name}"
        cache_path = os.path.join(self.cache_dir, f"{unique_key}.pkl")
        
        cached_data = load_pickle(cache_path)
        if cached_data:
            logging.info(f"Loading cached CLIP features from {cache_path}")
            return cached_data

        logging.info(f"Starting CLIP feature extraction from layer '{self.layer_name}'...")
        
        images_pil = self._load_images_as_pil(image_paths_or_data)
        if not images_pil:
            return None, None
        
        image_batch = torch.stack([self.preprocess(img) for img in images_pil]).to(self.device)
        
        self.captured_features = None
        with torch.no_grad():
            self.visual_model(image_batch)

        if self.captured_features is None:
            raise RuntimeError(f"Hook for layer '{self.layer_name}' was not activated.")

        raw_output = self.captured_features
        raw_output = raw_output.permute(1, 0, 2)
        patch_features = raw_output[:, 1:, :]

        ln_post = self.visual_model.ln_post
        proj_matrix = self.visual_model.proj
        projected_features = ln_post(patch_features) @ proj_matrix
        
        B, N_patches, C_final = projected_features.shape
        grid_size = int(math.sqrt(N_patches))
        if grid_size * grid_size != N_patches:
            raise ValueError(f"Cannot form a square grid with {N_patches} patches.")

        features_map = projected_features.reshape(B, grid_size, grid_size, C_final).permute(0, 3, 1, 2)

        features_np = features_map.cpu().detach().numpy()

        
        logging.info(f"Extraction complete. Final features shape: {features_np.shape}")

        save_pickle((features_np, image_paths_or_data), cache_path)
        logging.info(f"CLIP features saved to: {cache_path}")

        return features_np, image_paths_or_data

    def _load_images_as_pil(self, image_paths_or_data):
        """ Helper to load images, ensuring they are in PIL.Image format. """
        if not image_paths_or_data: return []
        
        if isinstance(image_paths_or_data[0], str):
            images_pil = []
            for path in image_paths_or_data:
                try:
                    images_pil.append(Image.open(path).convert('RGB'))
                except Exception as e:
                    logging.warning(f"Could not load image {path}: {e}")
            return images_pil
        elif isinstance(image_paths_or_data[0], np.ndarray):
            return [Image.fromarray(img) for img in image_paths_or_data]
        else:
            logging.error("Unsupported image input format.")
            return None