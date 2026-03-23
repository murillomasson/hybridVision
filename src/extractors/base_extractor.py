import torch
import numpy as np
import logging
import cv2
from PIL import Image
from src.utils.image_loader import ImageLoader


class BaseExtractor:
    def __init__(self, device='cuda', output_dir='./output'):
        self.device = device
        self.output_dir = output_dir
        self.image_loader = ImageLoader()
        self.model_dtype = torch.float32 

    def _load_images_and_prepare_batch(self, image_paths_or_data):
        images_processed = []
        original_images = []

        for item in image_paths_or_data:
            img_bgr = None
            if isinstance(item, str):
                img_bgr = self.image_loader.load_single_image(item, return_rgb=False)
            elif isinstance(item, np.ndarray):
                img_bgr = item
            else:
                logging.warning(f"Unsupported image input type: {type(item)}")

            if img_bgr is not None:
                try:
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(img_rgb)
                    images_processed.append(self.preprocess(pil_image))
                    original_images.append(img_bgr)
                except Exception as e:
                    logging.error(f"Error preprocessing image: {item}. Error: {e}", exc_info=True)
            else:
                logging.warning(f"Image could not be loaded or was None: {item}")
        
        if not images_processed:
            return None, None
        
        image_batch = torch.stack(images_processed).to(self.device, dtype=self.model_dtype)
        
        return image_batch, original_images

    def extract(self, image_paths, cache_key):
        raise NotImplementedError("The extract() method must be implemented by the subclass.")
    