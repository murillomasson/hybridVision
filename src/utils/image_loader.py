import os
import cv2
import logging
from glob import glob


class ImageLoader:
    def get_image_paths(self, directory: str) -> list:
        supported_formats = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_paths = []
        for fmt in supported_formats:
            image_paths.extend(glob(os.path.join(directory, fmt)))
        
        if not image_paths:
            logging.warning(f"No images found in directory: {directory}")
            
        return sorted(image_paths)

    def load_single_image(self, path: str, return_rgb: bool = False) -> 'np.ndarray':
        if not os.path.exists(path):
            logging.error(f"Image file not found at: {path}")
            return None
            
        try:
            image_bgr = cv2.imread(path)
            if image_bgr is None:
                logging.error(f"Failed to load image: {path}")
                return None

            if return_rgb:
                logging.debug("Converting image to RGB.")
                return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            else:
                return image_bgr
        except Exception as e:
            logging.error(f"Error processing image {path}: {e}")
            return None
        