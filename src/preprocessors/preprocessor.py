import cv2
import numpy as np
import logging


class Preprocessor:
    def __init__(self, config: dict):
        self.config = config
        logging.info(f"Preprocessor initialized with config: {config}")

    def process(self, image: np.ndarray) -> np.ndarray:
        processed_image = image.copy()

        if self.config.get('use_clahe', False):
            logging.info("Applying CLAHE...")
            clahe_clip_limit = self.config.get('clahe_clip_limit', 2.0)
            clahe_tile_size = tuple(self.config.get('clahe_tile_grid_size', [8, 8]))
            
            lab_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab_image)
            
            clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_size)
            l_channel_eq = clahe.apply(l_channel)
            
            lab_image_eq = cv2.merge((l_channel_eq, a_channel, b_channel))
            processed_image = cv2.cvtColor(lab_image_eq, cv2.COLOR_LAB2BGR)

        if self.config.get('use_grayscale', False):
            logging.info("Converting to grayscale...")
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)

        if self.config.get('use_gaussian_blur', False):
            k_size = self.config.get('blur_kernel_size', 5)
            logging.info(f"Applying Gaussian Blur with kernel size {k_size}...")
            if k_size % 2 == 0:
                k_size += 1
            processed_image = cv2.GaussianBlur(processed_image, (k_size, k_size), 0)
            
        if self.config.get('use_sobel_edges', False):
            logging.info("Applying Sobel edge enhancement...")
            if len(processed_image.shape) > 2:
                gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = processed_image

            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=self.config.get('sobel_kernel_size', 5))
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=self.config.get('sobel_kernel_size', 5))
            
            processed_image = cv2.magnitude(sobelx, sobely)
            processed_image = cv2.normalize(processed_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        if len(processed_image.shape) == 2:
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)

        return processed_image
    