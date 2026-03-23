import numpy as np
import logging
import os
import hashlib
from src.wrappers.sam2_wrapper import Sam2Wrapper
from src.utils.io import load_pickle, save_pickle


class ClusterCountEstimator:
    def __init__(self, configs: dict, device: str, output_dir: str):
        self.configs = configs
        self.cache_dir = os.path.join(output_dir, 'k_estimator_cache')
        os.makedirs(self.cache_dir, exist_ok=True)

        self.sam_wrapper = Sam2Wrapper(
            model_name=self.configs['sam_model'],
            generator_params=self.configs['generator_params'],
            device=device
        )

    def _calculate_iou(self, mask1, mask2):
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return intersection / union if union > 0 else 0

    def _apply_nms(self, masks: list, iou_threshold: float) -> list:
        if not masks:
            return []

        masks = sorted(masks, key=lambda x: x.get('predicted_iou', 0), reverse=True)
        
        kept_masks = []
        suppressed_indices = set()

        for i in range(len(masks)):
            if i in suppressed_indices:
                continue
            
            kept_masks.append(masks[i])
            
            for j in range(i + 1, len(masks)):
                if j in suppressed_indices:
                    continue
                
                iou = self._calculate_iou(masks[i]['segmentation'], masks[j]['segmentation'])
                if iou > iou_threshold:
                    suppressed_indices.add(j)
        
        logging.info(f"NMS applied. Kept {len(kept_masks)} of {len(masks)} masks.")
        return kept_masks

    def _filter_and_process_masks(self, masks: list, image_shape: tuple) -> list:
        if not masks:
            return []
        
        filter_cfg = self.configs.get('filter_params', {})
        
        min_area_perc = filter_cfg.get('filter_min_area_perc', 0.005)
        min_area = image_shape[0] * image_shape[1] * min_area_perc
        filtered_masks = [m for m in masks if m['area'] > min_area]
        
        if filter_cfg.get('use_nms', False):
            iou_threshold = filter_cfg.get('nms_iou_threshold', 0.7)
            filtered_masks = self._apply_nms(filtered_masks, iou_threshold)

        return filtered_masks
        
    def estimate_k_with_heuristics(self, image: np.ndarray) -> dict:
        image_hash = hashlib.md5(image.tobytes()).hexdigest()
        params_hash = str(hash(str(self.configs)))
        cache_path = os.path.join(self.cache_dir, f"k_estimates_{image_hash}_{params_hash}.pkl")

        cached_data = load_pickle(cache_path)
        if cached_data:
            logging.info(f"Loaded k_semantic ({cached_data.get('k_semantic')}) and k_structural ({cached_data.get('k_structural')}) from cache.")
            return cached_data

        raw_masks = self.sam_wrapper.generate_masks(image)
        processed_masks = self._filter_and_process_masks(raw_masks, image.shape)
        num_filtered_masks = len(processed_masks)
        
        formulas = self.configs.get('k_heuristic_formulas', {})
        max_k = self.configs.get('max_k', 20)

        multiplier = formulas.get('structural_multiplier', 2.0)
        k_structural = int(multiplier * (num_filtered_masks ** 0.5))
        
        base = formulas.get('semantic_base', 2)
        scale_factor = formulas.get('semantic_scale_factor', 0.8)
        k_semantic = int(base + (num_filtered_masks * scale_factor))
        
        k_structural = max(2, min(k_structural, max_k))
        k_semantic = max(2, min(k_semantic, max_k))

        logging.info(f"Heuristics estimated: k_semantic={k_semantic}, k_structural={k_structural}")

        data_to_cache = {
            'k_semantic': k_semantic,
            'k_structural': k_structural,
            'processed_masks': processed_masks
        }
        save_pickle(data_to_cache, cache_path)
        
        return data_to_cache
    