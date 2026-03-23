import numpy as np
import pandas as pd
import logging


class PerformanceAnalyzer:
    def __init__(self, pipeline_output: dict, ground_truth_data: dict):
        self.output = pipeline_output
        self.gt = ground_truth_data
        self.detailed_metrics_df = None

    def _calculate_iou(self, pred_mask, gt_mask):
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        return intersection / union if union > 0 else 0

    def _find_best_gt_match(self, pred_mask, gt_masks_for_image):
        best_iou = -1
        best_gt_mask_data = None
        for gt_mask_data in gt_masks_for_image:
            iou = self._calculate_iou(pred_mask, gt_mask_data['segmentation'])
            if iou > best_iou:
                best_iou = iou
                best_gt_mask_data = gt_mask_data
        return best_gt_mask_data, best_iou

    def run_evaluation(self):
        logging.info("Starting performance evaluation against ground truth...")
        all_metrics = []

        for image_name, validated_masks in self.output.items():
            if image_name not in self.gt:
                logging.warning(f"No ground truth found for image {image_name}. Skipping.")
                continue

            gt_masks_for_image = self.gt[image_name]
            
            for i, pred_mask_data in enumerate(validated_masks):
                pred_mask = pred_mask_data['mask']
                
                best_gt_match, iou = self._find_best_gt_match(pred_mask, gt_masks_for_image)
                
                if best_gt_match is None:
                    continue

                metrics_row = {
                    'image_name': image_name,
                    'mask_id': i,
                    'predicted_label': pred_mask_data['label'],
                    'gt_label': best_gt_match['label'],
                    'is_label_correct': pred_mask_data['label'] == best_gt_match['label'],
                    'iou': iou,
                    'model_confidence': pred_mask_data['final_confidence'],
                    **pred_mask_data['scores_breakdown']
                }
                all_metrics.append(metrics_row)
        
        self.detailed_metrics_df = pd.DataFrame(all_metrics)
        logging.info("Performance evaluation finished.")
        return self.detailed_metrics_df

    def get_summary(self) -> dict:
        """Calculates and returns aggregated metrics for the entire run."""
        if self.detailed_metrics_df is None or self.detailed_metrics_df.empty:
            logging.warning("Detailed metrics DataFrame is not available. Cannot generate summary.")
            return {}

        summary = {
            'total_images': len(self.output),
            'total_validated_masks': len(self.detailed_metrics_df),
            'mean_iou': self.detailed_metrics_df['iou'].mean(),
            'median_iou': self.detailed_metrics_df['iou'].median(),
            'label_accuracy': self.detailed_metrics_df['is_label_correct'].mean(),
            'mean_model_confidence': self.detailed_metrics_df['model_confidence'].mean()
        }
        return summary
    