import numpy as np
from scipy.stats import entropy

class TuningAnalyzer:
    """
    Calcula um score de performance v3.0 para um trial de otimização,
    usando uma função objetivo com curvatura para guiar o otimizador.
    """
    def __init__(self, pipeline_output: dict, diagnostic_data: dict, expected_labels: list, objective_weights: dict):
        self.output = pipeline_output
        self.diagnostic_data = diagnostic_data
        self.expected_labels = expected_labels
        self.weights = objective_weights
        self.image_name = list(self.output.keys())[0] if self.output else None
        self.validated_masks = self.output.get(self.image_name, []) if self.image_name else []

    def _calculate_segment_penalty(self, num_segments: int) -> float:
        """
        Calcula uma penalidade se o número de segmentos estiver fora do intervalo ideal.
        """
        min_target = self.weights.get('segment_target_range', {}).get('min', 2)
        max_target = self.weights.get('segment_target_range', {}).get('max', 10)
        
        if num_segments < min_target:
            return min_target - num_segments
        elif num_segments > max_target:
            return num_segments - max_target
        return 0.0

    def calculate_all_metrics(self) -> dict:
        """
        Calcula todas as métricas de análise e o score final v3.0 para o Optuna.
        """
        all_dino_ious = [m['scores_breakdown'].get('dino_iou', 0.0) for m in self.validated_masks]
        all_resnet_ious = [m['scores_breakdown'].get('resnet_iou', 0.0) for m in self.validated_masks]
        all_clip_confidences = [m['scores_breakdown'].get('clip_confidence', 0.0) for m in self.validated_masks]

        metrics = {}
        
        metrics['num_segments'] = len(self.validated_masks)
        metrics['avg_clip_confidence'] = np.mean(all_clip_confidences) if all_clip_confidences else 0.0

        if self.validated_masks:
            entropies = []
            for m in self.validated_masks:
                if 'labels' in m and 'top_matches' in m['labels']:
                    sims = np.array([match['similarity'] for match in m['labels']['top_matches']])
                    if len(sims) > 1:
                        probs = np.exp(sims) / np.sum(np.exp(sims))
                        entropies.append(entropy(probs))
            metrics['avg_entropy_labels'] = np.mean(entropies) if entropies else 0.0
        else:
            metrics['avg_entropy_labels'] = 0.0

        best_ious_per_mask = [max(d, r) for d, r in zip(all_dino_ious, all_resnet_ious)]
        metrics['iou_variance'] = np.var(best_ious_per_mask) if best_ious_per_mask else 0.0

        best_dino_iou = max(all_dino_ious) if all_dino_ious else 0.0
        best_resnet_iou = max(all_resnet_ious) if all_resnet_ious else 0.0
        
        metrics['mean_iou'] = (best_dino_iou + best_resnet_iou) / 2.0
        metrics['consistency_reward'] = 1.0 - abs(best_dino_iou - best_resnet_iou)
        metrics['iou_ratio_dino_resnet'] = (best_dino_iou + 1e-6) / (best_resnet_iou + 1e-6)

        target_label = self.expected_labels[0] if self.expected_labels else None
        semantic_score = 0.0
        for mask_data in self.validated_masks:
            if 'labels' in mask_data and 'top_matches' in mask_data['labels']:
                for candidate in mask_data['labels']['top_matches']:
                    if candidate['label'] == target_label and candidate['similarity'] > semantic_score:
                        semantic_score = candidate['similarity']
        metrics['semantic_score'] = max(0.0, semantic_score)

        metrics['semantic_structural_alignment'] = metrics['semantic_score'] * metrics['mean_iou']
        
        alpha = self.weights.get('alpha', 0.5)
        beta = self.weights.get('beta', 0.5)
        gamma = self.weights.get('gamma', 0.05)
        
        base_semantic_score = metrics.get('semantic_score', 0.0)
        structural_bonus = alpha * (metrics.get('mean_iou', 0.0) + (beta * metrics.get('consistency_reward', 0.0)))
        segment_penalty = gamma * self._calculate_segment_penalty(metrics.get('num_segments', 0))
        
        final_score = base_semantic_score + structural_bonus - segment_penalty
        
        metrics['value'] = max(0.0, final_score)

        if self.diagnostic_data:
            metrics.update(self.diagnostic_data)
        correct_segments_found = [m for m in self.validated_masks if m.get('label') in self.expected_labels]
        metrics['num_correct_segments'] = len(correct_segments_found)
            
        return metrics