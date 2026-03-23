import numpy as np
import logging
import cv2


class FinalValidator:
    def __init__(self, config: dict):
        self.config = config or {}

        self.weights = self.config.get("weights", {})
        self.final_threshold = float(self.config.get("final_threshold", 0.5))
        self.semantic_threshold = float(self.config.get("semantic_threshold", 0.6))
        self.structural_threshold = float(self.config.get("structural_threshold", 0.5))

        self.clip_label_threshold = float(self.config.get("clip_label_threshold", 0.0))
        self.clip_margin_threshold = float(self.config.get("clip_margin_threshold", 0.0))
        self.clip_ambiguous_label = str(self.config.get("clip_ambiguous_label", "ambiguous"))

        logging.info(f"FinalValidator initialized with weights: {self.weights}")

    def _calculate_iou(self, mask1, mask2) -> float:
        intersection = np.logical_and(mask1, mask2)
        union = np.logical_or(mask1, mask2)
        denom = np.sum(union)
        return float(np.sum(intersection) / denom) if denom > 0 else 0.0

    def _find_best_match(self, target_mask, candidate_clusters):
        if candidate_clusters is None:
            return {"id": -1, "iou": 0.0}

        best_iou = -1.0
        best_id = -1

        # cv2.resize expects (W, H)
        target_shape_wh = (target_mask.shape[1], target_mask.shape[0])
        unique_ids = np.unique(candidate_clusters)

        for cluster_id in unique_ids:
            if cluster_id == -1:
                continue

            candidate_mask_low_res = (candidate_clusters == cluster_id)

            candidate_mask_high_res = cv2.resize(
                candidate_mask_low_res.astype(np.uint8),
                target_shape_wh,
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

            iou = self._calculate_iou(target_mask, candidate_mask_high_res)
            if iou > best_iou:
                best_iou = iou
                best_id = int(cluster_id)

        return {"id": best_id, "iou": float(best_iou)}

    def validate_and_label(
        self,
        sam_mask_data: dict,
        dino_results: dict,
        resnet_results: dict,
        clip_feat_map=None,
        clip_labeler=None,
    ):
        sam_mask = sam_mask_data.get("segmentation", None)
        if sam_mask is None:
            sam_mask = sam_mask_data.get("mask", None)

        if sam_mask is None:
            raw_scores = {
                "error": "sam_mask_missing",
                "dino_cluster_id": -1,
                "resnet_cluster_id": -1,
                "dino_iou": 0.0,
                "resnet_iou": 0.0,
                "clip_confidence": 0.0,
                "clip_margin": None,
                "clip_label": "unknown",
                "final_score": 0.0,
                "is_semantically_valid": False,
                "is_structurally_valid": False,
                "passed_any_iou_threshold": False,
                "semantic_threshold": float(self.semantic_threshold),
                "structural_threshold": float(self.structural_threshold),
                "final_threshold": float(self.final_threshold),
                "clip_label_threshold": float(self.clip_label_threshold),
                "clip_margin_threshold": float(self.clip_margin_threshold),
                "clip_ambiguous_label": str(self.clip_ambiguous_label),
                "is_ambiguous": False,
                "weights": dict(self.weights),
            }
            return None, raw_scores

        sam_mask = sam_mask.astype(bool)

        dino_match = {"id": -1, "iou": 0.0}
        resnet_match = {"id": -1, "iou": 0.0}

        dino_iou = 0.0
        resnet_iou = 0.0

        if dino_results and dino_results.get("clusters") is not None:
            dino_match = self._find_best_match(sam_mask, dino_results["clusters"])
            dino_iou = float(dino_match["iou"])

        if resnet_results and resnet_results.get("clusters") is not None:
            resnet_match = self._find_best_match(sam_mask, resnet_results["clusters"])
            resnet_iou = float(resnet_match["iou"])

        clip_conf = 0.0
        clip_label = "unknown"
        candidate_label = "unknown"
        clip_margin = None

        clip_info = sam_mask_data.get("clip_info", None)

        if isinstance(clip_info, dict):
            if "best_label" in clip_info:
                candidate_label = clip_info.get("best_label", "unknown")
                clip_label = candidate_label
                clip_conf = float(clip_info.get("best_similarity", 0.0))
                if "margin" in clip_info:
                    clip_margin = float(clip_info.get("margin", 0.0))

            elif "label" in clip_info:
                candidate_label = clip_info.get("label", "unknown")
                clip_label = candidate_label
                clip_conf = float(clip_info.get("confidence", clip_info.get("score", 0.0)))
                if "margin" in clip_info:
                    clip_margin = float(clip_info.get("margin", 0.0))

        elif (clip_labeler is not None) and (clip_feat_map is not None):
            try:
                tmp = clip_labeler.label_mask(
                    clip_feat_map=clip_feat_map,
                    sam_mask=sam_mask,
                    aggregation_method="top_k_mean",
                    top_k_perc=0.2,
                    top_n=5,
                    min_covered_patches=1,
                )
                if isinstance(tmp, dict):
                    clip_info = tmp  
                    clip_label = tmp.get("best_label", "unknown")
                    candidate_label = clip_label
                    clip_conf = float(tmp.get("best_similarity", 0.0))
                    if "margin" in tmp:
                        clip_margin = float(tmp.get("margin", 0.0))
            except Exception:
                clip_info = None
                clip_label = "unknown"
                candidate_label = "unknown"
                clip_conf = 0.0
                clip_margin = None

        if (dino_results is not None) and (dino_iou >= float(self.semantic_threshold)):
            dino_labels = dino_results.get("labels", None)
            if isinstance(dino_labels, dict):
                info = dino_labels.get(int(dino_match["id"]), None)
                if isinstance(info, dict) and "label" in info:
                    candidate_label = info.get("label", candidate_label)

        w_d = float(self.weights.get("dino_iou", 0.5))
        w_r = float(self.weights.get("resnet_iou", 0.5))
        w_c = float(self.weights.get("clip_confidence", 0.0))

        den = max(w_d + w_r + w_c, 1e-8)
        final_score = (
            w_d * dino_iou +
            w_r * resnet_iou +
            w_c * clip_conf
        ) / den

        is_semantically_valid = (dino_iou >= float(self.semantic_threshold))
        is_structurally_valid = (resnet_iou >= float(self.structural_threshold))
        passed_any_iou_threshold = (is_semantically_valid or is_structurally_valid)

        require_iou = (w_d > 0.0) or (w_r > 0.0)

        raw_scores = {
            "dino_cluster_id": int(dino_match["id"]),
            "resnet_cluster_id": int(resnet_match["id"]),
            "dino_iou": float(dino_iou),
            "resnet_iou": float(resnet_iou),
            "clip_confidence": float(clip_conf),
            "clip_margin": None if clip_margin is None else float(clip_margin),
            "clip_label": str(clip_label),
            "candidate_label": str(candidate_label),
            "final_score": float(final_score),
            "is_semantically_valid": bool(is_semantically_valid),
            "is_structurally_valid": bool(is_structurally_valid),
            "passed_any_iou_threshold": bool(passed_any_iou_threshold),
            "semantic_threshold": float(self.semantic_threshold),
            "structural_threshold": float(self.structural_threshold),
            "final_threshold": float(self.final_threshold),
            "clip_label_threshold": float(self.clip_label_threshold),
            "clip_margin_threshold": float(self.clip_margin_threshold),
            "clip_ambiguous_label": str(self.clip_ambiguous_label) if clip_margin is not None else None,
            "weights": dict(self.weights),
            "is_ambiguous": False,
        }

        logging.info(
            "[VAL_DEBUG] dino_id=%s dino_iou=%.3f | resnet_id=%s resnet_iou=%.3f | "
            "clip=%.3f | final=%.3f | sem_ok=%s str_ok=%s | label=%s",
            dino_match["id"], dino_iou,
            resnet_match["id"], resnet_iou,
            clip_conf, final_score,
            is_semantically_valid, is_structurally_valid,
            clip_label
        )

        if (require_iou and (not passed_any_iou_threshold)) or (final_score < float(self.final_threshold)):
            return None, raw_scores

        near_final = raw_scores["final_score"] >= (self.final_threshold - 0.03)
        near_struct = raw_scores["resnet_iou"] >= (self.structural_threshold - 0.05)
        near_sem = raw_scores["dino_iou"] >= (self.semantic_threshold - 0.05)

        if (clip_info is not None) and isinstance(clip_info, dict) and (near_final or near_struct or near_sem):
            tm = clip_info.get("top_matches", [])
            if tm:
                logging.info(
                    "[CLIP_TOP] label=%s conf=%.3f margin=%s top=%s",
                    raw_scores["clip_label"],
                    raw_scores["clip_confidence"],
                    str(raw_scores["clip_margin"]),
                    [(x["label"], round(x["similarity"], 3)) for x in tm[:5]]
        )

        if is_semantically_valid and is_structurally_valid:
            status = "VALIDATED_HYBRID"
        elif is_semantically_valid:
            status = "VALIDATED_SEMANTIC_ONLY"
        elif (not require_iou):
            status = "VALIDATED_CLIP_ONLY"
        else:
            status = "VALIDATED_STRUCTURAL_ONLY"

        clip_ok = (clip_conf >= self.clip_label_threshold) and (
            (clip_margin is None) or (clip_margin >= self.clip_margin_threshold)
        )

        if clip_ok:
            final_label = clip_label
        elif is_semantically_valid or is_structurally_valid:
            final_label = candidate_label
        else:
            final_label = "unknown"

        is_ambiguous = False
        if (clip_margin is not None) and (clip_conf >= self.clip_label_threshold) and (clip_margin < self.clip_margin_threshold):
            is_ambiguous = True
            final_label = self.clip_ambiguous_label

        raw_scores["is_ambiguous"] = bool(is_ambiguous)

        result = {
            "mask": sam_mask,
            "label": final_label,
            "clip_info": clip_info,
            "final_confidence": float(final_score),
            "status": status,
            "scores_breakdown": raw_scores,
        }
        return result, raw_scores
