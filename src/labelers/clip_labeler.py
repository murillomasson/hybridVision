import os
import json
import logging
import torch
import clip
import numpy as np
import cv2
from sklearn.preprocessing import normalize
from PIL import Image

logging.basicConfig(level=logging.INFO)


class ClipLabeler:
    def __init__(
            self, 
            words_json_path: str, 
            prompt_template="a photo of a {}", 
            device: str = 'cuda',
            context_enabled: bool = False,
            context_scale: float = 1.3,
        ):
        self.device = device
        self.context_enabled = context_enabled
        self.context_scale = float(context_scale)

        self.model, _ = clip.load("ViT-B/32", device=self.device)
        self.model = self.model.to(torch.float32)
        self.model.visual.eval()

        self.text_features, self.word_list = self._load_and_encode_words(words_json_path, prompt_template)
        self.word_to_idx = {w: i for i, w in enumerate(self.word_list)}

        if self.text_features is None:
            raise ValueError("Could not load or encode words from the provided JSON path.")

    def _load_and_encode_words(self, json_path: str, prompt_template):
        if not os.path.exists(json_path):
            logging.error(f"Words JSON file not found at: {json_path}")
            return None, None

        with open(json_path, 'r') as f:
            word_list = json.load(f)

        if not isinstance(word_list, list) or not all(isinstance(s, str) for s in word_list):
            raise TypeError(f"Expected '{json_path}' to contain a JSON list of strings.")

        templates = [prompt_template] if isinstance(prompt_template, str) else prompt_template
        logging.info(f"Encoding {len(word_list)} words with {len(templates)} templates each...")

        final_text_features = []
        with torch.no_grad():
            for word in word_list:
                prompts_for_word = [template.format(word) for template in templates]
                text_tokens = clip.tokenize(prompts_for_word).to(self.device)
                template_embeddings = self.model.encode_text(text_tokens) 
                mean_embedding = template_embeddings.mean(dim=0)         
                final_text_features.append(mean_embedding)

        text_features_tensor = torch.stack(final_text_features) 
        text_features = normalize(text_features_tensor.cpu().numpy(), norm='l2', axis=1)
        return text_features, word_list

    def _aggregate_features(self, features: np.ndarray, method: str, top_k_perc: float = 0.2) -> np.ndarray:
        if features.ndim != 2:
            raise ValueError(f"Expected features as (N,D), got {features.shape}")

        if method == 'mean':
            return np.mean(features, axis=0)

        if method == 'median':
            return np.median(features, axis=0)

        if method == 'top_k_mean':
            if features.shape[0] < 3:
                return np.mean(features, axis=0)

            centroid = np.mean(features, axis=0, keepdims=True)
            features_norm = normalize(features, norm='l2', axis=1)
            centroid_norm = normalize(centroid, norm='l2', axis=1)
            sims = (features_norm @ centroid_norm.T).flatten()

            k = max(1, int(features.shape[0] * top_k_perc))
            top_k_idx = np.argsort(sims)[-k:]
            return np.mean(features[top_k_idx], axis=0)

        raise ValueError(f"Unknown aggregation method: {method}")

    def _pool_masked_features(
        self,
        clip_feat_map: np.ndarray,  
        sam_mask: np.ndarray,      
        aggregation_method: str = "top_k_mean",
        top_k_perc: float = 0.2,
        min_covered_patches: int = 1
    ) -> np.ndarray | None:
        if clip_feat_map.ndim != 3:
            raise ValueError(f"clip_feat_map must be (C,Hf,Wf). Got {clip_feat_map.shape}")
        C, Hf, Wf = clip_feat_map.shape

        if sam_mask.dtype != np.uint8:
            sam_mask_u8 = sam_mask.astype(np.uint8)
        else:
            sam_mask_u8 = sam_mask

        mask_small = cv2.resize(sam_mask_u8, (Wf, Hf), interpolation=cv2.INTER_NEAREST).astype(bool)

        if mask_small.sum() < min_covered_patches:
            return None
        
        select_mask = mask_small

        if self.context_enabled:
            ys, xs = np.where(mask_small)
            if len(xs) > 0:
                x1, x2 = int(xs.min()), int(xs.max())
                y1, y2 = int(ys.min()), int(ys.max())

                x1, y1, x2, y2 = self._expand_bbox(
                    (x1, y1, x2, y2),
                    img_shape_hw=(Hf, Wf),
                    scale=self.context_scale,
                )

                bbox_mask = np.zeros_like(mask_small, dtype=bool)
                bbox_mask[y1:y2+1, x1:x2+1] = True

                select_mask = bbox_mask

        feat_flat = clip_feat_map.reshape(C, -1).T
        masked_feats = feat_flat[select_mask.flatten()]

        if masked_feats.shape[0] == 0:
            return None

        pooled = self._aggregate_features(masked_feats, aggregation_method, top_k_perc=top_k_perc)
        return pooled
    
    def _expand_bbox(self, bbox, img_shape_hw, scale: float):
        x1, y1, x2, y2 = bbox
        H, W = img_shape_hw

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        bw = (x2 - x1 + 1)
        bh = (y2 - y1 + 1)

        new_w = bw * float(scale)
        new_h = bh * float(scale)

        nx1 = int(round(cx - new_w / 2.0))
        nx2 = int(round(cx + new_w / 2.0))
        ny1 = int(round(cy - new_h / 2.0))
        ny2 = int(round(cy + new_h / 2.0))

        nx1 = max(0, min(W - 1, nx1))
        nx2 = max(0, min(W - 1, nx2))
        ny1 = max(0, min(H - 1, ny1))
        ny2 = max(0, min(H - 1, ny2))

        if nx2 < nx1:
            nx1, nx2 = nx2, nx1
        if ny2 < ny1:
            ny1, ny2 = ny2, ny1

        return nx1, ny1, nx2, ny2


    def label_mask(
        self,
        clip_feat_map: np.ndarray,
        sam_mask: np.ndarray,
        aggregation_method: str = "top_k_mean",
        top_k_perc: float = 0.2,
        top_n: int = 5,
        min_covered_patches: int = 1,
        allowed_labels: list[str] | None = None,
    ) -> dict:

        pooled = self._pool_masked_features(
            clip_feat_map=clip_feat_map,
            sam_mask=sam_mask,
            aggregation_method=aggregation_method,
            top_k_perc=top_k_perc,
            min_covered_patches=min_covered_patches,
        )

        if pooled is None:
            return {
                "best_label": "unknown",
                "best_similarity": 0.0,
                "top_matches": []
            }

        pooled_norm = normalize(pooled.reshape(1, -1), norm='l2', axis=1)
        if allowed_labels is not None and len(allowed_labels) > 0:
            idxs = [self.word_to_idx[l] for l in allowed_labels if l in self.word_to_idx]
            if len(idxs) == 0:
                return {
                    "best_label": "unknown",
                    "best_similarity": 0.0,
                    "margin": 0.0,
                    "top_matches": [],
                }

            text_feats = self.text_features[idxs]
            sims = (pooled_norm @ text_feats.T).flatten()
            top_local = np.argsort(sims)[-top_n:][::-1]

            top_matches = []
            for j in top_local:
                global_idx = idxs[j]
                top_matches.append({
                    "label": self.word_list[global_idx],
                    "similarity": float(sims[j]),
                })
        else:
            sims = (pooled_norm @ self.text_features.T).flatten()
            top_indices = np.argsort(sims)[-top_n:][::-1]
            top_matches = [
                {"label": self.word_list[idx], "similarity": float(sims[idx])}
                for idx in top_indices
            ]

        best = top_matches[0] if top_matches else {"label": "unknown", "similarity": 0.0}

        if len(top_matches) >= 2:
            margin = float(top_matches[0]["similarity"] - top_matches[1]["similarity"])
        else:
            margin = float(best["similarity"])

        return {
            "best_label": best["label"],
            "best_similarity": float(best["similarity"]),
            "margin": margin,
            "top_matches": top_matches
        }

    def label_clusters(
        self,
        image_clip_features: np.ndarray,
        cluster_map: np.ndarray,
        aggregation_method: str = "mean",
        top_k_perc: float = 0.2,
        top_n: int = 5,
    ):
        cluster_map = np.array(cluster_map)

        C, H_feat, W_feat = image_clip_features.shape
        H_map, W_map = cluster_map.shape

        if (H_feat, W_feat) != (H_map, W_map):
            pil_map = Image.fromarray(cluster_map.astype(np.uint8))
            resized_map = np.array(pil_map.resize((W_feat, H_feat), Image.NEAREST))
        else:
            resized_map = cluster_map

        clip_vectors = image_clip_features.reshape(C, -1).T
        cluster_labels_info = {}
        unique_clusters = np.unique(resized_map)

        for cluster_id in unique_clusters:
            ys, xs = np.where(resized_map == cluster_id)
            if len(xs) == 0:
                continue

            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())

            if self.context_enabled:
                x1, y1, x2, y2 = self._expand_bbox(
                    (x1, y1, x2, y2),
                    img_shape_hw=(H_feat, W_feat),
                    scale=self.context_scale,
                )

            in_cluster = (resized_map == cluster_id)
            if self.context_enabled:
                in_bbox = np.zeros_like(in_cluster, dtype=bool)
                in_bbox[y1 : y2 + 1, x1 : x2 + 1] = True
                final_mask = (in_cluster & in_bbox).reshape(-1)
            else:
                final_mask = in_cluster.reshape(-1)

            cluster_clip_vectors = clip_vectors[final_mask]
            if cluster_clip_vectors.shape[0] == 0:
                continue

            aggregated_vector = self._aggregate_features(
                cluster_clip_vectors, aggregation_method, top_k_perc=top_k_perc
            )

            projected_vector = normalize(aggregated_vector.reshape(1, -1), norm="l2", axis=1)
            similarities = (projected_vector @ self.text_features.T).flatten()

            top_indices = np.argsort(similarities)[-top_n:][::-1]
            top_matches = [
                {"label": self.word_list[idx], "similarity": float(similarities[idx])}
                for idx in top_indices
            ]
            best_match = top_matches[0]

            logging.info(f"Cluster {cluster_id} → '{best_match['label']}' (sim: {best_match['similarity']:.3f})")

            cluster_labels_info[int(cluster_id)] = {
                "best_label": best_match["label"],
                "best_similarity": best_match["similarity"],
                "top_matches": top_matches,
            }

        return cluster_labels_info
    
    def label_all_images(self, features_list, cluster_maps, image_paths,
                        aggregation_method='mean', top_k_perc=0.2, top_n=5):
        results = {}
        for features, cmap, path in zip(features_list, cluster_maps, image_paths):
            image_name = os.path.basename(path)
            results[image_name] = self.label_clusters(
                image_clip_features=features,
                cluster_map=cmap,
                aggregation_method=aggregation_method,
                top_k_perc=top_k_perc,
                top_n=top_n
            )
        return results
