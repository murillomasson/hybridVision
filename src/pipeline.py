import os
import logging
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import gc

from src.utils.io import load_pickle, save_pickle
from src.utils.image_loader import ImageLoader
from src.utils.visualizer import Visualizer
from src.extractors.dino_v2_extractor import DINOv2Extractor
from src.extractors.clip_extractor import CLIPExtractor
from src.extractors.resnet_extractor import ResNetExtractor
from src.normalizers.feature_normalizer import get_normalizer
from src.reducers.dimensionality_reducer import get_reducer
from src.labelers.clip_labeler import ClipLabeler
from src.preprocessors.preprocessor import Preprocessor
from src.validators.final_validator import FinalValidator
from src.estimators.cluster_count_estimator import ClusterCountEstimator
from src.clusterers.utils_clusterer import create_clusterer
from src.optimizers.k_optimizer import KValueOptimizer
from src.postprocessors.identify_resolver import IdentityResolver


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Pipeline:
    def __init__(self, config: dict):
        self.config = config
        self.output_dir = config.get('output_dir', './output')
        os.makedirs(self.output_dir, exist_ok=True)

        self.device = config.get('device', 'cuda')
        self.generate_plots = config.get('generate_plots', True)

        logging.info("Initializing architectural components...")

        self.preprocessor = Preprocessor(
            config.get('resnet_pipeline', {}).get('image_preprocessing', {})
        )

        self.k_estimator = ClusterCountEstimator(
            configs=config['heuristic_settings'],
            device=self.device,
            output_dir=self.output_dir
        )
        self.final_validator = FinalValidator(config.get('validator_settings', {}))
        self.image_loader = ImageLoader()
        self.visualizer = Visualizer(output_dir=os.path.join(self.output_dir, "plots"))

        self._run_dino = bool(self.config.get('run_dino_pipeline') or self.config.get('run_fusion_pipeline'))
        self._run_resnet = bool(self.config.get('run_resnet_pipeline') or self.config.get('run_fusion_pipeline'))

        dino_clip_labeling = self.config.get('dino_pipeline', {}).get('clustering', {}).get('run_clip_labeling', False)
        resnet_clip_labeling = self.config.get('resnet_pipeline', {}).get('clustering', {}).get('run_clip_labeling', False)
        fusion_clip_labeling = self.config.get('fusion_pipeline', {}).get('clustering', {}).get('run_clip_labeling', False)
        self._run_clip_labeling = bool(dino_clip_labeling or resnet_clip_labeling or fusion_clip_labeling)

        self.resnet_extractor = None
        self.dino_extractor = None
        self.clip_extractor = None
        self.clip_labeler = None
        
        post_cfg = self.config.get("postprocess", {}).get("identity_resolver", {})
        self.identity_resolver = IdentityResolver(post_cfg)


    def _init_feature_models(self):
        if self._run_resnet and self.resnet_extractor is None:
            self.resnet_extractor = ResNetExtractor(
                **self.config['extractors']['resnet'],
                output_dir=self.output_dir,
                device=self.device
            )

        if self._run_dino and self.dino_extractor is None:
            self.dino_extractor = DINOv2Extractor(
                **self.config['extractors']['dino_v2'],
                output_dir=self.output_dir,
                device=self.device
            )

        if self._run_clip_labeling and self.clip_extractor is None:
            self.clip_extractor = CLIPExtractor(
                **self.config['extractors']['clip'],
                output_dir=self.output_dir,
                device=self.device
            )

        if self._run_clip_labeling and self.clip_labeler is None:
            labeler_config = self.config.get('labeler', {})
            ctx = labeler_config.get("context_crop", {}) or {}

            self.clip_labeler = ClipLabeler(
                device=self.device,
                words_json_path=labeler_config['words_json_path'],
                prompt_template=labeler_config.get('prompt_template', "a photo of a {}"),
                context_enabled=bool(ctx.get("enabled", False)),
                context_scale=float(ctx.get("scale", 1.3)),
            )

    def _preprocess_features(self, features, config):
        if config.get('run_normalization', False):
            normalizer_config = config.get('normalizer', {})
            if normalizer_config:
                normalizer = get_normalizer(**normalizer_config)
                features = normalizer.fit_transform(features)

        if config.get('run_reduction', False):
            reducer_config = config.get('reducer', {})
            if reducer_config:
                reducer = get_reducer(**reducer_config)
                features = reducer.fit_transform(features)

        return features

    def _create_fused_features(self, dino_features, resnet_features):
        logging.info("--- Creating Fused Features (DINO + ResNet) ---")

        dino_tensor = torch.from_numpy(dino_features).float()
        resnet_tensor = torch.from_numpy(resnet_features).float()

        B, C_dino, H_dino, W_dino = dino_tensor.shape

        logging.info(
            f"Resizing ResNet features from {resnet_tensor.shape} to match DINO's spatial dimensions {(H_dino, W_dino)}."
        )
        resnet_resized = F.interpolate(
            resnet_tensor, size=(H_dino, W_dino), mode='bilinear', align_corners=False
        )

        dino_flat = dino_tensor.view(B, C_dino, -1).permute(0, 2, 1)
        resnet_flat = resnet_resized.view(B, resnet_resized.shape[1], -1).permute(0, 2, 1)

        dino_norm = F.normalize(dino_flat, p=2, dim=2)
        resnet_norm = F.normalize(resnet_flat, p=2, dim=2)

        fused_flat = torch.cat([dino_norm, resnet_norm], dim=2)
        logging.info(f"Concatenated features shape: {fused_flat.shape}")

        fused_features = fused_flat.permute(0, 2, 1).view(B, -1, H_dino, W_dino)

        return fused_features.numpy()

    def _run_clustering_per_image(self, features, config, k_values, image_paths, pipeline_name):
        num_images, C, H, W = features.shape
        all_labels_maps = []

        clusterer_config = config.get('clusterer', {})
        clusterer_method_name = clusterer_config.get('method', 'unknown_clusterer')

        for i in range(num_images):
            k = k_values[i]
            image_path = image_paths[i]
            image_name = os.path.basename(image_path)

            logging.info(
                f"Clustering image {i+1}/{num_images} ({image_name}) with k={k} using method '{clusterer_method_name}'..."
            )

            patch_vectors = features[i].reshape(C, H * W).T
            processed_vectors = self._preprocess_features(
                patch_vectors, config.get('feature_processing', {})
            )

            clusterer = create_clusterer(clusterer_config, n_clusters=k)
            labels = clusterer.fit_predict(processed_vectors)
            labels_map = labels.reshape(H, W)

            if self.generate_plots:
                plot_layer_name = f"{pipeline_name}_img{i}"

                logging.info(f"Visualizing silhouette score for {image_name}...")
                self.visualizer.plot_silhouette(
                    X=processed_vectors,
                    labels=labels,
                    layer_name=plot_layer_name,
                    method=clusterer_method_name
                )

                logging.info(f"Visualizing 2D cluster scatter plot for {image_name}...")
                self.visualizer.plot_cluster_scatter(
                    X=processed_vectors,
                    labels=labels,
                    layer_name=plot_layer_name,
                    method=clusterer_method_name
                )

                logging.info(f"Visualizing cluster heatmap for {image_name}...")
                original_image = self.image_loader.load_single_image(image_path, return_rgb=True)

                self.visualizer.plot_heatmap(
                    labels_map=labels_map,
                    overlay_image=original_image,
                    layer_name=plot_layer_name,
                    method=clusterer_method_name,
                    idx=i
                )

            all_labels_maps.append(labels_map)

        return np.array(all_labels_maps)

    def _run_analysis_pipeline(self, image_paths, features, config, k_values, pipeline_name):
        logging.info(f"--- Running Analysis Pipeline: {pipeline_name} ---")

        cluster_maps = self._run_clustering_per_image(
            features, config, k_values, image_paths, pipeline_name
        )

        labels_per_image = {}
        clip_features_per_image = None

        if self.clip_labeler and config.get('run_clip_labeling', False):
            logging.info(f"Running CLIP labeling for {pipeline_name} clusters...")

            clip_features_per_image, _ = self.clip_extractor.extract(
                image_paths, cache_key=f"clip_features_for_{pipeline_name}"
            )

            labeler_cfg = self.config.get('labeler', {})
            agg_method = labeler_cfg.get('aggregation_method', 'mean')
            top_k_perc = labeler_cfg.get('aggregation_top_k_perc', 0.2)
            top_n = labeler_cfg.get('top_n_labels', 5)

            labels_per_image = self.clip_labeler.label_all_images(
                features_list=clip_features_per_image,
                cluster_maps=cluster_maps,
                image_paths=image_paths,
                aggregation_method=agg_method,
                top_k_perc=top_k_perc,
                top_n=top_n
            )

        return {
            'clusters': cluster_maps,
            'labels': labels_per_image,
            'clip_features': clip_features_per_image
        }

    def run(self):
        logging.info("--- Starting Main Pipeline Run ---")

        image_paths = self.image_loader.get_image_paths(self.config['image_dir'])
        if not image_paths:
            logging.error("No images found in the specified directory.")
            return {}, {}

        # ==========================================================
        # Phase 1: SAM2 masks + heuristic k (pra não estourar VRAM)
        # ==========================================================
        logging.info("--- Phase 2: K-Value Estimation and SAM Mask Generation (SAM2-first) ---")

        k_values_semantic, k_values_structural, all_sam_masks = [], [], []

        for i, img_path in enumerate(image_paths):
            image_name = os.path.basename(img_path)
            logging.info(f"Processing k-values and masks for image {image_name}...")

            image_bgr = self.image_loader.load_single_image(img_path, return_rgb=False)
            if image_bgr is None:
                logging.warning(f"Skipping k-estimation for {image_name} as it could not be loaded.")
                if self._run_dino:
                    k_values_semantic.append(8)
                if self._run_resnet:
                    k_values_structural.append(8)
                all_sam_masks.append([])
                continue

            heuristic_results = self.k_estimator.estimate_k_with_heuristics(image_bgr)

            all_sam_masks.append(heuristic_results['processed_masks'])

            if self._run_dino:
                k_values_semantic.append(heuristic_results.get('k_semantic', 8))
            if self._run_resnet:
                k_values_structural.append(heuristic_results.get('k_structural', 8))

        logging.info(f"Initial semantic k-values (heuristic): {k_values_semantic}")
        logging.info(f"Initial structural k-values (heuristic): {k_values_structural}")

        try:
            del self.k_estimator
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ==========================================================
        # Phase 2: feature models + feature extraction
        # ==========================================================
        self._init_feature_models()

        logging.info("--- Phase 1: Global Feature Extraction ---")
        dino_features, resnet_features, fused_features = None, None, None

        if self.dino_extractor:
            dino_features, _ = self.dino_extractor.extract(image_paths, "dino_features")

        if self.resnet_extractor:
            processed_images_resnet = [
                self.preprocessor.process(self.image_loader.load_single_image(p))
                for p in image_paths
            ]
            resnet_features, _ = self.resnet_extractor.extract(
                processed_images_resnet, "resnet_features_preprocessed"
            )

        if self.config.get('run_fusion_pipeline'):
            if dino_features is not None and resnet_features is not None:
                fused_features = self._create_fused_features(dino_features, resnet_features)
            else:
                logging.warning("Skipping fusion pipeline because DINO or ResNet features are missing.")

        if self.dino_extractor:
            dino_config = self.config['dino_pipeline']['clustering']
            if dino_config.get('k_selection', {}).get('method') == 'silhouette':
                optimizer = KValueOptimizer(dino_config['clusterer'])
                k_range = tuple(dino_config['k_selection']['k_range'])
                for i, img_path in enumerate(image_paths):
                    image_name = os.path.basename(img_path)
                    flat_features = dino_features[i].reshape(dino_features.shape[1], -1).T
                    plot_save_path = os.path.join(self.visualizer.output_dir, f"dino_k_opt_{image_name}.png")
                    k_values_semantic[i] = optimizer.find_best_k(flat_features, k_range, plot_save_path)

        if self.resnet_extractor:
            resnet_config = self.config['resnet_pipeline']['clustering']
            if resnet_config.get('k_selection', {}).get('method') == 'silhouette':
                optimizer = KValueOptimizer(resnet_config['clusterer'])
                k_range = tuple(resnet_config['k_selection']['k_range'])
                for i, img_path in enumerate(image_paths):
                    image_name = os.path.basename(img_path)
                    flat_features = resnet_features[i].reshape(resnet_features.shape[1], -1).T
                    plot_save_path = os.path.join(self.visualizer.output_dir, f"resnet_k_opt_{image_name}.png")
                    k_values_structural[i] = optimizer.find_best_k(flat_features, k_range, plot_save_path)

        logging.info(f"Determined semantic k-values: {k_values_semantic}")
        logging.info(f"Determined structural k-values: {k_values_structural}")

        # ==========================================================
        # Phase 3: Multi-View Analysis
        # ==========================================================
        logging.info("--- Phase 3: Multi-View Analysis ---")
        dino_results, resnet_results, fusion_results = None, None, None

        if self.config.get('run_dino_pipeline') and dino_features is not None:
            dino_results = self._run_analysis_pipeline(
                image_paths, dino_features, self.config['dino_pipeline']['clustering'],
                k_values_semantic, 'dino_semantic'
            )

        if self.config.get('run_resnet_pipeline') and resnet_features is not None:
            resnet_results = self._run_analysis_pipeline(
                image_paths, resnet_features, self.config['resnet_pipeline']['clustering'],
                k_values_structural, 'resnet_structural'
            )

        if self.config.get('run_fusion_pipeline') and fused_features is not None:
            fusion_config = self.config['fusion_pipeline']['clustering']
            k_values_fusion = []

            if fusion_config.get('k_selection', {}).get('method') == 'silhouette':
                optimizer = KValueOptimizer(fusion_config['clusterer'])
                k_range = tuple(fusion_config['k_selection']['k_range'])
                for i, img_path in enumerate(image_paths):
                    image_name = os.path.basename(img_path)
                    flat_features = fused_features[i].reshape(fused_features.shape[1], -1).T
                    plot_save_path = os.path.join(self.visualizer.output_dir, f"fusion_k_opt_{image_name}.png")
                    k_values_fusion.append(optimizer.find_best_k(flat_features, k_range, plot_save_path))
            else:
                k_source = fusion_config.get('k_selection', {}).get('k_source', 'semantic')
                logging.info(f"Using '{k_source}' k-values for fusion pipeline.")
                k_values_fusion = k_values_structural if k_source == 'structural' else k_values_semantic

            logging.info(f"Determined fusion k-values: {k_values_fusion}")
            fusion_results = self._run_analysis_pipeline(
                image_paths, fused_features, fusion_config,
                k_values_fusion, 'fusion_hybrid'
            )

        # ==========================================================
        # Phase 4 & 5: Validation and Output
        # ==========================================================
        logging.info("--- Phase 4 & 5: Running Final Validation and Output ---")

        final_output = {}
        max_potential_dino_iou = 0.0
        max_potential_resnet_iou = 0.0

        labeler_cfg = self.config.get('labeler', {})

        clip_features_for_validation = None
        if self.clip_extractor is not None:
            clip_features_for_validation, _ = self.clip_extractor.extract(
                image_paths,
                cache_key="clip_features_for_validation"
            )

        for i, img_path in enumerate(image_paths):
            image_name = os.path.basename(img_path)
            logging.info(f"Validating masks for {image_name}...")

            image_rgb = self.image_loader.load_single_image(img_path, return_rgb=True)
            if image_rgb is None:
                continue

            sam_masks_for_this_image = all_sam_masks[i]
            validated_masks = []

            for sam_mask_data in sam_masks_for_this_image:
                dino_res_img = None
                resnet_res_img = None

                if dino_results and dino_results.get('clusters') is not None:
                    dino_res_img = {
                        'clusters': dino_results['clusters'][i],
                        'labels': dino_results['labels'].get(image_name, {})
                    }

                if resnet_results and resnet_results.get('clusters') is not None:
                    resnet_res_img = {
                        'clusters': resnet_results['clusters'][i]
                    }
                
                labeler_cfg = self.config.get("labeler", {})

                if clip_features_for_validation is not None and self.clip_labeler is not None:
                    clip_feat_img = clip_features_for_validation[i]  # (512,7,7)
                    clip_info = self.clip_labeler.label_mask(
                        clip_feat_map=clip_feat_img,
                        sam_mask=sam_mask_data["segmentation"],
                        aggregation_method=labeler_cfg.get("aggregation_method", "top_k_mean"),
                        top_k_perc=labeler_cfg.get("aggregation_top_k_perc", 0.2),
                        top_n=labeler_cfg.get("top_n_labels", 5),
                        min_covered_patches=1,
                    )
                    sam_mask_data["clip_info"] = clip_info

                verdict, raw_scores = self.final_validator.validate_and_label(
                    sam_mask_data,
                    dino_res_img,
                    resnet_res_img,
                    clip_feat_map=None,
                    clip_labeler=self.clip_labeler
                )

                if raw_scores.get('dino_iou', 0.0) > max_potential_dino_iou:
                    max_potential_dino_iou = raw_scores['dino_iou']

                if raw_scores.get('resnet_iou', 0.0) > max_potential_resnet_iou:
                    max_potential_resnet_iou = raw_scores['resnet_iou']

                if verdict:
                    validated_masks.append(verdict)

            if validated_masks:
                validated_masks.sort(
                    key=lambda m: (
                        m.get("scores_breakdown", {}).get("resnet_iou", 0.0),
                        m.get("final_confidence", 0.0)
                    ),
                    reverse=True
                )

            if getattr(self, "identity_resolver", None) is not None and self.identity_resolver.enabled:
                try:
                    validated_masks = self.identity_resolver.run(image_rgb, validated_masks)
                except Exception:
                    logging.exception("IdentityResolver failed; keeping original validated masks.")

            final_output[image_name] = validated_masks

            if self.generate_plots:
                try:
                    self.visualizer.plot_final_segmentation(
                        image_rgb=image_rgb,
                        validated_masks=validated_masks,
                        image_name=image_name
                    )

                    self.visualizer.plot_validated_masks_debug(
                        image_rgb=image_rgb,
                        validated_masks=validated_masks,
                        image_name=image_name
                    )

                except Exception:
                    logging.exception("Failed to generate final validation plots.")

            validated_masks.sort(
                key=lambda m: (
                    m.get("scores_breakdown", {}).get("resnet_iou", 0.0),
                    m.get("final_confidence", 0.0)
                ),
                reverse=True
            )

        diagnostic_data = {
            'num_sam_masks': len(all_sam_masks[0]) if all_sam_masks and all_sam_masks[0] else 0,
            'k_semantic_used': k_values_semantic[0] if k_values_semantic else 0,
            'k_structural_used': k_values_structural[0] if k_values_structural else 0,
            'max_potential_dino_iou': max_potential_dino_iou,
            'max_potential_resnet_iou': max_potential_resnet_iou
        }

        logging.info("--- Main Pipeline Finished ---")
        return final_output, diagnostic_data
