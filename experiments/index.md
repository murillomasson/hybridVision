# Experiment Index

This document summarizes the experimental runs performed during the development and tuning of the segmentation framework.  
Each experiment corresponds to a specific configuration of the pipeline and records the scope of changes, execution conditions, and the main observations obtained from the run.

Experiments are divided into categories:

**_OX.Y (Optimized experiments)**

Runs executed under the automated hyperparameter optimization framework. These experiments explore the parameter search space defined in `tuner_config.yaml` and evaluate candidate configurations using the objective function implemented in the tuning module.

 **_IX.Y (Individual experiments)**

Standalone runs executed with manually specified configurations. These experiments are used to validate pipeline behavior, inspect qualitative segmentation outputs, and establish interpretable baseline references outside the automated optimization loop.

**C_X.Y (current experiments)**

Represents runs executed with the current framework configuration and considered valid within the experimental record.

**D_X.Y (deprecated experiments)**

Corresponding to runs that are no longer considered valid due to framework bugs, unstable configurations, or exploratory setups that were later replaced.

Each experiment identifier follows the structure **X.Y**, where:

- **X** denotes the dataset scope or experimental phase  
- **Y** denotes the iteration or variation performed within that scope



## Experiment Summary

| Section | Target Image | Iterations | Change Scope | Key Observation |
| :--- | :--- | :--- | :--- | :--- |
| CI2.01 | `000000002149.jpg`, `000000014007.jpg` | 2 | `pipeline_config.yaml` | **Full pipeline run.** <br> • Executed with all main branches enabled. <br> • Active components: DINOv2 semantic branch, ResNet structural branch, fusion branch, and CLIP labeling. <br> • Purpose: provide the complete reference configuration for comparison against partial branch activations. |
| CI2.02 | `000000002149.jpg`, `000000014007.jpg` | 2 | `pipeline_config.yaml` | **CLIP-only run.** <br> • Executed with only CLIP-based labeling active. <br> • DINOv2 semantic branch, ResNet structural branch, and fusion branch disabled. <br> • Purpose: isolate the contribution of CLIP-based semantic labeling without structural or multimodal clustering branches. |
| CI2.03 | `000000002149.jpg`, `000000014007.jpg` | 2 | `pipeline_config.yaml` | **DINOv2 + ResNet + CLIP run.** <br> • Executed with DINOv2 semantic branch, ResNet structural branch, and CLIP labeling enabled. <br> • Fusion branch disabled. <br> • Purpose: evaluate the joint behavior of semantic and structural branches without hybrid feature fusion. |
| CI2.04 | `000000002149.jpg`, `000000014007.jpg` | 2 | `pipeline_config.yaml` | **Hybrid + CLIP run.** <br> • Executed with fusion branch and CLIP labeling enabled. <br> • Standalone DINOv2 semantic branch and ResNet structural branch disabled. <br> • Purpose: isolate the behavior of the fused feature representation with semantic labeling, without separate branch outputs. |
| CI2.05 | `000000002149.jpg`, `000000014007.jpg` | 2 | `pipeline_config.yaml` | **DINOv2 + CLIP run.** <br> • Executed with DINOv2 semantic branch and CLIP labeling enabled. <br> • ResNet structural branch and fusion branch disabled. <br> • Purpose: isolate the semantic branch combined with CLIP-based labeling. |
| CI2.06 | `000000002149.jpg`, `000000014007.jpg` | 2 | `pipeline_config.yaml` | **ResNet + CLIP run.** <br> • Executed with ResNet structural branch and CLIP labeling enabled. <br> • DINOv2 semantic branch and fusion branch disabled. <br> • Purpose: isolate the structural branch combined with CLIP-based labeling. |
| _DO1.01_ | `000000017029.jpg` | 230 | - | **Initial exploratory run.** <br> • First attempt at running the optimization framework. <br> • Used preliminary search space and tuning configuration. |
| _DO2.01_ | `000000024919.jpg` | 900 | - Dataset <br> - `tuner_search_space.json` | **Search space restructuring.** <br> • Removed Spectral Clustering from the search space due to computational cost ($O(n³)$). <br> • Target image changed to a visually simpler case for tuning iterations. |
| _D02.02_ | `000000024919.jpg` | 200 | - `tuner_search_space.json` | **Search space redesign.** <br> • Previous configuration (`tuner_search_space_v1.json`) contained ~10³⁰ combinations. <br> • Tuning space redesigned to reduce dimensionality and improve sampling coverage. <br> • Main changes: <br> – Restrict tuning to hybrid mode (DINOv2 + ResNet). <br> – Remove computationally expensive or redundant options (Spectral Clustering, multiple dimensionality reduction methods). <br> – Discretize continuous parameters using empirical ranges. <br> • Estimated effective search space reduced to ~10⁶–10⁷ configurations. |
| _DO2.03_ | `000000024919.jpg` | 200 | - `tuner_search_space.json` | **Parameter prioritization.** <br> • Search space adjusted to focus on parameters with stronger observed influence. <br> • Positive influence observed for: `k_semantic_used`, `k_structural_used`. <br> • Negative influence observed for: `stability_score_thresh`. <br> • Several discrete parameters converted to continuous ranges (`0.0–1.0`) for optimization using Optuna. <br> • Validator weights (`dino_iou`, `clip_confidence`, `resnet_iou`) normalized to sum to 1.0 to enforce balanced weighting. |
| _DO2.04_ | `000000024919.jpg` | 200 | - `tuner_search_space.json` <br> - `validators/final_validator.py` | **Validation logic modification.** <br> • Final score calculation changed to `max(semantic_score, structural_score)`. <br> • Validation logic updated to use OR condition (`is_semantically_valid or is_structurally_valid`). <br> • Introduced validation states: `VALIDATED_HYBRID`, `SEMANTIC_ONLY`, `STRUCTURAL_ONLY`. <br> • Masks failing both IoU thresholds or minimum score constraints are filtered out. |
| CO2.05 | `000000024919.jpg` | 300 | - `src/extractors/clip_extractor.py` <br> - `src/labelers/clip_labeler.py` <br> - `src/pipeline.py` | **Feature extraction bug fix.** <br> • CLIPExtractor previously returned 768-dimensional DINOv2 features instead of CLIP embeddings. <br> • CLIPExtractor implementation rewritten. <br> • Associated pipeline and labeler logic updated to align with CLIP embedding dimensions. |
| CO2.06 | `000000024919.jpg` | 40 | - `experiments/tuner.py` <br> - `src/analyzers/tuning_analyzer.py` <br> - `analyzer/experiment_analyzer.py` <br> - `tuner_config.yaml` | **Objective function redesign.** <br> • Introduced continuous objective function composed of: <br> – Semantic Score (CLIP similarity) <br> – Mean IoU (DINO + ResNet) <br> – Consistency reward (`1 - |iou_dino - iou_resnet|`). <br> • Implemented `TuningAnalyzer` to compute additional metrics (IoU variance, label entropy, semantic-structural alignment). <br> • Refactored tuning module to separate optimization and analytical diagnostics. |
| CO2.07 | `000000024919.jpg` | 200 | - `src/extractors/clip_extractor.py` <br> - `src/labelers/clip_labeler.py` <br> - `src/pipeline.py` <br> - `src/utils/visualizer.py` | **Objective function refinement and search space adjustment.** <br> • Introduced objective formulation: <br> `Final Score = Semantic Base + Structural & Consistency Bonus - Segment Count Penalty`. <br> • Components: <br> – Semantic base: `semantic_score` <br> – Structural bonus: `mean_iou` <br> – Consistency reward: alignment between DINO and ResNet IoU. <br> • Segment count penalty introduced to control segmentation granularity. <br> • Search space updated: <br> – Validator thresholds expanded (`0.20–0.90`). <br> – SAM thresholds (`pred_iou_thresh`, `stability_score_thresh`) relaxed. <br> – Cluster multiplier ranges expanded. |