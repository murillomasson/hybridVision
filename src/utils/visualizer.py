import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, silhouette_samples
from PIL import Image
from pathlib import Path
import logging
import os
import cv2

logging.basicConfig(level=logging.INFO)


class Visualizer:
    def __init__(self, output_dir="./output/plots", cmap_name="tab10"):
        self.cmap_name = cmap_name
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def get_cluster_colors(self, n_clusters):
        cmap = plt.colormaps.get_cmap(self.cmap_name) 
        return [cmap(i) for i in np.linspace(0, 1, n_clusters)] 

    def plot_heatmap(self, labels_map, overlay_image=None, layer_name="", method="", idx=0, show=False):
        n_clusters = len(np.unique(labels_map))
        if n_clusters <= 1:
            logging.warning("Skipping heatmap plot as only one cluster was found.")
            return None
            
        cluster_colors = self.get_cluster_colors(n_clusters)
        label_rgb = np.zeros(labels_map.shape + (3,), dtype=np.uint8)
        
        unique_labels = sorted(np.unique(labels_map))
        for i, label in enumerate(unique_labels):
            color = cluster_colors[i]
            label_rgb[labels_map == label] = (np.array(color[:3]) * 255).astype(np.uint8)

        if overlay_image is not None:
            img_array = np.array(overlay_image)
            if img_array.max() <= 1.0:
                img_array = (img_array * 255).astype(np.uint8)

            label_resized = np.array(
                Image.fromarray(label_rgb).resize(
                    (img_array.shape[1], img_array.shape[0]), Image.NEAREST
                )
            )

            overlay_alpha = 0.5
            if img_array.ndim == 2:
                img_array = np.stack([img_array] * 3, axis=-1)

            overlay_img = cv2.addWeighted(img_array, 1 - overlay_alpha, label_resized, overlay_alpha, 0)

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(label_rgb)
            axes[0].set_title("Cluster Heatmap")
            axes[1].imshow(overlay_img)
            axes[1].set_title("Heatmap + Image")
            for ax in axes:
                ax.axis("off")
            fig.suptitle(f"{layer_name} [{method}] image {idx}")
        else:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(label_rgb)
            ax.set_title(f"Cluster only - {layer_name} [{method}] image {idx}")
            ax.axis("off")

        subfolder_path = Path(self.output_dir) / method
        subfolder_path.mkdir(parents=True, exist_ok=True)
        
        save_path = subfolder_path / f"heatmap_{layer_name}_img{idx}.png"
        plt.savefig(save_path, bbox_inches='tight')
        logging.info(f"Saved plot to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close(fig)
        
        return label_rgb

    def plot_silhouette(self, X, labels, layer_name="", method="", show=False):
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)

        if n_clusters < 2:
            logging.warning(f"Cannot compute silhouette score for {n_clusters} cluster(s). Skipping plot.")
            return

        cluster_colors = self.get_cluster_colors(n_clusters)
        try:
            metric = 'cosine' if np.allclose(np.linalg.norm(X, axis=1), 1.0, atol=1e-3) else 'euclidean'
            silhouette_avg = silhouette_score(X, labels, metric=metric)
            sample_silhouette_values = silhouette_samples(X, labels, metric=metric)
        except Exception as e:
            logging.warning(f"WARNING: Cannot compute silhouette score: {e}")
            return

        fig, ax = plt.subplots(figsize=(7, 5))
        y_lower = 10
        for i, label in enumerate(unique_labels):
            ith_vals = sample_silhouette_values[labels == label]
            ith_vals.sort()
            size_cluster_i = ith_vals.shape[0]
            y_upper = y_lower + size_cluster_i
            color = cluster_colors[i]
            ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_vals, facecolor=color[:3], edgecolor=color[:3], alpha=0.7)
            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(label))
            y_lower = y_upper + 10

        ax.axvline(x=silhouette_avg, color="red", linestyle="--")
        ax.set_title(f"Silhouette Plot - Layer: {layer_name}, Method: {method} (avg={silhouette_avg:.2f})")
        ax.set_xlabel(f"Silhouette Coefficient (Metric: {metric})")
        ax.set_ylabel("Cluster")
        
        subfolder_path = Path(self.output_dir) / method
        subfolder_path.mkdir(parents=True, exist_ok=True)
        save_path = subfolder_path / f"silhouette_{layer_name}.png"
        plt.savefig(save_path, bbox_inches='tight')
        logging.info(f"Saved plot to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close(fig)

    def plot_final_segmentation(self, image_rgb, validated_masks, image_name):
        if image_rgb is None or not validated_masks:
            logging.warning(f"Skipping final plot for {image_name} due to missing data.")
            return

        overlay_image = image_rgb.copy()
        legend_canvas = np.ones((max(200, len(validated_masks) * 30 + 20), 300, 3), dtype=np.uint8) * 255
        
        unique_labels = sorted(list(set([m['label'] for m in validated_masks])))
        colors = self.get_cluster_colors(len(unique_labels))
        label_to_color = {label: (np.array(colors[i][:3]) * 255) for i, label in enumerate(unique_labels)}

        for mask_data in validated_masks:
            mask = mask_data.get('mask')
            if mask is None: continue
            
            label = mask_data.get('label', 'unknown')
            color = label_to_color.get(label, (255, 255, 255))
            
            colored_mask_image = np.zeros_like(overlay_image)
            colored_mask_image[mask] = color.astype(np.uint8)
            overlay_image = cv2.addWeighted(overlay_image, 1.0, colored_mask_image, 0.5, 0)
            
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(c)
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                text = f"{label} ({mask_data.get('final_confidence', 0):.2f})"
                text_size, _ = cv2.getTextSize(text, font, 0.5, 1)
                
                rect_start = (x, y - text_size[1] - 10)
                rect_end = (x + text_size[0], y - 10)
                text_pos = (x, y - 10)
                
                cv2.rectangle(overlay_image, rect_start, rect_end, (255,255,255), -1)
                cv2.putText(overlay_image, text, text_pos, font, 0.5, (0,0,0), 1, cv2.LINE_AA)

        for i, label in enumerate(unique_labels):
            color = label_to_color[label]
            display_color = tuple(int(c) for c in color[::-1])
            cv2.putText(legend_canvas, label, (10, 30 * (i+1)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, display_color, 2)

        fig, axes = plt.subplots(1, 3, figsize=(20, 7), gridspec_kw={'width_ratios': [3, 3, 1]})
        axes[0].imshow(image_rgb)
        axes[0].set_title("Original Image")
        axes[0].axis("off")
        
        axes[1].imshow(overlay_image)
        axes[1].set_title("Validated Segmentation")
        axes[1].axis("off")

        axes[2].imshow(legend_canvas)
        axes[2].set_title("Legend")
        axes[2].axis("off")

        fig.suptitle(f"Final Results for {os.path.basename(image_name)}", fontsize=16)
        
        safe_image_name = "".join([c for c in os.path.basename(image_name) if c.isalpha() or c.isdigit() or c in ['.', '_']]).rstrip()
        save_path = os.path.join(self.output_dir, f"final_result_{safe_image_name}.png")
        
        plt.savefig(save_path, bbox_inches='tight')
        logging.info(f"Saved final segmentation plot to {save_path}")
        plt.close(fig)

    def plot_cluster_scatter(self, X, labels, layer_name="", method="", show=False):

        if X.shape[1] < 2:
            logging.warning("Cannot create scatter plot for 1D features. Skipping.")
            return

        from sklearn.decomposition import PCA

        logging.info("Generating 2D scatter plot of clusters...")
        
        pca = PCA(n_components=2, random_state=42)
        X_reduced = pca.fit_transform(X)

        fig, ax = plt.subplots(figsize=(8, 6))
        
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        colors = self.get_cluster_colors(n_clusters)
        
        for i, label in enumerate(unique_labels):
            ax.scatter(
                X_reduced[labels == label, 0],
                X_reduced[labels == label, 1],
                s=50,
                color=colors[i],
                label=f'Cluster {label}',
                alpha=0.7
            )

        ax.set_title(f'2D Cluster Visualization (PCA) - {layer_name} [{method}]')
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.legend()
        ax.grid(True)
        
        subfolder_path = Path(self.output_dir) / method
        subfolder_path.mkdir(parents=True, exist_ok=True)
        save_path = subfolder_path / f"scatter_2d_{layer_name}.png"
        
        plt.savefig(save_path, bbox_inches='tight')
        logging.info(f"Saved scatter plot to {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

    def plot_validated_masks_debug(self, image_rgb, validated_masks, image_name, show=False):
        if image_rgb is None:
            logging.warning(f"Skipping validated mask debug plot for {image_name}: image is None.")
            return

        if not validated_masks:
            logging.warning(f"No validated masks to plot for {image_name}.")
            return

        safe_image_name = "".join([c for c in os.path.basename(image_name) if c.isalpha() or c.isdigit() or c in ['.', '_']]).rstrip()

        n = len(validated_masks)
        fig, axes = plt.subplots(n, 3, figsize=(18, 5 * n))

        if n == 1:
            axes = np.expand_dims(axes, axis=0)

        for i, m in enumerate(validated_masks):
            mask = m.get("mask")
            label = m.get("label", "unknown")
            status = m.get("status", "UNVALIDATED")
            conf = m.get("final_confidence", 0.0)
            sb = m.get("scores_breakdown", {})

            axes[i, 0].imshow(mask.astype(np.uint8), cmap="gray")
            axes[i, 0].set_title(f"Mask {i} (binary)")
            axes[i, 0].axis("off")

            overlay = image_rgb.copy()
            colored = np.zeros_like(overlay)
            colored[mask] = (255, 0, 0) 
            overlay = cv2.addWeighted(overlay, 1.0, colored, 0.5, 0)

            axes[i, 1].imshow(overlay)
            axes[i, 1].set_title(f"Overlay {i}")
            axes[i, 1].axis("off")

            text = (
                f"label: {label}\n"
                f"status: {status}\n"
                f"final_confidence: {conf:.3f}\n\n"
                f"dino_iou: {sb.get('dino_iou', 0.0):.3f}\n"
                f"resnet_iou: {sb.get('resnet_iou', 0.0):.3f}\n"
                f"clip_confidence: {sb.get('clip_confidence', 0.0):.3f}\n"
            )
            axes[i, 2].text(0.01, 0.99, text, va="top", ha="left", fontsize=12)
            axes[i, 2].set_title("Scores")
            axes[i, 2].axis("off")

        fig.suptitle(f"Validated masks debug: {safe_image_name}", fontsize=16)
        save_path = os.path.join(self.output_dir, f"debug_validated_{safe_image_name}.png")
        plt.savefig(save_path, bbox_inches="tight")
        logging.info(f"Saved validated debug plot to {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)
