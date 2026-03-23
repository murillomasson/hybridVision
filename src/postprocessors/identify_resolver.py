import logging
import numpy as np

logger = logging.getLogger(__name__)


class IdentityResolver:
    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}

        self.enabled = bool(cfg.get("enabled", False))

        self.nms_iou_thr_same_label = float(cfg.get("nms_iou_thr_same_label", 0.65))
        self.nms_iou_thr_global = float(cfg.get("nms_iou_thr_global", 0.85))
        self.use_global_nms = bool(cfg.get("use_global_nms", True))

        self.border_touch_frac_thr = float(cfg.get("border_touch_frac_thr", 0.40))
        self.border_penalty = float(cfg.get("border_penalty", 0.15))

        self.max_area_frac = float(cfg.get("max_area_frac", 0.80))
        self.area_penalty = float(cfg.get("area_penalty", 0.10))

        self.strip_aspect_thr = float(cfg.get("strip_aspect_thr", 7.0))
        self.strip_area_frac_max = float(cfg.get("strip_area_frac_max", 0.25))
        self.strip_penalty = float(cfg.get("strip_penalty", 0.12))

        self.top_k = int(cfg.get("top_k", 10))

    def _get_mask(self, verdict: dict):
        for k in ("mask", "segmentation", "binary_mask", "mask_binary", "sam_mask"):
            m = verdict.get(k, None)
            if m is not None:
                return m
        sb = verdict.get("sam_mask_data", None)
        if isinstance(sb, dict):
            for k in ("mask", "segmentation", "binary_mask"):
                if k in sb:
                    return sb[k]
        return None

    def _get_label(self, verdict: dict):
        for k in ("label", "best_label", "pred_label"):
            v = verdict.get(k, None)
            if isinstance(v, str) and v.strip():
                return v.strip()
        sb = verdict.get("scores_breakdown", {})
        if isinstance(sb, dict):
            v = sb.get("label", None)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return "unknown"

    def _get_base_score(self, verdict: dict):
        s = verdict.get("final_confidence", None)
        if s is None:
            s = verdict.get("final_score", None)
        if s is None:
            sb = verdict.get("scores_breakdown", {}) or {}
            d = float(sb.get("dino_iou", 0.0) or 0.0)
            r = float(sb.get("resnet_iou", 0.0) or 0.0)
            c = float(sb.get("clip_confidence", 0.0) or 0.0)
            s = 0.34 * d + 0.33 * r + 0.33 * c
        try:
            return float(s)
        except Exception:
            return 0.0

    def _bbox_from_mask(self, mask: np.ndarray):
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return None
        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()
        return int(x1), int(y1), int(x2), int(y2)

    def _iou(self, a: np.ndarray, b: np.ndarray):
        inter = np.logical_and(a, b).sum()
        if inter == 0:
            return 0.0
        union = np.logical_or(a, b).sum()
        return float(inter) / float(union + 1e-9)

    def _border_touch_fraction(self, mask: np.ndarray):
        h, w = mask.shape[:2]
        top = mask[0, :].mean()
        bottom = mask[h - 1, :].mean()
        left = mask[:, 0].mean()
        right = mask[:, w - 1].mean()
        return float((top + bottom + left + right) / 4.0)

    def _area_fraction(self, mask: np.ndarray):
        h, w = mask.shape[:2]
        return float(mask.sum()) / float(h * w + 1e-9)

    def _strip_like(self, mask: np.ndarray):
        bb = self._bbox_from_mask(mask)
        if bb is None:
            return False, None
        x1, y1, x2, y2 = bb
        bw = max(1, x2 - x1 + 1)
        bh = max(1, y2 - y1 + 1)
        aspect = max(bw / bh, bh / bw)
        return True, (bw, bh, aspect)

    def _adjust_score(self, base_score: float, mask: np.ndarray):
        bt = self._border_touch_fraction(mask)
        score = base_score
        if bt >= self.border_touch_frac_thr:
            score -= self.border_penalty

        af = self._area_fraction(mask)
        if af >= self.max_area_frac:
            score -= self.area_penalty

        ok, info = self._strip_like(mask)
        if ok and info is not None:
            _, _, aspect = info
            if (aspect >= self.strip_aspect_thr) and (af <= self.strip_area_frac_max):
                score -= self.strip_penalty

        return float(score), {"border_touch": bt, "area_frac": af}

    def _nms(self, items, iou_thr: float):
        kept = []
        for it in items:
            discard = False
            for kt in kept:
                if self._iou(it["mask"], kt["mask"]) >= iou_thr:
                    discard = True
                    break
            if not discard:
                kept.append(it)
        return kept

    def run(self, image_rgb, validated_masks: list[dict]):
        if not self.enabled:
            return validated_masks

        if not validated_masks:
            return validated_masks

        prepared = []
        for v in validated_masks:
            m = self._get_mask(v)
            if m is None:
                continue

            m = np.array(m).astype(bool)
            base = self._get_base_score(v)
            label = self._get_label(v)

            adj, diag = self._adjust_score(base, m)

            v = dict(v) 
            v.setdefault("postprocess", {})
            v["postprocess"]["identity_resolver"] = {
                "base_score": float(base),
                "adjusted_score": float(adj),
                **diag,
            }

            prepared.append({"verdict": v, "mask": m, "label": label, "score": adj})

        if not prepared:
            return validated_masks

        prepared.sort(key=lambda x: x["score"], reverse=True)

        by_label = {}
        for it in prepared:
            by_label.setdefault(it["label"], []).append(it)

        kept_all = []
        for lbl, items in by_label.items():
            items.sort(key=lambda x: x["score"], reverse=True)
            kept = self._nms(items, self.nms_iou_thr_same_label)
            kept_all.extend(kept)

        kept_all.sort(key=lambda x: x["score"], reverse=True)
        if self.use_global_nms:
            kept_all = self._nms(kept_all, self.nms_iou_thr_global)

        kept_all.sort(key=lambda x: x["score"], reverse=True)
        kept_all = kept_all[: self.top_k]

        out = [it["verdict"] for it in kept_all]
        logger.info(
            f"[IdentityResolver] in={len(validated_masks)} -> out={len(out)} "
            f"(top_k={self.top_k}, nms_lbl={self.nms_iou_thr_same_label}, nms_g={self.nms_iou_thr_global})"
        )
        return out
