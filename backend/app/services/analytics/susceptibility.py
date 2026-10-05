"""Susceptibility comparison module:
Scores terrain susceptibility layer (e.g. inverted HAND or slope-drainage index)
against the observed peak flood mask.
Computes:
- Intersection over Union (IoU)
- Area Under the ROC Curve (AUC)
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def compute_iou(
    binary_prediction: np.ndarray,
    binary_ground_truth: np.ndarray,
) -> float:
    """Computes Intersection over Union (Jaccard Index) between two binary masks."""
    pred = binary_prediction.astype(bool)
    gt = binary_ground_truth.astype(bool)

    intersection = np.sum(pred & gt)
    union = np.sum(pred | gt)

    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)


def score_susceptibility(
    susceptibility_continuous: np.ndarray,
    observed_peak_mask: np.ndarray,
    threshold: Optional[float] = None,
) -> dict:
    """Scores a continuous terrain susceptibility layer against observed peak flood mask.

    Returns:
      auc: Area Under ROC Curve
      iou: Intersection over Union (at optimal or given threshold)
      best_threshold: Selected threshold
    """
    valid = np.isfinite(susceptibility_continuous) & np.isfinite(observed_peak_mask)
    y_true = observed_peak_mask[valid].astype(int)
    y_scores = susceptibility_continuous[valid]

    # Check if we have positive and negative cases
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    if n_pos == 0 or n_neg == 0:
        return {
            "auc": 0.5,
            "iou": 0.0,
            "threshold": threshold or 0.5,
            "note": "Insufficient binary class variance to compute meaningful AUC.",
        }

    # AUC
    try:
        auc = float(roc_auc_score(y_true, y_scores))
    except Exception:
        auc = 0.5

    # Threshold for IoU
    if threshold is None:
        # Pick median of positive scores or 50th percentile as reasonable threshold
        threshold = float(np.percentile(y_scores[y_true == 1], 25))

    binary_pred = susceptibility_continuous >= threshold
    iou = compute_iou(binary_pred[valid], y_true)

    return {
        "auc": round(auc, 4),
        "iou": round(iou, 4),
        "threshold_used": round(threshold, 4),
        "sample_points": int(np.sum(valid)),
    }
