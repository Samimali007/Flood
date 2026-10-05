"""Speckle filtering module for SAR backscatter data (Lee filter and Median filter)."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter, uniform_filter

from backend.app.services.flood_extent.config import SpeckleFilterType


def lee_filter(img: np.ndarray, size: int = 5, damping: float = 1.0) -> np.ndarray:
    """Applies Lee speckle filter to 2D SAR image.

    Can operate on linear power or dB images.
    Preserves edges while smoothing homogeneous speckle noise:
      y = mean + W * (x - mean)
      where W = var / (var + noise_var)
    """
    if size % 2 == 0:
        size += 1

    valid = np.isfinite(img)
    filled = np.where(valid, img, 0.0)

    # Local mean and local squared mean using uniform filter
    local_mean = uniform_filter(filled, size=size, mode="reflect")
    local_mean_sq = uniform_filter(filled**2, size=size, mode="reflect")

    # Local variance: Var(X) = E[X^2] - (E[X])^2
    local_var = np.maximum(local_mean_sq - local_mean**2, 0.0)

    # Standard Lee filter for speckle reduction
    # In homogeneous regions, local_var ~ noise_var -> weights ~ 0 -> filtered = local_mean
    # Near true edges, local_var >> noise_var -> weights ~ 1 -> filtered = filled (preserves edges)
    # For isolated noise spikes (1 pixel in a 5x5 window of 25 pixels), the pixel has a local mean around (24*50 + 200)/25 = 56
    # noise_var can be estimated from the local mean and damping factor
    noise_var = (local_mean * 0.5)**2 * damping

    # Weighting factor k = (local_var - noise_var) / local_var
    # For smoothing isolated speckles: weights = local_var / (local_var + noise_var)
    weights = local_var / (local_var + noise_var + 1e-7)
    weights = np.clip(weights, 0.0, 1.0)

    filtered = local_mean + weights * (filled - local_mean)
    filtered[~valid] = np.nan
    return filtered.astype(np.float32)


def apply_speckle_filter(
    img: np.ndarray,
    filter_type: SpeckleFilterType = SpeckleFilterType.LEE,
    size: int = 5,
    damping: float = 1.0,
) -> np.ndarray:
    """Applies the configured speckle filter to the SAR raster."""
    if filter_type == SpeckleFilterType.NONE:
        return img.copy()
    elif filter_type == SpeckleFilterType.LEE:
        return lee_filter(img, size=size, damping=damping)
    elif filter_type == SpeckleFilterType.MEDIAN:
        valid = np.isfinite(img)
        filled = np.where(valid, img, np.nanmedian(img) if np.sum(valid) > 0 else 0.0)
        med = median_filter(filled, size=size, mode="reflect")
        med[~valid] = np.nan
        return med.astype(np.float32)
    else:
        raise ValueError(f"Unknown speckle filter type: {filter_type}")
