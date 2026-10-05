"""Morphological post-processing and cleanup for binary flood masks."""

from __future__ import annotations

import numpy as np
from skimage.morphology import remove_small_holes, remove_small_objects


def clean_flood_mask(
    binary_mask: np.ndarray,
    min_size_pixels: int = 25,
    hole_size_pixels: int = 25,
) -> np.ndarray:
    """Cleans a binary flood mask by removing tiny isolated speckle regions and filling small interior holes.

    Args:
        binary_mask: 2D boolean array (True = flooded, False = unflooded).
        min_size_pixels: Minimum contiguous area in pixels to keep; smaller objects are eliminated.
        hole_size_pixels: Maximum area in pixels of small enclosed dry holes to fill.

    Returns:
        Cleaned 2D boolean array.
    """
    if not np.any(binary_mask):
        return binary_mask.copy()

    # Ensure boolean
    mask_bool = binary_mask.astype(bool)

    # 1. Remove tiny isolated speckle artifacts
    if min_size_pixels > 0:
        try:
            mask_bool = remove_small_objects(mask_bool, max_size=min_size_pixels - 1, connectivity=2)
        except TypeError:
            mask_bool = remove_small_objects(mask_bool, min_size=min_size_pixels, connectivity=2)

    # 2. Fill small dry holes within flood bodies
    if hole_size_pixels > 0:
        try:
            mask_bool = remove_small_holes(mask_bool, max_size=hole_size_pixels, connectivity=2)
        except TypeError:
            mask_bool = remove_small_holes(mask_bool, area_threshold=hole_size_pixels, connectivity=2)

    return mask_bool
