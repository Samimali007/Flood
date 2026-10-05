"""Rough water depth estimation module.

Calculates approximate water depth by extracting water surface elevation
along the flood boundary perimeter and subtracting terrain elevation.
Labels outputs strictly as "rough estimate" and reports uncertainty metrics.
Skips calculation if DEM quality or vertical accuracy is insufficient.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion

logger = logging.getLogger(__name__)


@dataclass
class DepthEstimationResult:
    status: str  # "success" or "skipped"
    depth_grid_m: Optional[np.ndarray]
    mean_depth_m: Optional[float]
    max_depth_m: Optional[float]
    volume_m3: Optional[float]
    label: str = "rough estimate"
    method_documentation: str = (
        "Calculated by sampling DEM elevations along the exterior waterline boundary of observed flood polygons "
        "and interpolating local water-surface elevation across inundated pixels (Water Surface Elevation - DEM). "
        "Sensitive to DEM vertical accuracy, tree canopies, and hydrodynamic slope."
    )
    uncertainty_indicators: Dict[str, float] = None


def estimate_flood_depth_boundary(
    flood_mask: np.ndarray,
    dem: np.ndarray,
    pixel_area_m2: float = 100.0,
    min_dem_valid_fraction: float = 0.9,
    max_boundary_std_m: float = 8.0,
) -> DepthEstimationResult:
    """Estimates rough flood water depth from flood boundary elevations.

    Skips if DEM is low quality (too many NaNs or excessive boundary variance).
    """
    flooded = flood_mask.astype(bool)
    if not np.any(flooded):
        return DepthEstimationResult(
            status="skipped",
            depth_grid_m=None,
            mean_depth_m=0.0,
            max_depth_m=0.0,
            volume_m3=0.0,
            uncertainty_indicators={"reason": "No flooded pixels observed"},
        )

    # Check DEM validity over flooded extent
    flooded_dem = dem[flooded]
    valid_fraction = np.sum(np.isfinite(flooded_dem)) / len(flooded_dem)
    if valid_fraction < min_dem_valid_fraction:
        logger.warning(f"DEM quality poor ({valid_fraction*100:.1f}% valid). Skipping rough depth estimation.")
        return DepthEstimationResult(
            status="skipped",
            depth_grid_m=None,
            mean_depth_m=None,
            max_depth_m=None,
            volume_m3=None,
            uncertainty_indicators={
                "reason": "Poor DEM quality or vertical coverage over inundated area",
                "valid_dem_fraction": round(valid_fraction, 3),
            },
        )

    # Extract flood boundary perimeter: (dilated AND NOT eroded)
    dilated = binary_dilation(flooded)
    eroded = binary_erosion(flooded)
    boundary = dilated & (~eroded)

    boundary_elevs = dem[boundary & np.isfinite(dem)]
    if len(boundary_elevs) < 10:
        return DepthEstimationResult(
            status="skipped",
            depth_grid_m=None,
            mean_depth_m=None,
            max_depth_m=None,
            volume_m3=None,
            uncertainty_indicators={"reason": "Insufficient boundary perimeter samples"},
        )

    # Estimate water surface level (use median / 75th percentile to guard against cutbanks/dikes)
    wse_estimate = float(np.percentile(boundary_elevs, 75))
    elev_std = float(np.std(boundary_elevs))

    # Calculate depth per pixel
    depth_grid = np.zeros(dem.shape, dtype=np.float32)
    depth_grid[flooded] = np.maximum(0.0, wse_estimate - dem[flooded])

    valid_depths = depth_grid[flooded]
    mean_depth = float(np.mean(valid_depths))
    max_depth = float(np.max(valid_depths))
    volume = float(np.sum(valid_depths) * pixel_area_m2)

    uncertainty = {
        "boundary_elevation_std_m": round(elev_std, 2),
        "water_surface_elev_p75_m": round(wse_estimate, 2),
        "estimated_depth_uncertainty_range_m": round(max(elev_std, 1.0), 2),
    }

    return DepthEstimationResult(
        status="success",
        depth_grid_m=depth_grid,
        mean_depth_m=round(mean_depth, 2),
        max_depth_m=round(max_depth, 2),
        volume_m3=round(volume, 1),
        uncertainty_indicators=uncertainty,
    )
