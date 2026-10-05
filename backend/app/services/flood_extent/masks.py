"""Mask generation and management for SAR flood detection.

Handles:
1. Permanent water masking (JRC Global Surface Water, OSM, or pre-event baseline water).
2. Steep slope and radar shadow masking derived from DEM.
3. Urban area masking.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject

logger = logging.getLogger(__name__)


def compute_slope_degrees(
    dem: np.ndarray,
    dx: float,
    dy: float,
) -> np.ndarray:
    """Calculates ground slope in degrees from a DEM raster grid using central differences."""
    dx = abs(dx)
    dy = abs(dy)
    # 2nd order central difference
    gy, gx = np.gradient(dem, dy, dx)
    slope_rad = np.arctan(np.sqrt(gx**2 + gy**2))
    return np.degrees(slope_rad).astype(np.float32)


def generate_dem_slope_mask(
    dem_path: Union[str, Path],
    target_transform: Affine,
    target_crs: CRS,
    target_shape: Tuple[int, int],
    max_slope_deg: float = 10.0,
) -> np.ndarray:
    """Reprojects DEM to target SAR grid, computes slope, and creates a mask (True = exclude)."""
    dem_file = Path(dem_path)
    if not dem_file.exists():
        logger.warning(f"DEM file not found: {dem_file}. Slope mask disabled.")
        return np.zeros(target_shape, dtype=bool)

    height, width = target_shape
    dem_reprojected = np.full((height, width), np.nan, dtype=np.float32)

    with rasterio.open(dem_file) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dem_reprojected,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=target_transform,
            dst_crs=target_crs,
            resampling=Resampling.bilinear,
            src_nodata=src.nodata,
            dst_nodata=np.nan,
        )

    # Pixel resolution in target CRS
    pixel_dx = abs(target_transform.a)
    pixel_dy = abs(target_transform.e)

    # Handle geographic coordinates approximation (degrees to meters at centroid)
    if target_crs.is_geographic:
        # Approximate lat from transform
        _, top_lat = target_transform * (0, 0)
        _, bot_lat = target_transform * (0, height)
        mid_lat = np.radians((top_lat + bot_lat) / 2.0)
        meters_per_deg_lat = 111320.0
        meters_per_deg_lon = 111320.0 * np.cos(mid_lat)
        pixel_dx = pixel_dx * meters_per_deg_lon
        pixel_dy = pixel_dy * meters_per_deg_lat

    slope = compute_slope_degrees(dem_reprojected, pixel_dx, pixel_dy)
    # Mask out areas with slope > max_slope_deg or invalid DEM
    exclude_mask = np.isnan(dem_reprojected) | (slope > max_slope_deg)
    return exclude_mask


def generate_raster_mask(
    mask_path: Union[str, Path],
    target_transform: Affine,
    target_crs: CRS,
    target_shape: Tuple[int, int],
    threshold: float = 1.0,
) -> np.ndarray:
    """Reprojects an auxiliary raster (e.g. permanent water or urban mask) to target grid."""
    path = Path(mask_path)
    if not path.exists():
        logger.warning(f"Mask file not found: {path}. Mask disabled.")
        return np.zeros(target_shape, dtype=bool)

    height, width = target_shape
    destination = np.zeros((height, width), dtype=np.float32)

    with rasterio.open(path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=target_transform,
            dst_crs=target_crs,
            resampling=Resampling.nearest,
            src_nodata=src.nodata,
            dst_nodata=0.0,
        )

    return destination >= threshold


def derive_baseline_permanent_water(
    baseline_composite_db: np.ndarray,
    water_threshold_db: float = -16.0,
) -> np.ndarray:
    """Derives a pre-event permanent water mask directly from a baseline composite image."""
    valid = np.isfinite(baseline_composite_db)
    perm_water = valid & (baseline_composite_db < water_threshold_db)
    return perm_water
