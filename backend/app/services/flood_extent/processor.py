"""Core flood extent processor service.

Orchestrates:
1. Co-registration to reference grid with orbit validation.
2. Speckle filtering (Lee or Median).
3. Water detection via:
   (a) Single-date dB threshold (manual or Otsu on bimodal tiles)
   (b) Change detection vs baseline composite (difference dB threshold)
4. Masking:
   - Permanent water (JRC, auxiliary raster, or pre-event baseline water)
   - Steep DEM terrain slope and radar shadow
   - Optional urban mask
5. Morphological cleanup (small objects removal and hole filling).
6. Metrics calculation:
   - Flooded area (km2)
   - Valid pixel coverage fraction
7. Cloud-Optimized GeoTIFF (COG) generation with provenance and known SAR limitations exposed.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine
from skimage.filters import threshold_otsu

from backend.app.services.flood_extent.config import (
    DetectionMethod,
    FilterConfig,
    FloodDetectionConfig,
    KNOWN_LIMITATIONS,
    ThresholdMode,
)
from backend.app.services.flood_extent.coregistration import (
    align_scenes_to_reference,
    check_orbit_consistency,
    coregister_scene_to_grid,
)
from backend.app.services.flood_extent.filters import apply_speckle_filter
from backend.app.services.flood_extent.masks import (
    derive_baseline_permanent_water,
    generate_dem_slope_mask,
    generate_raster_mask,
)
from backend.app.services.flood_extent.morphology import clean_flood_mask
from src.readers import Scene

logger = logging.getLogger(__name__)


@dataclass
class FloodExtentResult:
    """Output summary and metadata for a processed acquisition date."""
    scene_id: str
    acquisition_date: str
    sensor: str
    relative_orbit: Optional[int]
    polarization: str
    flooded_area_km2: float
    flooded_pixel_count: int
    valid_pixel_fraction: float
    threshold_used_db: float
    method: str
    cog_path: Optional[str]
    parameters: Dict[str, Any]
    provenance: Dict[str, Any]
    limitations: List[str]


def compute_pixel_area_km2(transform: Affine, crs: CRS, shape: Tuple[int, int]) -> float:
    """Computes single-pixel area in km²."""
    dx = abs(transform.a)
    dy = abs(transform.e)

    if crs.is_geographic:
        # Latitude from mid-row
        height, _ = shape
        _, top_lat = transform * (0, 0)
        _, bot_lat = transform * (0, height)
        mid_lat = np.radians((top_lat + bot_lat) / 2.0)
        m_per_deg_lat = 111320.0
        m_per_deg_lon = 111320.0 * np.cos(mid_lat)
        dx_m = dx * m_per_deg_lon
        dy_m = dy * m_per_deg_lat
    else:
        dx_m = dx
        dy_m = dy

    area_m2 = dx_m * dy_m
    return area_m2 / 1e6


def compute_otsu_threshold(arr_db: np.ndarray, fallback_db: float = -16.0) -> float:
    """Computes optimal Otsu threshold on valid backscatter pixels."""
    valid = arr_db[np.isfinite(arr_db)]
    if len(valid) < 100:
        return fallback_db
    try:
        # Clip extreme outliers for stability
        p1, p99 = np.percentile(valid, [1, 99])
        clipped = valid[(valid >= p1) & (valid <= p99)]
        thresh = float(threshold_otsu(clipped))
        # Clamp threshold to reasonable SAR water backscatter range (-25 dB to -10 dB)
        thresh = max(min(thresh, -10.0), -25.0)
        return thresh
    except Exception as e:
        logger.warning(f"Otsu thresholding failed ({e}), using fallback {fallback_db} dB")
        return fallback_db


def create_baseline_composite(baseline_scenes: List[Scene]) -> Optional[np.ndarray]:
    """Generates a median baseline composite backscatter grid across all baseline acquisitions."""
    if not baseline_scenes:
        return None

    # Check orbit consistency
    check_orbit_consistency(baseline_scenes)

    # Co-register all baseline scenes to the first baseline scene grid
    aligned_arrays, _, _, _ = align_scenes_to_reference(baseline_scenes)
    stack = np.stack(aligned_arrays, axis=0)

    # Median over time axis ignoring NaNs
    composite_db = np.nanmedian(stack, axis=0).astype(np.float32)
    return composite_db


class FloodExtentService:
    """Service to process SAR flood extents per satellite acquisition date."""

    def __init__(
        self,
        config: FloodDetectionConfig = FloodDetectionConfig(),
        filter_config: FilterConfig = FilterConfig(),
    ) -> None:
        self.config = config
        self.filter_config = filter_config

    def process_date(
        self,
        event_scene: Scene,
        baseline_scene_or_composite: Optional[Union[Scene, np.ndarray]] = None,
        dem_path: Optional[Union[str, Path]] = None,
        permanent_water_mask_path: Optional[Union[str, Path]] = None,
        urban_mask_path: Optional[Union[str, Path]] = None,
        output_cog_path: Optional[Union[str, Path]] = None,
    ) -> Tuple[np.ndarray, FloodExtentResult]:
        """Processes flood extent for a single event scene."""
        # 1. Base grid parameters from event scene
        grid_transform = event_scene.transform
        grid_crs = event_scene.crs
        grid_shape = event_scene.data_db.shape
        pixel_area_km2 = compute_pixel_area_km2(grid_transform, grid_crs, grid_shape)

        # 2. Speckle filter
        filtered_event_db = apply_speckle_filter(
            event_scene.data_db,
            filter_type=self.filter_config.filter_type,
            size=self.filter_config.window_size,
            damping=self.filter_config.damping_factor,
        )

        valid_mask = np.isfinite(filtered_event_db)
        total_pixels = grid_shape[0] * grid_shape[1]
        valid_pixel_count = int(np.sum(valid_mask))
        valid_fraction = float(valid_pixel_count / total_pixels) if total_pixels > 0 else 0.0

        # 3. Detection logic
        raw_flood_mask = np.zeros(grid_shape, dtype=bool)
        threshold_used = self.config.water_threshold_db

        if self.config.method == DetectionMethod.THRESHOLD:
            if self.config.threshold_mode == ThresholdMode.OTSU:
                threshold_used = compute_otsu_threshold(
                    filtered_event_db,
                    fallback_db=self.config.water_threshold_db,
                )
            else:
                threshold_used = self.config.water_threshold_db

            # Specular water has low backscatter
            raw_flood_mask = valid_mask & (filtered_event_db < threshold_used)

        elif self.config.method == DetectionMethod.CHANGE:
            if baseline_scene_or_composite is None:
                raise ValueError("Change detection method requires a baseline scene or baseline composite.")

            if isinstance(baseline_scene_or_composite, Scene):
                # Reproject baseline to event scene grid
                baseline_db = coregister_scene_to_grid(
                    baseline_scene_or_composite,
                    target_transform=grid_transform,
                    target_crs=grid_crs,
                    target_shape=grid_shape,
                )
            else:
                baseline_db = baseline_scene_or_composite

            filtered_baseline_db = apply_speckle_filter(
                baseline_db,
                filter_type=self.filter_config.filter_type,
                size=self.filter_config.window_size,
                damping=self.filter_config.damping_factor,
            )

            # Difference: drop in backscatter indicates inundation
            diff_db = filtered_event_db - filtered_baseline_db
            threshold_used = self.config.change_threshold_db
            raw_flood_mask = valid_mask & np.isfinite(filtered_baseline_db) & (diff_db < threshold_used)

        # 4. Exclusion Masks
        exclusion_mask = np.zeros(grid_shape, dtype=bool)

        # Permanent Water Mask
        if self.config.use_permanent_water_mask:
            if permanent_water_mask_path and Path(permanent_water_mask_path).exists():
                perm_mask = generate_raster_mask(
                    permanent_water_mask_path,
                    target_transform=grid_transform,
                    target_crs=grid_crs,
                    target_shape=grid_shape,
                )
                exclusion_mask |= perm_mask
            elif baseline_scene_or_composite is not None:
                # Derive from baseline
                base_arr = (
                    baseline_scene_or_composite.data_db
                    if isinstance(baseline_scene_or_composite, Scene)
                    else baseline_scene_or_composite
                )
                perm_mask = derive_baseline_permanent_water(
                    base_arr,
                    water_threshold_db=self.config.water_threshold_db,
                )
                exclusion_mask |= perm_mask

        # Terrain Slope / Shadow Mask from DEM
        if self.config.use_terrain_slope_mask and dem_path:
            slope_mask = generate_dem_slope_mask(
                dem_path,
                target_transform=grid_transform,
                target_crs=grid_crs,
                target_shape=grid_shape,
                max_slope_deg=self.config.max_slope_deg,
            )
            exclusion_mask |= slope_mask

        # Urban Mask
        if self.config.use_urban_mask and urban_mask_path:
            urban_mask = generate_raster_mask(
                urban_mask_path,
                target_transform=grid_transform,
                target_crs=grid_crs,
                target_shape=grid_shape,
            )
            exclusion_mask |= urban_mask

        # Apply exclusion
        flood_masked = raw_flood_mask & (~exclusion_mask)

        # 5. Morphological Cleanup
        cleaned_mask = clean_flood_mask(
            flood_masked,
            min_size_pixels=self.config.min_size_pixels,
            hole_size_pixels=self.config.hole_size_pixels,
        )

        # 6. Metrics
        flooded_pixel_count = int(np.sum(cleaned_mask))
        flooded_area_km2 = float(flooded_pixel_count * pixel_area_km2)

        # 7. Provenance & Metadata
        acq_str = (
            event_scene.acquisition_time.isoformat()
            if event_scene.acquisition_time
            else datetime.now(timezone.utc).isoformat()
        )

        provenance = {
            "scene_id": event_scene.scene_id,
            "sensor": event_scene.sensor,
            "product_level": event_scene.product_level,
            "polarization": event_scene.polarization,
            "relative_orbit": event_scene.relative_orbit,
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "crs": event_scene.crs.to_string(),
            "pixel_resolution": [abs(grid_transform.a), abs(grid_transform.e)],
        }

        # 8. Export Cloud-Optimized GeoTIFF (COG) if path provided
        cog_saved_path = None
        if output_cog_path:
            out_p = Path(output_cog_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            self._write_cog(
                mask_bool=cleaned_mask,
                valid_mask=valid_mask,
                transform=grid_transform,
                crs=grid_crs,
                out_path=out_p,
                meta_tags={
                    "SCENE_ID": event_scene.scene_id,
                    "FLOODED_AREA_KM2": str(round(flooded_area_km2, 4)),
                    "METHOD": self.config.method.value,
                    "THRESHOLD_DB": str(threshold_used),
                },
            )
            cog_saved_path = str(out_p)

        result = FloodExtentResult(
            scene_id=event_scene.scene_id,
            acquisition_date=acq_str,
            sensor=event_scene.sensor,
            relative_orbit=event_scene.relative_orbit,
            polarization=event_scene.polarization,
            flooded_area_km2=round(flooded_area_km2, 4),
            flooded_pixel_count=flooded_pixel_count,
            valid_pixel_fraction=round(valid_fraction, 4),
            threshold_used_db=round(threshold_used, 2),
            method=self.config.method.value,
            cog_path=cog_saved_path,
            parameters=self.config.model_dump(),
            provenance=provenance,
            limitations=KNOWN_LIMITATIONS,
        )

        return cleaned_mask, result

    @staticmethod
    def _write_cog(
        mask_bool: np.ndarray,
        valid_mask: np.ndarray,
        transform: Affine,
        crs: CRS,
        out_path: Path,
        meta_tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Writes binary flood mask as a standard Cloud-Optimized GeoTIFF (COG).

        Values:
          1 = Flooded
          0 = Not flooded
          255 = No data / unobserved
        """
        h, w = mask_bool.shape
        out_arr = np.full((h, w), 255, dtype=np.uint8)
        out_arr[valid_mask] = 0
        out_arr[mask_bool] = 1

        profile = {
            "driver": "GTiff",
            "height": h,
            "width": w,
            "count": 1,
            "dtype": np.uint8,
            "crs": crs,
            "transform": transform,
            "nodata": 255,
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "compress": "deflate",
            "interleave": "pixel",
        }

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(out_arr, 1)
            if meta_tags:
                dst.update_tags(**meta_tags)
