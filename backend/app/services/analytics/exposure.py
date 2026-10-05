"""Exposure analytics module:
Intersects flood masks with OSM roads, OSM buildings, and ESA WorldCover classes.
Reports:
- km of roads affected
- building count affected
- land-cover area (km2) affected
- explicit data-source completeness caveats (OSM completeness varies by region).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from rasterio.crs import CRS
from rasterio.transform import Affine

logger = logging.getLogger(__name__)

# Standard ESA WorldCover 10m class mapping:
# 10: Tree cover
# 20: Shrubland
# 30: Grassland
# 40: Cropland
# 50: Built-up
# 60: Bare / sparse vegetation
# 70: Snow and ice
# 80: Permanent water bodies
# 90: Herbaceous wetland
# 95: Mangroves
# 100: Moss and lichen
WORLDCOVER_CLASSES: Dict[int, str] = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse vegetation",
    70: "Snow and ice",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen",
}

OSM_COMPLETENESS_NOTE: str = (
    "OpenStreetMap (OSM) data completeness varies significantly by region and rural vs urban density. "
    "Unmapped rural buildings or minor secondary/tertiary roads may not be reflected in exposure counts."
)


def compute_landcover_exposure(
    flood_mask: np.ndarray,
    worldcover_raster: np.ndarray,
    pixel_area_km2: float,
) -> Dict[str, float]:
    """Computes flooded area (km2) per ESA WorldCover land-cover class."""
    flooded_pixels = flood_mask.astype(bool)
    classes_present = worldcover_raster[flooded_pixels]

    breakdown: Dict[str, float] = {}
    unique, counts = np.unique(classes_present, return_counts=True)

    for code, count in zip(unique, counts):
        class_name = WORLDCOVER_CLASSES.get(int(code), f"Class {int(code)}")
        area_km2 = float(count * pixel_area_km2)
        breakdown[class_name] = round(area_km2, 4)

    return breakdown


def compute_osm_exposure(
    flood_mask: np.ndarray,
    road_raster_or_lines: Optional[np.ndarray] = None,
    building_raster_or_polygons: Optional[np.ndarray] = None,
    pixel_size_m: float = 10.0,
) -> Dict[str, Any]:
    """Computes OSM road km and building count affected by the flood mask."""
    flooded_bool = flood_mask.astype(bool)

    # Roads: if rasterized (1 = road present), each pixel accounts for ~ pixel_size_m length
    road_km = 0.0
    if road_raster_or_lines is not None:
        road_flooded_pixels = np.sum(flooded_bool & (road_raster_or_lines > 0))
        road_km = float((road_flooded_pixels * pixel_size_m) / 1000.0)

    # Buildings: if rasterized (each building has an ID or 1 = building footprint)
    building_count = 0
    if building_raster_or_polygons is not None:
        bld_intersected = building_raster_or_polygons[flooded_bool]
        bld_ids = np.unique(bld_intersected)
        bld_ids = bld_ids[bld_ids > 0]
        building_count = int(len(bld_ids))

    return {
        "roads_affected_km": round(road_km, 2),
        "buildings_affected_count": building_count,
        "data_source_notes": OSM_COMPLETENESS_NOTE,
    }
