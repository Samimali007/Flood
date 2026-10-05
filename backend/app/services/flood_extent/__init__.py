"""Flood Extent Service package."""

from backend.app.services.flood_extent.config import (
    DetectionMethod,
    FilterConfig,
    FloodDetectionConfig,
    KNOWN_LIMITATIONS,
    SpeckleFilterType,
    ThresholdMode,
)
from backend.app.services.flood_extent.coregistration import (
    OrbitMismatchWarning,
    align_scenes_to_reference,
    check_orbit_consistency,
    coregister_scene_to_grid,
)
from backend.app.services.flood_extent.filters import apply_speckle_filter, lee_filter
from backend.app.services.flood_extent.masks import (
    compute_slope_degrees,
    derive_baseline_permanent_water,
    generate_dem_slope_mask,
    generate_raster_mask,
)
from backend.app.services.flood_extent.morphology import clean_flood_mask
from backend.app.services.flood_extent.processor import (
    FloodExtentResult,
    FloodExtentService,
    compute_otsu_threshold,
    compute_pixel_area_km2,
    create_baseline_composite,
)

__all__ = [
    "DetectionMethod",
    "FilterConfig",
    "FloodDetectionConfig",
    "KNOWN_LIMITATIONS",
    "SpeckleFilterType",
    "ThresholdMode",
    "OrbitMismatchWarning",
    "align_scenes_to_reference",
    "check_orbit_consistency",
    "coregister_scene_to_grid",
    "apply_speckle_filter",
    "lee_filter",
    "compute_slope_degrees",
    "derive_baseline_permanent_water",
    "generate_dem_slope_mask",
    "generate_raster_mask",
    "clean_flood_mask",
    "FloodExtentResult",
    "FloodExtentService",
    "compute_otsu_threshold",
    "compute_pixel_area_km2",
    "create_baseline_composite",
]
