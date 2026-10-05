"""Configuration schemas and data models for Flood Extent Analysis Service."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field


class SpeckleFilterType(str, Enum):
    NONE = "none"
    LEE = "lee"
    MEDIAN = "median"


class DetectionMethod(str, Enum):
    THRESHOLD = "threshold"      # Single-image dB threshold (manual or Otsu)
    CHANGE = "change"            # Change detection vs. baseline composite (difference dB threshold)


class ThresholdMode(str, Enum):
    MANUAL = "manual"
    OTSU = "otsu"


class FilterConfig(BaseModel):
    filter_type: SpeckleFilterType = SpeckleFilterType.LEE
    window_size: int = Field(default=5, ge=3, description="Window size for Lee or median filter (must be odd).")
    damping_factor: float = Field(default=1.0, description="Damping factor for Lee filter.")


class FloodDetectionConfig(BaseModel):
    method: DetectionMethod = DetectionMethod.THRESHOLD
    threshold_mode: ThresholdMode = ThresholdMode.MANUAL
    water_threshold_db: float = Field(
        default=-16.0,
        description="Threshold (dB) below which backscatter is detected as specular water."
    )
    change_threshold_db: float = Field(
        default=-3.0,
        description="Drop in dB (event - baseline) to detect newly inundated areas."
    )
    # Morphological cleaning
    min_size_pixels: int = Field(
        default=25,
        description="Minimum object size in pixels to remove isolated noise speckles."
    )
    hole_size_pixels: int = Field(
        default=25,
        description="Maximum hole size in pixels to fill small interior holes in flood extents."
    )
    # DEM Terrain filters
    max_slope_deg: float = Field(
        default=10.0,
        description="Maximum ground slope in degrees; steeper terrain is masked out as radar layover/shadow."
    )
    # Masks
    use_permanent_water_mask: bool = True
    use_terrain_slope_mask: bool = True
    use_urban_mask: bool = False


# Known SAR limitations to explicitly expose in every output
KNOWN_LIMITATIONS: List[str] = [
    "Flooded vegetation: SAR C-band and L-band double-bounce scattering from emergent vegetation may appear bright and escape dark-surface water detection.",
    "Wind-roughened open water: Wind creates capillary waves that increase backscatter, potentially causing false negatives for water.",
    "Radar shadow and layover: Steep topography causes dark radar shadows (false water) and geometric layover.",
    "Urban areas: Dense structures cause corner-reflector bounce, preventing direct specular water detection in narrow streets without interferometric coherence.",
    "Threshold sensitivity: Manual or bimodal Otsu thresholds can vary depending on soil moisture and local land cover.",
]
