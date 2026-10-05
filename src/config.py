"""Event Configuration models and loader for SAR flood analysis.

Supports YAML configuration with Pydantic validation:
- id, name, AOI polygon (GeoJSON / list of coordinates / Shapely Polygon / WKT)
- event date range (start, end)
- baseline date range (start, end)
- DEM path
- sensor: sentinel1 | nisar
- polarization: VV, VH, HH, HV, or combinations
- thresholds: water detection threshold (dB), change threshold, etc.
- permanent-water source: e.g. jrc_occurrence, osm_water, esa_worldcover, none
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from shapely.geometry import Polygon, mapping, shape
from shapely import wkt


class SensorType(str, Enum):
    SENTINEL1 = "sentinel1"
    NISAR = "nisar"


class PermanentWaterSource(str, Enum):
    JRC_OCCURRENCE = "jrc_occurrence"
    OSM_WATER = "osm_water"
    ESA_WORLDCOVER = "esa_worldcover"
    CUSTOM = "custom"
    NONE = "none"


class DateRange(BaseModel):
    start: date
    end: date

    @model_validator(mode="after")
    def validate_range(self) -> DateRange:
        if self.start > self.end:
            raise ValueError(f"Start date ({self.start}) must not be after end date ({self.end})")
        return self


class ThresholdConfig(BaseModel):
    """Threshold settings for flood detection from SAR backscatter (dB)."""
    water_threshold_db: float = Field(
        default=-16.0,
        description="Threshold below which backscatter (dB) is classified as specular water surface."
    )
    change_threshold_db: Optional[float] = Field(
        default=-3.0,
        description="Drop in backscatter (event - baseline) to flag newly flooded area."
    )
    hand_threshold_m: Optional[float] = Field(
        default=15.0,
        description="Height Above Nearest Drainage (HAND) max filter in meters."
    )
    slope_threshold_deg: Optional[float] = Field(
        default=10.0,
        description="Max terrain slope in degrees to filter false water detections."
    )


class PermanentWaterConfig(BaseModel):
    source: PermanentWaterSource = PermanentWaterSource.JRC_OCCURRENCE
    path: Optional[str] = Field(
        default=None,
        description="Local raster or vector path if source is custom or pre-cached."
    )
    occurrence_min_percent: float = Field(
        default=80.0,
        description="Occurrence percentage threshold for permanent water mask (e.g. JRC)."
    )


class EventConfig(BaseModel):
    id: str = Field(..., description="Unique slug or identifier for the flood event.")
    name: str = Field(..., description="Descriptive human-readable event name.")
    aoi: Dict[str, Any] = Field(
        ...,
        description="Area of Interest represented as a GeoJSON geometry dictionary or Polygon coordinates."
    )
    event_date_range: DateRange = Field(..., description="Acquisition date window for the flood event.")
    baseline_date_range: DateRange = Field(..., description="Acquisition date window for pre-event / baseline reference.")
    dem_path: Optional[str] = Field(
        default=None,
        description="File path to the Digital Elevation Model (GeoTIFF) covering the AOI."
    )
    sensor: SensorType = Field(..., description="SAR sensor: sentinel1 or nisar.")
    polarization: str = Field(
        default="VV",
        description="Target polarization channel(s), e.g. 'VV', 'VH', 'HH', 'HV', 'VV+VH', 'HH+HV'."
    )
    thresholds: ThresholdConfig = Field(default_factory=ThresholdConfig)
    permanent_water: PermanentWaterConfig = Field(default_factory=PermanentWaterConfig)

    @field_validator("aoi", mode="before")
    @classmethod
    def parse_aoi(cls, v: Any) -> Dict[str, Any]:
        """Accept GeoJSON dict, Shapely Polygon, WKT string, or list of [lon, lat] coordinates."""
        if isinstance(v, Polygon):
            return mapping(v)
        if isinstance(v, str):
            # Try WKT parsing
            try:
                geom = wkt.loads(v)
                return mapping(geom)
            except Exception as e:
                raise ValueError(f"Failed to parse AOI from WKT string: {e}")
        if isinstance(v, list):
            # Check if list of coordinates [[lon, lat], ...]
            coords = v
            # If not closed, close it
            if coords and coords[0] != coords[-1]:
                coords = coords + [coords[0]]
            poly = Polygon(coords)
            return mapping(poly)
        if isinstance(v, dict):
            if "type" in v and "coordinates" in v:
                # Valid geojson-like dict
                return v
            elif "geometry" in v:
                return v["geometry"]
        raise ValueError(f"Unsupported AOI format: {type(v)}. Provide GeoJSON dict, WKT, or coordinate list.")

    def get_aoi_polygon(self) -> Polygon:
        """Returns the AOI as a Shapely Polygon."""
        geom = shape(self.aoi)
        if not isinstance(geom, Polygon):
            # Return convex hull or envelope if multipolygon
            return geom
        return geom

    def get_aoi_wkt(self) -> str:
        """Returns the AOI Polygon in Well-Known Text format (e.g. for asf_search)."""
        return self.get_aoi_polygon().wkt

    @classmethod
    def from_yaml(cls, path_or_stream: Union[str, Path, Any]) -> EventConfig:
        """Loads and validates EventConfig from a YAML file path or stream."""
        if isinstance(path_or_stream, (str, Path)):
            with open(path_or_stream, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        else:
            data = yaml.safe_load(path_or_stream)
        
        # Support aliases like 'permanent_water_source' flat field if provided
        if "permanent_water_source" in data and "permanent_water" not in data:
            data["permanent_water"] = {"source": data.pop("permanent_water_source")}
        return cls(**data)

    def to_yaml(self, path: Optional[Union[str, Path]] = None) -> str:
        """Dumps EventConfig to YAML string, optionally saving to path."""
        # Convert model to python dict with string dates
        data = self.model_dump(mode="json")
        out = yaml.dump(data, sort_keys=False, default_flow_style=False)
        if path:
            Path(path).write_text(out, encoding="utf-8")
        return out
