"""Terrain pipeline configuration model."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class TerrainPipelineConfig(BaseModel):
    # Spikes and NoData
    max_valid_elevation_m: float = Field(default=9000.0, description="Upper threshold to filter positive DEM spikes.")
    min_valid_elevation_m: float = Field(default=-500.0, description="Lower threshold to filter negative DEM spikes.")
    spike_relative_threshold_m: float = Field(default=150.0, description="Local elevation difference to flag and interpolate spikes.")

    # Hydrological Conditioning
    fill_depressions: bool = True
    epsilon_slope: float = Field(default=1e-4, description="Small elevation increment applied to resolved depressions to enforce drainage.")

    # Stream Extraction
    stream_accumulation_threshold: int = Field(
        default=50,
        description="Minimum flow accumulation count (pixels) to initiate a likely drainage pathway / stream."
    )

    # Drainage labelling
    drainage_output_label: str = Field(
        default="likely drainage pathways (terrain-derived)",
        description="Strict user-facing naming requirement for terrain-derived flow networks (not observed water flow)."
    )

    # Basin delineation
    min_basin_area_pixels: int = Field(default=100, description="Minimum catchment basin area in pixels.")

    # Outputs
    output_dir: str = Field(default="data/terrain_outputs", description="Directory to save generated terrain COGs.")
    save_cogs: bool = True
