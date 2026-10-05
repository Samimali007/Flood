"""End-to-end Terrain Pipeline service for the event AOI.

Produces:
- Conditioned DEM (despiked, pits filled)
- D8 Flow Direction COG
- Flow Accumulation COG
- Likely drainage pathways (terrain-derived) COG (labeled strictly as terrain-derived, never observed flow)
- Catchment Basins COG
- HAND (Height Above Nearest Drainage) COG
- Slope (degrees) COG
- Parameters in config & provenance saved
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
from rasterio.transform import Affine

from backend.app.services.terrain.conditioning import (
    fill_depressions_priority_flood,
    handle_dem_spikes,
)
from backend.app.services.terrain.config import TerrainPipelineConfig
from backend.app.services.terrain.hydro import (
    compute_d8_flow_direction,
    compute_flow_accumulation,
    compute_hand,
    compute_slope,
    delineate_basins,
    extract_streams,
)

logger = logging.getLogger(__name__)


@dataclass
class TerrainPipelineResult:
    dem_path: str
    crs: str
    shape: Tuple[int, int]
    pixel_resolution_m: Tuple[float, float]
    stream_pixel_count: int
    num_basins: int
    max_hand_m: float
    mean_slope_deg: float
    output_files: Dict[str, str]
    drainage_label: str
    parameters: Dict[str, Any]
    provenance: Dict[str, Any]


class TerrainPipelineService:
    def __init__(self, config: TerrainPipelineConfig = TerrainPipelineConfig()) -> None:
        self.config = config

    def run(
        self,
        dem_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
    ) -> TerrainPipelineResult:
        """Executes full terrain analysis pipeline on the given DEM file."""
        dem_file = Path(dem_path)
        if not dem_file.exists():
            raise FileNotFoundError(f"DEM raster not found at: {dem_file}")

        out_d = Path(output_dir or self.config.output_dir)
        out_d.mkdir(parents=True, exist_ok=True)

        with rasterio.open(dem_file) as src:
            raw_dem = src.read(1).astype(np.float32)
            transform = src.transform
            crs = src.crs
            src_nodata = src.nodata

        # Handle NoData
        if src_nodata is not None:
            raw_dem[np.isclose(raw_dem, src_nodata)] = np.nan

        # Calculate pixel resolution in meters
        dx = abs(transform.a)
        dy = abs(transform.e)
        if crs.is_geographic:
            height, _ = raw_dem.shape
            _, top_lat = transform * (0, 0)
            _, bot_lat = transform * (0, height)
            mid_lat = np.radians((top_lat + bot_lat) / 2.0)
            dx = dx * 111320.0 * np.cos(mid_lat)
            dy = dy * 111320.0

        # 1. Spike handling
        dem_despiked = handle_dem_spikes(
            raw_dem,
            min_elev=self.config.min_valid_elevation_m,
            max_elev=self.config.max_valid_elevation_m,
            spike_relative_thresh=self.config.spike_relative_threshold_m,
        )

        # 2. Priority-flood depression filling
        if self.config.fill_depressions:
            conditioned_dem = fill_depressions_priority_flood(
                dem_despiked,
                epsilon=self.config.epsilon_slope,
            )
        else:
            conditioned_dem = dem_despiked

        # 3. D8 Flow Direction
        fdir = compute_d8_flow_direction(conditioned_dem, dx=dx, dy=dy)

        # 4. Flow Accumulation
        acc = compute_flow_accumulation(fdir)

        # 5. Stream Extraction ("likely drainage pathways (terrain-derived)")
        stream_mask = extract_streams(acc, threshold=self.config.stream_accumulation_threshold)

        # 6. Basins
        basins = delineate_basins(fdir, stream_mask)

        # 7. HAND
        hand = compute_hand(conditioned_dem, fdir, stream_mask)

        # 8. Slope
        slope = compute_slope(conditioned_dem, dx=dx, dy=dy)

        # 9. Save COGs
        output_files = {}
        if self.config.save_cogs:
            stem = dem_file.stem
            output_files = {
                "conditioned_dem": self._write_cog(
                    conditioned_dem, transform, crs, out_d / f"{stem}_conditioned.tif", dtype="float32", nodata=-9999.0
                ),
                "flow_direction": self._write_cog(
                    fdir, transform, crs, out_d / f"{stem}_flow_direction_d8.tif", dtype="uint8", nodata=0
                ),
                "flow_accumulation": self._write_cog(
                    acc, transform, crs, out_d / f"{stem}_flow_accumulation.tif", dtype="int32", nodata=-1
                ),
                # Note strict label in file name & metadata:
                "likely_drainage_pathways": self._write_cog(
                    stream_mask.astype(np.uint8),
                    transform,
                    crs,
                    out_d / f"{stem}_likely_drainage_pathways.tif",
                    dtype="uint8",
                    nodata=255,
                    tags={"DESCRIPTION": self.config.drainage_output_label, "IS_OBSERVED": "FALSE"},
                ),
                "basins": self._write_cog(
                    basins, transform, crs, out_d / f"{stem}_basins.tif", dtype="int32", nodata=0
                ),
                "hand": self._write_cog(
                    hand, transform, crs, out_d / f"{stem}_hand.tif", dtype="float32", nodata=-9999.0
                ),
                "slope": self._write_cog(
                    slope, transform, crs, out_d / f"{stem}_slope_deg.tif", dtype="float32", nodata=-9999.0
                ),
            }

        # Provenance
        provenance = {
            "source_dem": str(dem_file.resolve()),
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "pipeline": "TerrainHydrologyPipeline",
            "crs": crs.to_string(),
            "dx_m": float(dx),
            "dy_m": float(dy),
            "drainage_classification": "Terrain-derived likelihood, not observed water movement",
        }

        # Save provenance json
        prov_path = out_d / f"{dem_file.stem}_provenance.json"
        prov_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        output_files["provenance"] = str(prov_path)

        valid_hand = hand[np.isfinite(hand)]
        valid_slope = slope[np.isfinite(slope)]

        return TerrainPipelineResult(
            dem_path=str(dem_file),
            crs=crs.to_string(),
            shape=raw_dem.shape,
            pixel_resolution_m=(float(dx), float(dy)),
            stream_pixel_count=int(np.sum(stream_mask)),
            num_basins=int(np.max(basins)),
            max_hand_m=float(np.max(valid_hand)) if len(valid_hand) > 0 else 0.0,
            mean_slope_deg=float(np.mean(valid_slope)) if len(valid_slope) > 0 else 0.0,
            output_files=output_files,
            drainage_label=self.config.drainage_output_label,
            parameters=self.config.model_dump(),
            provenance=provenance,
        )

    @staticmethod
    def _write_cog(
        data: np.ndarray,
        transform: Affine,
        crs: CRS,
        path: Path,
        dtype: str = "float32",
        nodata: Optional[float] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """Writes raster array as tiled Cloud-Optimized GeoTIFF (COG)."""
        h, w = data.shape
        out_data = data.copy()
        if nodata is not None:
            out_data = np.where(np.isnan(out_data), nodata, out_data)

        profile = {
            "driver": "GTiff",
            "height": h,
            "width": w,
            "count": 1,
            "dtype": dtype,
            "crs": crs,
            "transform": transform,
            "nodata": nodata,
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "compress": "deflate",
        }

        with rasterio.open(path, "w", **profile) as dst:
            dst.write(out_data.astype(dtype), 1)
            if tags:
                dst.update_tags(**tags)

        return str(path)
