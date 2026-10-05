"""Tests for Terrain Pipeline on a synthetic valley DEM."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

from backend.app.services.terrain import (
    TerrainPipelineConfig,
    TerrainPipelineService,
    compute_d8_flow_direction,
    compute_flow_accumulation,
    compute_hand,
    compute_slope,
    extract_streams,
    fill_depressions_priority_flood,
    handle_dem_spikes,
)


@pytest.fixture
def synthetic_valley_dem(tmp_path: Path) -> Path:
    """Creates a 60x60 synthetic V-shaped valley DEM draining from North to South.

    Terrain geometry:
      - Central channel runs along column 30 from row 0 down to row 59.
      - Elevation descends gently from North (row 0, elev = 50m) to South (row 59, elev = 20m).
      - Valley sides slope steeply upward towards East and West ridges (elev up to 150m).
      - Includes an artificial depression (pit) at row 20, col 30 to test depression filling.
      - Includes an unphysical positive spike at row 10, col 10 (elev = 800m).
      - Includes an unphysical negative pit at row 10, col 50 (elev = -1000m).
    """
    dem_path = tmp_path / "synthetic_valley.tif"
    rows, cols = 60, 60
    elev = np.zeros((rows, cols), dtype=np.float32)

    for r in range(rows):
        # Base channel elevation along North-South gradient
        channel_elev = 50.0 - (r * 0.5)  # 50m at row 0, 20.5m at row 59
        for c in range(cols):
            dist_from_channel = abs(c - 30)
            # Quadratic valley sides
            elev[r, c] = channel_elev + (dist_from_channel**2 * 0.15)

    # 1. Artificial depression (sink) in the river channel
    elev[20, 30] -= 10.0  # deep sink

    # 2. Positive spike
    elev[10, 10] = 800.0

    # 3. Negative out-of-bounds pit
    elev[10, 50] = -1000.0

    transform = from_origin(500000.0, 4500000.0, 10.0, 10.0)
    crs = CRS.from_epsg(32632)

    with rasterio.open(
        dem_path,
        "w",
        driver="GTiff",
        height=rows,
        width=cols,
        count=1,
        dtype=np.float32,
        crs=crs,
        transform=transform,
        nodata=-9999.0,
    ) as dst:
        dst.write(elev, 1)

    return dem_path


def test_spike_handling():
    arr = np.full((20, 20), 100.0, dtype=np.float32)
    arr[5, 5] = 9500.0   # exceeds max elev
    arr[5, 15] = -800.0  # below min elev
    arr[12, 12] = 500.0  # local spike +400m

    cleaned = handle_dem_spikes(arr, min_elev=-500.0, max_elev=9000.0, spike_relative_thresh=150.0)

    assert np.isnan(cleaned[5, 5])
    assert np.isnan(cleaned[5, 15])
    assert np.isclose(cleaned[12, 12], 100.0, atol=1.0)


def test_depression_filling():
    dem = np.array([
        [10.0, 10.0, 10.0, 10.0],
        [10.0,  5.0,  5.0, 10.0],  # Sink inside
        [10.0, 10.0,  9.0, 10.0],  # Spillway at (2, 2) = 9.0
        [10.0, 10.0,  8.0, 10.0],
    ], dtype=np.float32)

    filled = fill_depressions_priority_flood(dem, epsilon=1e-4)

    # Sink at (1, 1) and (1, 2) must be filled at least to spillway elevation
    assert filled[1, 1] >= 9.0
    assert filled[1, 2] >= 9.0
    assert filled[2, 2] == 9.0
    assert filled[3, 2] == 8.0


def test_d8_flow_direction_and_accumulation():
    # Inclined plane sloping towards South (increasing row)
    dem = np.zeros((10, 10), dtype=np.float32)
    for r in range(10):
        dem[r, :] = 100.0 - r * 10.0

    fdir = compute_d8_flow_direction(dem, dx=10.0, dy=10.0)
    # Direction 4 is South in ESRI D8
    assert np.all(fdir[0:9, :] == 4)

    acc = compute_flow_accumulation(fdir)
    # Top row has accumulation 1
    assert np.all(acc[0, :] == 1)
    # Bottom row has accumulation 10
    assert np.all(acc[9, :] == 10)


def test_terrain_pipeline_end_to_end(synthetic_valley_dem: Path, tmp_path: Path):
    out_dir = tmp_path / "terrain_cogs"

    config = TerrainPipelineConfig(
        output_dir=str(out_dir),
        stream_accumulation_threshold=40,
        save_cogs=True,
    )

    service = TerrainPipelineService(config=config)
    result = service.run(synthetic_valley_dem)

    # 1. Output Files and strict labeling
    assert result.drainage_label == "likely drainage pathways (terrain-derived)"
    assert "likely_drainage_pathways" in result.output_files
    stream_cog = Path(result.output_files["likely_drainage_pathways"])
    assert stream_cog.exists()

    # 2. Check COG metadata
    with rasterio.open(stream_cog) as src:
        tags = src.tags()
        assert tags.get("DESCRIPTION") == "likely drainage pathways (terrain-derived)"
        assert tags.get("IS_OBSERVED") == "FALSE"
        assert src.profile["compress"] == "deflate"

    # 3. Stream network along central valley (column 30)
    assert result.stream_pixel_count > 0

    # 4. HAND validation
    # Bottom of the valley should have HAND close to 0
    hand_cog = Path(result.output_files["hand"])
    with rasterio.open(hand_cog) as src:
        hand_arr = src.read(1)
        # Channel cells should have HAND == 0
        assert np.isclose(hand_arr[55, 30], 0.0, atol=0.1)
        # Ridge cells should have positive HAND > 20m
        assert hand_arr[55, 10] > 20.0

    # 5. Slope validation
    assert result.mean_slope_deg > 0.0

    # 6. Provenance saved
    prov_file = Path(result.output_files["provenance"])
    assert prov_file.exists()
    prov_data = json.loads(prov_file.read_text(encoding="utf-8"))
    assert prov_data["pipeline"] == "TerrainHydrologyPipeline"
    assert "terrain-derived" in prov_data["drainage_classification"].lower()
