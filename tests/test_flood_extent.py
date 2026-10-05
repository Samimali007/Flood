"""Comprehensive tests for flood extent service using synthetic dB rasters with known flood areas."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine, from_origin

from backend.app.services.flood_extent import (
    DetectionMethod,
    FilterConfig,
    FloodDetectionConfig,
    FloodExtentService,
    KNOWN_LIMITATIONS,
    SpeckleFilterType,
    ThresholdMode,
    align_scenes_to_reference,
    apply_speckle_filter,
    check_orbit_consistency,
    clean_flood_mask,
    coregister_scene_to_grid,
    create_baseline_composite,
    generate_dem_slope_mask,
    lee_filter,
)
from src.readers import Scene


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_scenes(tmp_path: Path):
    """Creates a pair of synthetic scenes (baseline and event) with known flooded areas in dB."""
    # 100 x 100 grid at 10m resolution (UTM EPSG:32632)
    # Area per pixel = 10m * 10m = 100 m² = 0.0001 km²
    # Total grid = 1 km²
    transform = from_origin(500000.0, 4500000.0, 10.0, 10.0)
    crs = CRS.from_epsg(32632)

    # 1. Baseline scene: Dry land is ~ -11 dB. Permanent water lake at [10:30, 10:30] = 20x20 = 400 px (~ -22 dB)
    base_arr = np.full((100, 100), -11.0, dtype=np.float32)
    base_arr[10:30, 10:30] = -22.0  # Permanent lake

    # Add minor speckle noise
    rng = np.random.default_rng(42)
    base_arr += rng.normal(0, 0.5, size=(100, 100)).astype(np.float32)

    baseline_scene = Scene(
        scene_id="S1_BASELINE_20230501",
        sensor="sentinel1",
        product_level="GRD",
        acquisition_time=datetime(2023, 5, 1, 6, 0),
        relative_orbit=95,
        polarization="VV",
        data_db=base_arr,
        transform=transform,
        crs=crs,
    )

    # 2. Event scene: Land is still ~ -11 dB. Permanent lake is -22 dB.
    # New flood inundation at [50:80, 50:80] = 30x30 = 900 pixels (~ -22 dB)
    # Total water = 400 (lake) + 900 (new flood) = 1300 pixels
    # Expected newly flooded area = 900 px * 0.0001 km² = 0.09 km²
    event_arr = base_arr.copy()
    event_arr[50:80, 50:80] = -22.0 + rng.normal(0, 0.5, size=(30, 30)).astype(np.float32)

    # Add a few tiny 1-2 pixel isolated noise speckles
    event_arr[5, 5] = -23.0
    event_arr[85, 10] = -23.0
    # Add a 1-pixel hole inside the new flood
    event_arr[65, 65] = -11.0

    event_scene = Scene(
        scene_id="S1_EVENT_20230518",
        sensor="sentinel1",
        product_level="GRD",
        acquisition_time=datetime(2023, 5, 18, 6, 0),
        relative_orbit=95,
        polarization="VV",
        data_db=event_arr,
        transform=transform,
        crs=crs,
    )

    return baseline_scene, event_scene


@pytest.fixture
def synthetic_dem(tmp_path: Path) -> Path:
    """Creates a synthetic DEM with a steep hill in the top-right."""
    dem_path = tmp_path / "dem.tif"
    dem_arr = np.zeros((100, 100), dtype=np.float32)
    # Hill in top-right [0:30, 70:100] with steep slope > 20 degrees
    # 10m pixels: elevation rise of 100m over 20 pixels is slope > 25 degrees
    for r in range(30):
        for c in range(70, 100):
            dem_arr[r, c] = float((30 - r) * 15.0)

    transform = from_origin(500000.0, 4500000.0, 10.0, 10.0)
    crs = CRS.from_epsg(32632)

    with rasterio.open(
        dem_path,
        "w",
        driver="GTiff",
        height=100,
        width=100,
        count=1,
        dtype=np.float32,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(dem_arr, 1)

    return dem_path


# ---------------------------------------------------------------------------
# Test 1: Orbit Consistency & Co-Registration
# ---------------------------------------------------------------------------

def test_orbit_consistency_warning(synthetic_scenes, caplog):
    base, event = synthetic_scenes
    # Same orbit (95) -> True
    assert check_orbit_consistency([base, event]) is True

    # Different orbits -> False + warning logged
    diff_scene = Scene(
        scene_id="DIFF_ORBIT",
        sensor="sentinel1",
        product_level="GRD",
        acquisition_time=datetime.now(timezone.utc),
        relative_orbit=44,  # different orbit!
        polarization="VV",
        data_db=base.data_db.copy(),
        transform=base.transform,
        crs=base.crs,
    )
    with caplog.at_level(logging.WARNING):
        is_consistent = check_orbit_consistency([base, diff_scene])
        assert is_consistent is False
        assert "Orbit mismatch detected" in caplog.text


def test_coregistration(synthetic_scenes):
    base, event = synthetic_scenes
    aligned, tr, crs, shape = align_scenes_to_reference([base, event])
    assert len(aligned) == 2
    assert aligned[0].shape == (100, 100)
    assert aligned[1].shape == (100, 100)


# ---------------------------------------------------------------------------
# Test 2: Speckle Filters (Lee and Median)
# ---------------------------------------------------------------------------

def test_speckle_filters():
    noisy = np.full((30, 30), 10.0, dtype=np.float32)
    noisy[10:20, 10:20] = 50.0
    noisy[15, 15] = 200.0  # extreme speckle spike

    filtered_lee = apply_speckle_filter(noisy, SpeckleFilterType.LEE, size=5)
    filtered_med = apply_speckle_filter(noisy, SpeckleFilterType.MEDIAN, size=5)

    assert filtered_lee.shape == (30, 30)
    assert filtered_med.shape == (30, 30)
    # The extreme spike should be substantially smoothed
    assert filtered_lee[15, 15] < 150.0
    assert filtered_med[15, 15] < 100.0


# ---------------------------------------------------------------------------
# Test 3: Morphological Cleanup (Small objects and small holes)
# ---------------------------------------------------------------------------

def test_morphology_cleanup():
    mask = np.zeros((50, 50), dtype=bool)
    # 20x20 flood body = 400 pixels
    mask[15:35, 15:35] = True
    # 2x2 hole inside flood body = 4 pixels
    mask[25:27, 25:27] = False
    # Tiny 2-pixel isolated noise
    mask[2, 2] = True
    mask[2, 3] = True

    cleaned = clean_flood_mask(mask, min_size_pixels=10, hole_size_pixels=10)

    # Noise removed
    assert not cleaned[2, 2]
    assert not cleaned[2, 3]
    # Hole filled
    assert cleaned[25, 25]
    assert cleaned[26, 26]
    # Main body preserved
    assert np.sum(cleaned) == 400


# ---------------------------------------------------------------------------
# Test 4: Flood Extent - Single-Date dB Threshold Method
# ---------------------------------------------------------------------------

def test_flood_extent_threshold_method(synthetic_scenes, tmp_path):
    base, event = synthetic_scenes
    cog_path = tmp_path / "flood_thresh.tif"

    service = FloodExtentService(
        config=FloodDetectionConfig(
            method=DetectionMethod.THRESHOLD,
            threshold_mode=ThresholdMode.MANUAL,
            water_threshold_db=-16.0,
            use_permanent_water_mask=True,  # Excludes the 400 px permanent lake
            min_size_pixels=10,
            hole_size_pixels=10,
        ),
        filter_config=FilterConfig(filter_type=SpeckleFilterType.MEDIAN, window_size=3),
    )

    mask, result = service.process_date(
        event_scene=event,
        baseline_scene_or_composite=base,
        output_cog_path=cog_path,
    )

    # Expected: The permanent lake (400 px) is masked out by permanent water.
    # The new flood (30x30 = 900 px) should be detected.
    # Hole (1 px) is filled. Tiny isolated speckles (1-2 px) are removed.
    # Flooded pixel count should be close to 900 pixels (+/- a few pixels due to noise at boundaries)
    assert 880 <= result.flooded_pixel_count <= 920
    # 900 px * 0.0001 km² = ~ 0.09 km²
    assert np.isclose(result.flooded_area_km2, 0.09, atol=0.005)
    assert result.method == "threshold"
    assert result.threshold_used_db == -16.0
    assert cog_path.exists()

    # Verify COG file
    with rasterio.open(cog_path) as src:
        assert src.count == 1
        assert src.shape == (100, 100)
        assert src.profile["compress"] == "deflate"
        data = src.read(1)
        assert set(np.unique(data)).issubset({0, 1, 255})


# ---------------------------------------------------------------------------
# Test 5: Flood Extent - Change Detection Method vs Baseline Composite
# ---------------------------------------------------------------------------

def test_flood_extent_change_detection_method(synthetic_scenes):
    base, event = synthetic_scenes

    # Create baseline composite (median)
    composite = create_baseline_composite([base])
    assert composite is not None

    service = FloodExtentService(
        config=FloodDetectionConfig(
            method=DetectionMethod.CHANGE,
            change_threshold_db=-4.0,  # Drop of 4 dB or more
            min_size_pixels=10,
            hole_size_pixels=10,
        ),
        filter_config=FilterConfig(filter_type=SpeckleFilterType.LEE, window_size=5),
    )

    mask, result = service.process_date(
        event_scene=event,
        baseline_scene_or_composite=composite,
    )

    # In change detection:
    # Dry land: -11 dB -> -11 dB (diff = 0 dB, not flooded)
    # Permanent lake: -22 dB -> -22 dB (diff = 0 dB, not newly flooded)
    # New flood: -11 dB -> -22 dB (diff = -11 dB < -4 dB threshold -> flooded!)
    assert 880 <= result.flooded_pixel_count <= 920
    assert np.isclose(result.flooded_area_km2, 0.09, atol=0.005)
    assert result.method == "change"


# ---------------------------------------------------------------------------
# Test 6: Steep Slope Masking from DEM
# ---------------------------------------------------------------------------

def test_dem_slope_masking(synthetic_scenes, synthetic_dem):
    base, event = synthetic_scenes

    # Place a fake dark spot on the steep hill in event image (e.g. radar shadow at [10:20, 80:90])
    event.data_db[10:20, 80:90] = -22.0

    # With slope mask enabled:
    service = FloodExtentService(
        config=FloodDetectionConfig(
            method=DetectionMethod.THRESHOLD,
            water_threshold_db=-16.0,
            use_permanent_water_mask=True,
            use_terrain_slope_mask=True,
            max_slope_deg=10.0,
        )
    )

    mask, result = service.process_date(
        event_scene=event,
        baseline_scene_or_composite=base,
        dem_path=synthetic_dem,
    )

    # The steep hill false detection at [10:20, 80:90] should be completely masked out
    assert not np.any(mask[10:20, 80:90])


# ---------------------------------------------------------------------------
# Test 7: Limitations and Provenance Exposure
# ---------------------------------------------------------------------------

def test_limitations_and_provenance_exposure(synthetic_scenes):
    base, event = synthetic_scenes
    service = FloodExtentService()
    _, result = service.process_date(event_scene=event, baseline_scene_or_composite=base)

    # Provenance
    assert result.provenance["scene_id"] == "S1_EVENT_20230518"
    assert result.provenance["relative_orbit"] == 95
    assert result.provenance["sensor"] == "sentinel1"

    # Known limitations must be explicitly exposed in result
    assert len(result.limitations) >= 4
    lim_str = " ".join(result.limitations).lower()
    assert "flooded vegetation" in lim_str
    assert "wind" in lim_str
    assert "shadow" in lim_str
    assert "urban" in lim_str
