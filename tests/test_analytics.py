"""Unit and integration tests for event analytics:
- Timeline construction (gap in days, flood mask URL, area)
- Peak extent date calculation & caveat
- Exposure analytics (OSM road km, building count, WorldCover)
- Rough depth estimation & uncertainty indicators
- Susceptibility comparison (AUC, IoU)
- FastAPI endpoints GET /api/events/{id}/timeline and /api/events/{id}/summary
"""

from __future__ import annotations

import tempfile
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.api import app, register_event_data
from backend.app.services.analytics import (
    CSVRainfallProvider,
    OpenMeteoRainfallProvider,
    compute_iou,
    compute_landcover_exposure,
    compute_osm_exposure,
    estimate_flood_depth_boundary,
    score_susceptibility,
)
from backend.app.services.analytics.timeline import build_event_timeline


# ---------------------------------------------------------------------------
# Test 1: Timeline Construction & Gap in Days
# ---------------------------------------------------------------------------

def test_timeline_gap_and_peak_extent():
    scene_results = [
        {
            "event_id": "test_event",
            "scene_id": "S1_DATE1",
            "acquisition_time": "2023-05-16T06:00:00Z",
            "sensor": "sentinel1",
            "relative_orbit": 95,
            "polarization": "VV",
            "flooded_area_km2": 45.2,
            "flooded_pixel_count": 452000,
            "valid_pixel_fraction": 0.98,
            "cog_url": "/api/events/test_event/masks/S1_DATE1.tif",
        },
        {
            "event_id": "test_event",
            "scene_id": "S1_DATE2",
            "acquisition_time": "2023-05-22T06:00:00Z",
            "sensor": "sentinel1",
            "relative_orbit": 95,
            "polarization": "VV",
            "flooded_area_km2": 112.8,  # Peak of available scenes
            "flooded_pixel_count": 1128000,
            "valid_pixel_fraction": 0.99,
            "cog_url": "/api/events/test_event/masks/S1_DATE2.tif",
        },
        {
            "event_id": "test_event",
            "scene_id": "S1_DATE3",
            "acquisition_time": "2023-05-28T06:00:00Z",
            "sensor": "sentinel1",
            "relative_orbit": 95,
            "polarization": "VV",
            "flooded_area_km2": 68.4,
            "flooded_pixel_count": 684000,
            "valid_pixel_fraction": 0.95,
            "cog_url": "/api/events/test_event/masks/S1_DATE3.tif",
        },
    ]

    timeline, summary = build_event_timeline(scene_results)

    assert len(timeline) == 3
    # First scene has no previous gap
    assert timeline[0].gap_days_since_previous is None
    # 2023-05-16 to 2023-05-22 = exactly 6.0 days
    assert np.isclose(timeline[1].gap_days_since_previous, 6.0)
    # 2023-05-22 to 2023-05-28 = exactly 6.0 days
    assert np.isclose(timeline[2].gap_days_since_previous, 6.0)

    # Peak-extent date must be 2023-05-22
    assert summary["peak_extent"]["date"] == "2023-05-22"
    assert summary["peak_extent"]["flooded_area_km2"] == 112.8

    # Caveat must explicitly state peak OF THE AVAILABLE SCENES
    caveat = summary["peak_extent"]["caveat"]
    assert "AVAILABLE SATELLITE SCENES" in caveat
    assert "between acquisitions" in caveat.lower()


# ---------------------------------------------------------------------------
# Test 2: Exposure Analytics (OSM Roads, Buildings, WorldCover)
# ---------------------------------------------------------------------------

def test_exposure_analytics():
    # 50x50 mask with flood at [10:30, 10:30] = 20x20 = 400 px
    flood_mask = np.zeros((50, 50), dtype=bool)
    flood_mask[10:30, 10:30] = True

    # 1. Road raster: road running vertically at col 15 from row 0 to 50
    # Overlap with flood is rows 10 to 29 (20 pixels)
    # 20 px * 10m / 1000 = 0.20 km
    roads = np.zeros((50, 50), dtype=np.uint8)
    roads[:, 15] = 1

    # 2. Buildings: 3 distinct buildings inside flood, 2 outside
    buildings = np.zeros((50, 50), dtype=np.int32)
    buildings[12, 12] = 101
    buildings[15, 20] = 102
    buildings[25, 25] = 103
    buildings[5, 5] = 104  # outside
    buildings[40, 40] = 105  # outside

    osm_exp = compute_osm_exposure(flood_mask, road_raster_or_lines=roads, building_raster_or_polygons=buildings, pixel_size_m=10.0)

    assert osm_exp["roads_affected_km"] == 0.2
    assert osm_exp["buildings_affected_count"] == 3
    assert "completeness varies" in osm_exp["data_source_notes"].lower()

    # 3. WorldCover classes: 300 px Cropland (40), 100 px Built-up (50)
    worldcover = np.full((50, 50), 10, dtype=np.uint8)  # Tree cover
    worldcover[10:30, 10:30] = 40  # Cropland
    worldcover[10:15, 10:30] = 50  # Built-up (5 rows * 20 cols = 100 px)

    lc_exp = compute_landcover_exposure(flood_mask, worldcover, pixel_area_km2=0.0001)

    assert "Cropland" in lc_exp
    assert "Built-up" in lc_exp
    # 300 px * 0.0001 = 0.03 km2
    assert np.isclose(lc_exp["Cropland"], 0.03)
    # 100 px * 0.0001 = 0.01 km2
    assert np.isclose(lc_exp["Built-up"], 0.01)


# ---------------------------------------------------------------------------
# Test 3: Rough Depth Estimation from Flood Boundary Elevations
# ---------------------------------------------------------------------------

def test_rough_depth_estimation():
    # 50x50 terrain sloping upward from elevation 10m to 30m
    dem = np.zeros((50, 50), dtype=np.float32)
    for r in range(50):
        for c in range(50):
            dem[r, c] = 10.0 + (r * 0.2) + (c * 0.2)

    # Flat water pond filling depression up to 15m water surface elevation
    flood_mask = dem <= 15.0

    res = estimate_flood_depth_boundary(flood_mask, dem, pixel_area_m2=100.0)

    assert res.status == "success"
    assert res.label == "rough estimate"
    assert res.mean_depth_m > 0.0
    assert res.volume_m3 > 0.0
    assert "boundary_elevation_std_m" in res.uncertainty_indicators


def test_depth_estimation_skips_on_poor_dem():
    # DEM with mostly NaNs (poor quality)
    dem_poor = np.full((50, 50), np.nan, dtype=np.float32)
    dem_poor[10, 10] = 12.0
    flood_mask = np.zeros((50, 50), dtype=bool)
    flood_mask[10:20, 10:20] = True

    res = estimate_flood_depth_boundary(flood_mask, dem_poor)
    assert res.status == "skipped"
    assert "Poor DEM quality" in res.uncertainty_indicators["reason"]


# ---------------------------------------------------------------------------
# Test 4: Susceptibility Scoring (AUC and IoU)
# ---------------------------------------------------------------------------

def test_susceptibility_scoring():
    # Ground truth peak mask
    gt_peak = np.zeros((40, 40), dtype=bool)
    gt_peak[10:30, 10:30] = True

    # High quality continuous susceptibility model (values 0 to 1, higher inside flood)
    susceptibility = np.full((40, 40), 0.1, dtype=np.float32)
    susceptibility[10:30, 10:30] = 0.9

    scores = score_susceptibility(susceptibility, gt_peak, threshold=0.5)

    assert scores["auc"] > 0.95
    assert np.isclose(scores["iou"], 1.0)
    assert scores["threshold_used"] == 0.5


# ---------------------------------------------------------------------------
# Test 5: Rainfall Providers
# ---------------------------------------------------------------------------

def test_csv_rainfall_provider(tmp_path: Path):
    csv_file = tmp_path / "rainfall.csv"
    csv_file.write_text(
        "date,precipitation_mm\n"
        "2023-05-15,12.5\n"
        "2023-05-16,45.0\n"
        "2023-05-17,68.2\n"
        "2023-05-18,5.1\n",
        encoding="utf-8",
    )

    provider = CSVRainfallProvider(csv_file)
    records = provider.fetch_daily_rainfall(
        latitude=44.4, longitude=12.0, start_date=date(2023, 5, 16), end_date=date(2023, 5, 17)
    )

    assert len(records) == 2
    assert records[0]["date"] == "2023-05-16"
    assert records[0]["precipitation_mm"] == 45.0
    assert records[1]["precipitation_mm"] == 68.2


# ---------------------------------------------------------------------------
# Test 6: FastAPI Endpoints (/timeline and /summary)
# ---------------------------------------------------------------------------

def test_api_endpoints():
    client = TestClient(app)

    # Register an event with mock data
    event_id = "test_event_api_123"
    register_event_data(
        event_id,
        {
            "scenes": [
                {
                    "event_id": event_id,
                    "scene_id": "S1_20230516",
                    "acquisition_time": "2023-05-16T05:30:00Z",
                    "sensor": "sentinel1",
                    "relative_orbit": 95,
                    "polarization": "VV",
                    "flooded_area_km2": 52.3,
                    "flooded_pixel_count": 523000,
                    "valid_pixel_fraction": 0.99,
                    "cog_url": f"/api/events/{event_id}/masks/S1_20230516.tif",
                },
                {
                    "event_id": event_id,
                    "scene_id": "S1_20230522",
                    "acquisition_time": "2023-05-22T05:30:00Z",
                    "sensor": "sentinel1",
                    "relative_orbit": 95,
                    "polarization": "VV",
                    "flooded_area_km2": 115.0,
                    "flooded_pixel_count": 1150000,
                    "valid_pixel_fraction": 0.98,
                    "cog_url": f"/api/events/{event_id}/masks/S1_20230522.tif",
                },
            ],
            "susceptibility_scores": {"auc": 0.88, "iou": 0.74},
            "rainfall": [
                {"date": "2023-05-16", "precipitation_mm": 50.0},
                {"date": "2023-05-17", "precipitation_mm": 70.0},
            ],
        },
    )

    # 1. Test GET /api/events/{id}/timeline
    resp_timeline = client.get(f"/api/events/{event_id}/timeline")
    assert resp_timeline.status_code == 200
    timeline = resp_timeline.json()
    assert len(timeline) == 2
    assert timeline[0]["date"] == "2023-05-16"
    assert timeline[1]["gap_days_since_previous"] == 6.0
    assert timeline[1]["flooded_area_km2"] == 115.0

    # 2. Test GET /api/events/{id}/summary
    resp_summary = client.get(f"/api/events/{event_id}/summary")
    assert resp_summary.status_code == 200
    summary = resp_summary.json()
    assert summary["peak_extent"]["date"] == "2023-05-22"
    assert summary["peak_extent"]["flooded_area_km2"] == 115.0
    assert "AVAILABLE SATELLITE SCENES" in summary["peak_extent"]["caveat"]
    assert summary["susceptibility_evaluation"]["auc"] == 0.88

    # 3. Test non-existent event 404
    resp_404 = client.get("/api/events/non_existent_id/timeline")
    assert resp_404.status_code == 404
