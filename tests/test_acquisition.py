"""Tests for EventConfig, SAR readers (NISAR HDF5 GCOV and Sentinel-1 GeoTIFF), and fetch_scenes."""

from __future__ import annotations

import tempfile
from datetime import date, datetime
from pathlib import Path

import h5py
import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine, from_origin
from shapely.geometry import Polygon

from src.config import (
    DateRange,
    EventConfig,
    PermanentWaterConfig,
    PermanentWaterSource,
    SensorType,
    ThresholdConfig,
)
from src.readers import NisarReader, Sentinel1Reader, linear_to_db, read_scene


# ---------------------------------------------------------------------------
# Fixtures: Synthetic HDF5 NISAR GCOV and GeoTIFF Sentinel-1
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_nisar_gcov(tmp_path: Path) -> Path:
    """Creates a synthetic NISAR HDF5 GCOV file conforming to the official product spec."""
    file_path = tmp_path / "NISAR_L2_PR_GCOV_001_042_A_020_0000_HHHH_20240501T120000.h5"

    with h5py.File(file_path, "w") as h5:
        # Create standard group hierarchy
        gcov_grid = h5.create_group("science/LSAR/GCOV/grids/frequencyA")

        # 50x50 synthetic radar backscatter matrix (power gamma0, positive floats with specular water region)
        data = np.full((50, 50), 0.1, dtype=np.float32)  # -10 dB background
        data[15:35, 15:35] = 0.001  # -30 dB water body
        data[0, 0] = -9999.0  # nodata pixel

        ds = gcov_grid.create_dataset("HHHH", data=data, compression="gzip")
        ds.attrs["_FillValue"] = -9999.0
        ds.attrs["description"] = "Gamma0 terrain-corrected backscatter HHHH"

        # Coordinates
        dx, dy = 10.0, -10.0
        x_coords = 500000.0 + np.arange(50) * dx
        y_coords = 4000000.0 + np.arange(50) * dy
        gcov_grid.create_dataset("xCoordinates", data=x_coords)
        gcov_grid.create_dataset("yCoordinates", data=y_coords)

        # Projection metadata
        proj = gcov_grid.create_dataset("projection", data=np.array([0], dtype=np.int32))
        proj.attrs["epsg_code"] = 32632
        proj.attrs["spatial_ref"] = "EPSG:32632"

        # Identification metadata group
        ident = h5.create_group("science/LSAR/identification")
        ident.create_dataset("trackNumber", data=np.int32(42))
        ident.create_dataset("zeroDopplerStartTime", data=np.bytes_("2024-05-01T12:00:00Z"))
        ident.attrs["mission"] = "NISAR"

    return file_path


@pytest.fixture
def synthetic_sentinel1_geotiff(tmp_path: Path) -> Path:
    """Creates a synthetic Sentinel-1 GRD GeoTIFF file."""
    file_path = tmp_path / "S1A_IW_GRDH_1SDV_20230518T053000_048500_05D5E1_VV.tif"

    # 40x40 synthetic linear power image
    data = np.full((40, 40), 0.08, dtype=np.float32)  # ~ -11 dB
    data[10:25, 10:25] = 0.002  # ~ -27 dB water
    data[0, 0] = -9999.0  # nodata

    transform = from_origin(11.85, 44.55, 0.0001, 0.0001)
    crs = CRS.from_epsg(4326)

    with rasterio.open(
        file_path,
        "w",
        driver="GTiff",
        height=40,
        width=40,
        count=1,
        dtype=np.float32,
        crs=crs,
        transform=transform,
        nodata=-9999.0,
    ) as dst:
        dst.write(data, 1)
        dst.update_tags(POLARIZATION="VV", SENSOR="C-SAR")

    return file_path


# ---------------------------------------------------------------------------
# Test 1: EventConfig (Pydantic + YAML parsing)
# ---------------------------------------------------------------------------

def test_event_config_validation():
    cfg = EventConfig(
        id="test_flood_event",
        name="Test Flood Event 2024",
        aoi=[[12.0, 44.0], [13.0, 44.0], [13.0, 45.0], [12.0, 45.0], [12.0, 44.0]],
        event_date_range=DateRange(start=date(2024, 5, 1), end=date(2024, 5, 10)),
        baseline_date_range=DateRange(start=date(2024, 4, 1), end=date(2024, 4, 20)),
        sensor=SensorType.SENTINEL1,
        polarization="VV",
        thresholds=ThresholdConfig(water_threshold_db=-17.0),
        permanent_water=PermanentWaterConfig(source=PermanentWaterSource.JRC_OCCURRENCE),
    )

    assert cfg.id == "test_flood_event"
    assert cfg.sensor == SensorType.SENTINEL1
    poly = cfg.get_aoi_polygon()
    assert isinstance(poly, Polygon)
    assert "POLYGON" in cfg.get_aoi_wkt()
    assert cfg.thresholds.water_threshold_db == -17.0
    assert cfg.permanent_water.source == PermanentWaterSource.JRC_OCCURRENCE


def test_event_config_yaml_roundtrip(tmp_path: Path):
    yaml_path = tmp_path / "event.yaml"
    content = """
id: "yaml_event_1"
name: "YAML Test Flood"
aoi:
  type: "Polygon"
  coordinates:
    - [[10.0, 50.0], [11.0, 50.0], [11.0, 51.0], [10.0, 51.0], [10.0, 50.0]]
event_date_range:
  start: "2024-06-01"
  end: "2024-06-05"
baseline_date_range:
  start: "2024-05-01"
  end: "2024-05-15"
dem_path: "dem.tif"
sensor: "nisar"
polarization: "HH"
thresholds:
  water_threshold_db: -18.5
permanent_water:
  source: "esa_worldcover"
"""
    yaml_path.write_text(content, encoding="utf-8")
    cfg = EventConfig.from_yaml(yaml_path)

    assert cfg.id == "yaml_event_1"
    assert cfg.sensor == SensorType.NISAR
    assert cfg.polarization == "HH"
    assert cfg.thresholds.water_threshold_db == -18.5
    assert cfg.dem_path == "dem.tif"

    # Test serialization to yaml
    serialized = cfg.to_yaml()
    assert "yaml_event_1" in serialized
    assert "esa_worldcover" in serialized


def test_invalid_date_range():
    with pytest.raises(ValueError):
        DateRange(start=date(2024, 6, 10), end=date(2024, 6, 1))


# ---------------------------------------------------------------------------
# Test 2: Linear to dB conversion
# ---------------------------------------------------------------------------

def test_linear_to_db():
    arr = np.array([1.0, 0.1, 0.01, -1.0], dtype=np.float32)
    db = linear_to_db(arr)

    assert np.isclose(db[0], 0.0)      # 10*log10(1) = 0 dB
    assert np.isclose(db[1], -10.0)    # 10*log10(0.1) = -10 dB
    assert np.isclose(db[2], -20.0)    # 10*log10(0.01) = -20 dB
    assert np.isnan(db[3])             # negative value -> NaN


# ---------------------------------------------------------------------------
# Test 3: NISAR HDF5 GCOV Reader
# ---------------------------------------------------------------------------

def test_nisar_gcov_reader(synthetic_nisar_gcov: Path):
    scene = NisarReader.read(synthetic_nisar_gcov, polarization="HHHH")

    assert scene.sensor == "nisar"
    assert scene.product_level == "GCOV"
    assert scene.polarization == "HH"
    assert scene.relative_orbit == 42
    assert scene.crs.to_epsg() == 32632
    assert scene.shape == (50, 50)
    assert scene.data_db.dtype == np.float32

    # Check backscatter dB conversion:
    # 0.1 -> -10 dB, 0.001 -> -30 dB, nodata (-9999) -> NaN
    assert np.isnan(scene.data_db[0, 0])
    assert np.isclose(scene.data_db[2, 2], -10.0, atol=0.1)
    assert np.isclose(scene.data_db[20, 20], -30.0, atol=0.1)

    # Check affine transform resolution
    assert scene.transform.a == 10.0
    assert scene.transform.e == -10.0


# ---------------------------------------------------------------------------
# Test 4: Sentinel-1 GeoTIFF Reader
# ---------------------------------------------------------------------------

def test_sentinel1_reader(synthetic_sentinel1_geotiff: Path):
    scene = Sentinel1Reader.read(synthetic_sentinel1_geotiff, relative_orbit=95)

    assert scene.sensor == "sentinel1"
    assert scene.product_level == "GRD"
    assert scene.polarization == "VV"
    assert scene.relative_orbit == 95
    assert scene.crs.to_epsg() == 4326
    assert scene.shape == (40, 40)

    # Nodata converted to NaN
    assert np.isnan(scene.data_db[0, 0])
    # 0.08 -> ~ -10.97 dB
    assert np.isclose(scene.data_db[2, 2], 10 * np.log10(0.08), atol=0.1)
    # 0.002 -> ~ -26.99 dB
    assert np.isclose(scene.data_db[15, 15], 10 * np.log10(0.002), atol=0.1)


# ---------------------------------------------------------------------------
# Test 5: Universal read_scene dispatcher
# ---------------------------------------------------------------------------

def test_read_scene_dispatch(synthetic_nisar_gcov: Path, synthetic_sentinel1_geotiff: Path):
    s_nisar = read_scene(synthetic_nisar_gcov)
    assert s_nisar.sensor == "nisar"
    assert s_nisar.product_level == "GCOV"

    s_s1 = read_scene(synthetic_sentinel1_geotiff)
    assert s_s1.sensor == "sentinel1"
    assert s_s1.product_level == "GRD"
