"""Common SAR Scene representation and readers for Sentinel-1 and NISAR.

Provides a unified SAR `Scene` object with backscatter calibrated to dB,
affine geotransform, CRS, and standardized metadata.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine, from_origin

logger = logging.getLogger(__name__)


@dataclass
class Scene:
    """Standardized representation of an acquired SAR scene.

    Attributes:
        scene_id: Granule / scene identifier string.
        sensor: 'sentinel1' or 'nisar'.
        product_level: e.g. 'GRD_HD', 'RTC', 'GCOV'.
        acquisition_time: UTC datetime of the acquisition.
        relative_orbit: Relative orbit / track number.
        polarization: Polarization channel of the array, e.g. 'VV', 'VH', 'HH', 'HV'.
        data_db: 2D numpy float32 array of backscatter in decibels (dB).
        transform: Affine transformation mapping pixel coordinates to spatial CRS coordinates.
        crs: Coordinate reference system (e.g. EPSG:4326, EPSG:32632).
        nodata: Value used for no-data / unobserved pixels (typically np.nan or -9999.0).
        metadata: Additional original metadata dictionary.
    """
    scene_id: str
    sensor: str
    product_level: str
    acquisition_time: Optional[datetime]
    relative_orbit: Optional[int]
    polarization: str
    data_db: np.ndarray
    transform: Affine
    crs: CRS
    nodata: float = -9999.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> Tuple[int, int]:
        """Return (height, width) of the SAR backscatter grid."""
        return self.data_db.shape

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Return (minx, miny, maxx, maxy) in CRS coordinates."""
        h, w = self.data_db.shape
        top_left_x, top_left_y = self.transform * (0, 0)
        bot_right_x, bot_right_y = self.transform * (w, h)
        minx = min(top_left_x, bot_right_x)
        maxx = max(top_left_x, bot_right_x)
        miny = min(top_left_y, bot_right_y)
        maxy = max(top_left_y, bot_right_y)
        return (minx, miny, maxx, maxy)


def linear_to_db(array: np.ndarray, nodata_mask: Optional[np.ndarray] = None, eps: float = 1e-7) -> np.ndarray:
    """Safely converts linear power backscatter (sigma0 / gamma0) to decibels (dB).

    Formula: 10 * log10(max(val, eps))
    Zero / negative values or invalid pixels become NaN.
    """
    out = np.full(array.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(array) & (array > 0)
    if nodata_mask is not None:
        valid = valid & (~nodata_mask)
    out[valid] = 10.0 * np.log10(np.maximum(array[valid], eps))
    return out


class Sentinel1Reader:
    """Reader for Sentinel-1 GRD or RTC GeoTIFF files."""

    @staticmethod
    def read(
        filepath: Union[str, Path],
        band: int = 1,
        polarization: Optional[str] = None,
        is_linear: Optional[bool] = None,
        scene_id: Optional[str] = None,
        relative_orbit: Optional[int] = None,
        acquisition_time: Optional[datetime] = None,
    ) -> Scene:
        """Reads a Sentinel-1 GeoTIFF and returns a common `Scene`.

        If `is_linear` is None, it detects whether values are linear (>= 0 and max > 0, typical mean < 1)
        or already in dB (negative numbers present).
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        with rasterio.open(path) as src:
            arr = src.read(band).astype(np.float32)
            transform = src.transform
            crs = src.crs
            src_nodata = src.nodata
            tags = src.tags()
            tags_item = src.tags(band)
            meta = {**tags, **tags_item}

        # Mask nodata
        nodata_mask = np.zeros(arr.shape, dtype=bool)
        if src_nodata is not None:
            nodata_mask |= np.isclose(arr, src_nodata)
        nodata_mask |= np.isnan(arr)

        # Infer if linear or dB
        valid_vals = arr[~nodata_mask]
        if is_linear is None:
            if len(valid_vals) > 0 and np.all(valid_vals >= 0):
                # Typically linear radar power is strictly positive
                is_linear = True
            else:
                is_linear = False

        if is_linear:
            data_db = linear_to_db(arr, nodata_mask=nodata_mask)
        else:
            data_db = arr.copy()
            data_db[nodata_mask] = np.nan

        # Determine polarization
        pol = polarization
        if not pol:
            # Check filename or tags
            fname = path.stem.upper()
            for p in ["VV", "VH", "HH", "HV"]:
                if f"_{p}" in fname or f"-{p}" in fname or fname.endswith(p):
                    pol = p
                    break
        if not pol:
            pol = meta.get("POLARIZATION", "VV")

        # Determine ID
        sid = scene_id or path.stem

        return Scene(
            scene_id=sid,
            sensor="sentinel1",
            product_level="RTC" if "RTC" in sid.upper() else "GRD",
            acquisition_time=acquisition_time,
            relative_orbit=relative_orbit,
            polarization=pol,
            data_db=data_db,
            transform=transform,
            crs=crs,
            nodata=np.nan,
            metadata=meta,
        )


class NisarReader:
    """Reader for NISAR HDF5 GCOV products based strictly on NISAR Product Specification:

    HDF5 Hierarchy:
      /science/LSAR/GCOV/grids/frequency[A|B]/
        - Covariance datasets: HHHH, HVHV, VVVV, etc. (real 2D float arrays, power gamma0)
        - xCoordinates (1D array)
        - yCoordinates (1D array)
        - projection (dataset with EPSG code or WKT attributes)
      /science/LSAR/identification/
        - trackNumber / relativeOrbit
        - zeroDopplerStartTime / acquisition datetime
    """

    @staticmethod
    def _resolve_frequency_group(h5_file: h5py.File, frequency: str = "A") -> str:
        """Resolves the frequency group path for GCOV."""
        grids_path = "science/LSAR/GCOV/grids"
        if grids_path not in h5_file:
            # Try searching for grids group
            for key in ["science/LSAR/GCOV/grids", "science/LSAR/grids"]:
                if key in h5_file:
                    grids_path = key
                    break
            else:
                raise KeyError(f"NISAR GCOV grids group not found. Expected: science/LSAR/GCOV/grids")

        freq_name = f"frequency{frequency.upper()}"
        freq_path = f"{grids_path}/{freq_name}"
        if freq_path in h5_file:
            return freq_path
        
        # Fall back to first available frequency group under grids
        available = list(h5_file[grids_path].keys())
        if not available:
            raise KeyError(f"No frequency subgroups found under {grids_path}")
        return f"{grids_path}/{available[0]}"

    @classmethod
    def read(
        name_or_path: Any,
        filepath: Union[str, Path],
        polarization: str = "HHHH",
        frequency: str = "A",
        scene_id: Optional[str] = None,
    ) -> Scene:
        """Reads a NISAR GCOV HDF5 file and extracts a common `Scene` in dB."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"NISAR HDF5 file not found: {path}")

        # Normalize target dataset name (e.g. 'HH' -> 'HHHH', 'VV' -> 'VVVV')
        pol_clean = polarization.upper().replace("_", "")
        if len(pol_clean) == 2:
            pol_clean = pol_clean * 2  # HH -> HHHH

        with h5py.File(path, "r") as h5:
            freq_group_path = NisarReader._resolve_frequency_group(h5, frequency=frequency)
            freq_group = h5[freq_group_path]

            # Look for polarization dataset in frequency group
            cov_name = None
            if pol_clean in freq_group:
                cov_name = pol_clean
            else:
                # Search case-insensitively or matching prefix
                for k in freq_group.keys():
                    if k.upper() == pol_clean or k.upper().startswith(pol_clean):
                        cov_name = k
                        break
            
            if cov_name is None:
                # If diagonal term requested but not found, pick first available 2D dataset
                candidates = [
                    k for k in freq_group.keys()
                    if isinstance(freq_group[k], h5py.Dataset) and freq_group[k].ndim == 2
                ]
                if not candidates:
                    raise KeyError(
                        f"Requested covariance dataset '{pol_clean}' not found in {freq_group_path}. "
                        f"Available: {list(freq_group.keys())}"
                    )
                cov_name = candidates[0]
                logger.warning(f"Polarization '{pol_clean}' not found. Using '{cov_name}'")

            ds = freq_group[cov_name]
            raw_data = ds[:].astype(np.float32)

            # Check nodata or fill value attribute
            nodata_val = ds.attrs.get("_FillValue", None)

            # Read coordinates
            if "xCoordinates" in freq_group and "yCoordinates" in freq_group:
                x_coords = freq_group["xCoordinates"][:]
                y_coords = freq_group["yCoordinates"][:]
                dx = float(x_coords[1] - x_coords[0]) if len(x_coords) > 1 else 10.0
                dy = float(y_coords[1] - y_coords[0]) if len(y_coords) > 1 else -10.0
                # Top-left is (min(x) - dx/2, max(y) - dy/2)
                origin_x = float(x_coords[0] - dx / 2.0)
                origin_y = float(y_coords[0] - dy / 2.0)
                transform = Affine(dx, 0.0, origin_x, 0.0, dy, origin_y)
            else:
                # Default fallback transform
                transform = Affine.identity()

            # Read projection / CRS
            crs = None
            if "projection" in freq_group:
                proj_ds = freq_group["projection"]
                epsg = proj_ds.attrs.get("epsg_code", None)
                if epsg is not None:
                    crs = CRS.from_epsg(int(epsg))
                elif "spatial_ref" in proj_ds.attrs:
                    crs = CRS.from_wkt(proj_ds.attrs["spatial_ref"])
            
            if crs is None:
                # Default to WGS 84 if not specified
                crs = CRS.from_epsg(4326)

            # Extract identification metadata (orbit, datetime)
            ident_path = "science/LSAR/identification"
            acq_time = None
            orbit = None
            metadata = {}
            if ident_path in h5:
                ident = h5[ident_path]
                for k, v in ident.attrs.items():
                    metadata[k] = v
                if "trackNumber" in ident:
                    orbit = int(ident["trackNumber"][()])
                elif "relativeOrbitNumber" in ident:
                    orbit = int(ident["relativeOrbitNumber"][()])
                
                for time_key in ["zeroDopplerStartTime", "acquisitionStartTime", "startTime"]:
                    if time_key in ident:
                        t_val = ident[time_key][()]
                        if isinstance(t_val, bytes):
                            t_val = t_val.decode("utf-8")
                        try:
                            acq_time = datetime.fromisoformat(t_val.replace("Z", "+00:00"))
                        except Exception:
                            pass
                        break

        # Convert linear backscatter power to dB
        nodata_mask = np.isnan(raw_data)
        if nodata_val is not None:
            nodata_mask |= np.isclose(raw_data, nodata_val)
        data_db = linear_to_db(raw_data, nodata_mask=nodata_mask)

        # Standardize 2-letter polarization name for Scene (e.g. HHHH -> HH)
        pol_code = cov_name[:2] if len(cov_name) >= 2 else cov_name

        return Scene(
            scene_id=scene_id or path.stem,
            sensor="nisar",
            product_level="GCOV",
            acquisition_time=acq_time,
            relative_orbit=orbit,
            polarization=pol_code,
            data_db=data_db,
            transform=transform,
            crs=crs,
            nodata=np.nan,
            metadata=metadata,
        )


def read_scene(filepath: Union[str, Path], sensor: Optional[str] = None, **kwargs) -> Scene:
    """Helper that automatically detects format (GeoTIFF vs HDF5) and reads the Scene."""
    path = Path(filepath)
    ext = path.suffix.lower()

    if sensor == "nisar" or ext in [".h5", ".hdf5", ".nc"]:
        return NisarReader.read(path, **kwargs)
    elif sensor == "sentinel1" or ext in [".tif", ".tiff", ".vrt"]:
        return Sentinel1Reader.read(path, **kwargs)
    else:
        # Check by content or extension
        if ext in [".h5", ".hdf5"]:
            return NisarReader.read(path, **kwargs)
        return Sentinel1Reader.read(path, **kwargs)
