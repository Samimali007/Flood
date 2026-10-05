"""Grid co-registration and spatial alignment for SAR scenes."""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject

from src.readers import Scene

logger = logging.getLogger(__name__)


class OrbitMismatchWarning(UserWarning):
    """Warning raised when co-registering SAR scenes across different relative orbits."""
    pass


def check_orbit_consistency(scenes: List[Scene]) -> bool:
    """Verifies that all scenes share the same relative orbit / track number."""
    orbits = set(s.relative_orbit for s in scenes if s.relative_orbit is not None)
    if len(orbits) > 1:
        msg = (
            f"Orbit mismatch detected! Scenes originate from multiple relative orbits: {sorted(list(orbits))}. "
            "SAR viewing geometry (look angle, incidence angle, layover, shadow) differs significantly between orbits. "
            "Mixing orbits may cause severe radiometric artifacts and false change detections."
        )
        logger.warning(msg)
        return False
    return True


def coregister_scene_to_grid(
    scene: Scene,
    target_transform: Affine,
    target_crs: CRS,
    target_shape: Tuple[int, int],
    resampling: Resampling = Resampling.bilinear,
) -> np.ndarray:
    """Reprojects and aligns a SAR Scene data_db array onto a reference target grid."""
    height, width = target_shape
    destination = np.full((height, width), np.nan, dtype=np.float32)

    reproject(
        source=scene.data_db,
        destination=destination,
        src_transform=scene.transform,
        src_crs=scene.crs,
        dst_transform=target_transform,
        dst_crs=target_crs,
        resampling=resampling,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )

    return destination


def align_scenes_to_reference(
    scenes: List[Scene],
    reference_idx: int = 0,
    warn_on_mixed_orbits: bool = True,
) -> Tuple[List[np.ndarray], Affine, CRS, Tuple[int, int]]:
    """Co-registers an array of SAR scenes to the grid of a reference scene."""
    if not scenes:
        raise ValueError("Cannot co-register an empty list of scenes.")

    if warn_on_mixed_orbits:
        check_orbit_consistency(scenes)

    ref = scenes[reference_idx]
    ref_transform = ref.transform
    ref_crs = ref.crs
    ref_shape = ref.data_db.shape

    aligned_arrays = []
    for idx, sc in enumerate(scenes):
        if idx == reference_idx:
            aligned_arrays.append(sc.data_db.copy())
        else:
            aligned = coregister_scene_to_grid(
                sc,
                target_transform=ref_transform,
                target_crs=ref_crs,
                target_shape=ref_shape,
            )
            aligned_arrays.append(aligned)

    return aligned_arrays, ref_transform, ref_crs, ref_shape
