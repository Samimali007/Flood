"""Event timeline and summary analytics orchestrator."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from backend.app.services.analytics.depth import (
    DepthEstimationResult,
    estimate_flood_depth_boundary,
)
from backend.app.services.analytics.exposure import (
    compute_landcover_exposure,
    compute_osm_exposure,
)
from backend.app.services.analytics.rainfall import RainfallProvider
from backend.app.services.analytics.susceptibility import score_susceptibility


@dataclass
class TimelineItem:
    date: str
    acquisition_time: str
    flood_mask_url: str
    flooded_area_km2: float
    flooded_pixel_count: int
    valid_fraction: float
    sensor: str
    relative_orbit: Optional[int]
    polarization: str
    gap_days_since_previous: Optional[float]
    exposure: Dict[str, Any]
    depth_estimate: Optional[Dict[str, Any]]


def build_event_timeline(
    scene_results: List[Dict[str, Any]],
    dem: Optional[np.ndarray] = None,
    worldcover_raster: Optional[np.ndarray] = None,
    road_raster: Optional[np.ndarray] = None,
    building_raster: Optional[np.ndarray] = None,
    pixel_size_m: float = 10.0,
    pixel_area_km2: float = 0.0001,
) -> Tuple[List[TimelineItem], Dict[str, Any]]:
    """Builds the complete temporal timeline and summary analytics.

    Returns:
      (timeline_items, summary_dict)
    """
    if not scene_results:
        return [], {}

    # Sort chronologically by acquisition date
    sorted_scenes = sorted(
        scene_results,
        key=lambda x: x.get("acquisition_time") or x.get("date") or "",
    )

    timeline_items: List[TimelineItem] = []
    prev_dt: Optional[datetime] = None

    for idx, sc in enumerate(sorted_scenes):
        acq_str = sc.get("acquisition_time") or sc.get("date")
        dt_val = None
        try:
            dt_val = datetime.fromisoformat(acq_str.replace("Z", "+00:00"))
        except Exception:
            pass

        gap_days = None
        if dt_val and prev_dt:
            gap_days = round((dt_val - prev_dt).total_seconds() / 86400.0, 2)
        if dt_val:
            prev_dt = dt_val

        # Exposure calculation if rasters provided
        mask_arr = sc.get("mask_array")
        exposure_data: Dict[str, Any] = {}
        depth_data: Optional[Dict[str, Any]] = None

        if mask_arr is not None:
            # OSM
            osm_res = compute_osm_exposure(
                mask_arr,
                road_raster_or_lines=road_raster,
                building_raster_or_polygons=building_raster,
                pixel_size_m=pixel_size_m,
            )
            # Landcover
            lc_res = {}
            if worldcover_raster is not None:
                lc_res = compute_landcover_exposure(mask_arr, worldcover_raster, pixel_area_km2)

            exposure_data = {
                **osm_res,
                "landcover_affected_km2": lc_res,
            }

            # Rough depth
            if dem is not None:
                depth_res = estimate_flood_depth_boundary(
                    flood_mask=mask_arr,
                    dem=dem,
                    pixel_area_m2=pixel_size_m**2,
                )
                depth_data = {
                    "status": depth_res.status,
                    "mean_depth_m": depth_res.mean_depth_m,
                    "max_depth_m": depth_res.max_depth_m,
                    "volume_m3": depth_res.volume_m3,
                    "label": depth_res.label,
                    "documentation": depth_res.method_documentation,
                    "uncertainty_indicators": depth_res.uncertainty_indicators,
                }

        item = TimelineItem(
            date=acq_str.split("T")[0] if "T" in acq_str else acq_str,
            acquisition_time=acq_str,
            flood_mask_url=sc.get("cog_url") or sc.get("cog_path") or f"/api/events/{sc.get('event_id', 'event')}/masks/{sc.get('scene_id')}.tif",
            flooded_area_km2=round(float(sc.get("flooded_area_km2", 0.0)), 4),
            flooded_pixel_count=int(sc.get("flooded_pixel_count", 0)),
            valid_fraction=round(float(sc.get("valid_pixel_fraction", 1.0)), 4),
            sensor=sc.get("sensor", "unknown"),
            relative_orbit=sc.get("relative_orbit"),
            polarization=sc.get("polarization", "VV"),
            gap_days_since_previous=gap_days,
            exposure=exposure_data,
            depth_estimate=depth_data,
        )
        timeline_items.append(item)

    # 2. Peak-extent date calculation
    # MUST STATE: It is the peak OF THE AVAILABLE SCENES, not necessarily the true peak.
    peak_scene = max(timeline_items, key=lambda it: it.flooded_area_km2)
    peak_summary = {
        "date": peak_scene.date,
        "acquisition_time": peak_scene.acquisition_time,
        "flooded_area_km2": peak_scene.flooded_area_km2,
        "sensor": peak_scene.sensor,
        "relative_orbit": peak_scene.relative_orbit,
        "caveat": "Peak extent represents the maximum observed flood area OF THE AVAILABLE SATELLITE SCENES. Because satellite revisits are discrete (every several days), the true peak may have occurred between acquisitions.",
    }

    summary = {
        "event_id": sorted_scenes[0].get("event_id", "unknown") if sorted_scenes else None,
        "total_scenes_processed": len(timeline_items),
        "start_date": timeline_items[0].date if timeline_items else None,
        "end_date": timeline_items[-1].date if timeline_items else None,
        "peak_extent": peak_summary,
        "all_scenes_peak_area_km2": peak_scene.flooded_area_km2,
        "sensors_used": sorted(list(set(it.sensor for it in timeline_items))),
        "orbits_used": sorted(list(set(it.relative_orbit for it in timeline_items if it.relative_orbit is not None))),
    }

    return timeline_items, summary
