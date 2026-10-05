"""FastAPI application with endpoints:
- GET /api/events/{id}/timeline
- GET /api/events/{id}/summary
Never fabricates values.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.app.services.analytics.rainfall import OpenMeteoRainfallProvider
from backend.app.services.analytics.timeline import TimelineItem, build_event_timeline

app = FastAPI(
    title="Flood Analytics API",
    description="Flood Event Reconstruction & Replay Analytics Services",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for processed events and scenes
# Key: event_id -> Dict with 'scenes', 'dem', 'rainfall', etc.
EVENT_STORE: Dict[str, Dict[str, Any]] = {}


def register_event_data(event_id: str, data: Dict[str, Any]) -> None:
    """Helper to register processed event data into the API store."""
    EVENT_STORE[event_id] = data


@app.get("/api/events/{event_id}/timeline", response_model=List[Dict[str, Any]])
def get_event_timeline(
    event_id: str,
    include_rainfall: bool = Query(default=False, description="Include daily rainfall time series."),
) -> List[Dict[str, Any]]:
    """Returns chronological timeline for an event:
    - per acquisition date: flood mask URL, flooded area (km2), valid fraction, sensor, orbit,
      acquisition time, gap in days since previous scene, exposure (OSM roads, buildings, WorldCover),
      and rough depth estimate.
    """
    if event_id not in EVENT_STORE:
        raise HTTPException(status_code=404, detail=f"Event '{event_id}' not found.")

    event_data = EVENT_STORE[event_id]
    scenes = event_data.get("scenes", [])
    if not scenes:
        return []

    timeline_items, _ = build_event_timeline(
        scene_results=scenes,
        dem=event_data.get("dem"),
        worldcover_raster=event_data.get("worldcover"),
        road_raster=event_data.get("roads"),
        building_raster=event_data.get("buildings"),
        pixel_size_m=event_data.get("pixel_size_m", 10.0),
        pixel_area_km2=event_data.get("pixel_area_km2", 0.0001),
    )

    out = []
    for item in timeline_items:
        it_dict = {
            "date": item.date,
            "acquisition_time": item.acquisition_time,
            "flood_mask_url": item.flood_mask_url,
            "flooded_area_km2": item.flooded_area_km2,
            "flooded_pixel_count": item.flooded_pixel_count,
            "valid_fraction": item.valid_fraction,
            "sensor": item.sensor,
            "relative_orbit": item.relative_orbit,
            "polarization": item.polarization,
            "gap_days_since_previous": item.gap_days_since_previous,
            "exposure": item.exposure,
            "depth_estimate": item.depth_estimate,
        }
        out.append(it_dict)

    return out


@app.get("/api/events/{event_id}/summary")
def get_event_summary(event_id: str) -> Dict[str, Any]:
    """Returns comprehensive event summary:
    - Peak-extent date among available scenes (with caveat that it is of available scenes, not true peak).
    - Aggregate statistics across scenes.
    - Susceptibility comparison scores (AUC, IoU) if terrain layer present.
    - Optional rainfall summary.
    """
    if event_id not in EVENT_STORE:
        raise HTTPException(status_code=404, detail=f"Event '{event_id}' not found.")

    event_data = EVENT_STORE[event_id]
    scenes = event_data.get("scenes", [])
    if not scenes:
        raise HTTPException(status_code=400, detail=f"Event '{event_id}' has no processed scenes.")

    _, summary = build_event_timeline(
        scene_results=scenes,
        dem=event_data.get("dem"),
        worldcover_raster=event_data.get("worldcover"),
        road_raster=event_data.get("roads"),
        building_raster=event_data.get("buildings"),
    )

    # Attach susceptibility evaluation if terrain model scored
    if "susceptibility_scores" in event_data:
        summary["susceptibility_evaluation"] = event_data["susceptibility_scores"]

    # Attach rainfall series if available
    if "rainfall" in event_data:
        summary["rainfall_series"] = event_data["rainfall"]

    return summary
