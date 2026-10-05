"""Analytics package init."""

from backend.app.services.analytics.depth import (
    DepthEstimationResult,
    estimate_flood_depth_boundary,
)
from backend.app.services.analytics.exposure import (
    OSM_COMPLETENESS_NOTE,
    WORLDCOVER_CLASSES,
    compute_landcover_exposure,
    compute_osm_exposure,
)
from backend.app.services.analytics.rainfall import (
    CSVRainfallProvider,
    OpenMeteoRainfallProvider,
    RainfallProvider,
)
from backend.app.services.analytics.susceptibility import compute_iou, score_susceptibility
from backend.app.services.analytics.timeline import TimelineItem, build_event_timeline

__all__ = [
    "DepthEstimationResult",
    "estimate_flood_depth_boundary",
    "OSM_COMPLETENESS_NOTE",
    "WORLDCOVER_CLASSES",
    "compute_landcover_exposure",
    "compute_osm_exposure",
    "RainfallProvider",
    "OpenMeteoRainfallProvider",
    "CSVRainfallProvider",
    "compute_iou",
    "score_susceptibility",
    "TimelineItem",
    "build_event_timeline",
]
