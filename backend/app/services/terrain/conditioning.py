"""Hydrological conditioning and terrain preprocessing:
- NoData masking
- DEM spike handling (despiking)
- Priority-flood depression filling / breaching (Planchon & Darboux / Wang & Liu algorithm)
"""

from __future__ import annotations

import heapq
import logging
from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import median_filter

logger = logging.getLogger(__name__)


def handle_dem_spikes(
    dem: np.ndarray,
    min_elev: float = -500.0,
    max_elev: float = 9000.0,
    spike_relative_thresh: float = 150.0,
) -> np.ndarray:
    """Detects and repairs unphysical DEM spikes and negative pits."""
    cleaned = dem.copy().astype(np.float32)
    valid = np.isfinite(cleaned)

    # 1. Global unphysical bounds -> NaN
    out_of_bounds = valid & ((cleaned < min_elev) | (cleaned > max_elev))
    cleaned[out_of_bounds] = np.nan
    valid = np.isfinite(cleaned)

    # 2. Local isolated spike detection via median difference
    filled_med = np.where(valid, cleaned, np.nanmedian(cleaned[valid]) if np.any(valid) else 0.0)
    med = median_filter(filled_med, size=3, mode="reflect")
    diff = np.abs(cleaned - med)

    spike_mask = valid & (diff > spike_relative_thresh)
    cleaned[spike_mask] = med[spike_mask]

    return cleaned


def fill_depressions_priority_flood(
    dem: np.ndarray,
    epsilon: float = 1e-4,
) -> np.ndarray:
    """Condition DEM by filling pits/sinks using the Priority-Flood algorithm.

    Guarantees every non-border pixel can drain toward the edge of the DEM.
    """
    rows, cols = dem.shape
    filled = np.full((rows, cols), np.inf, dtype=np.float32)
    valid = np.isfinite(dem)

    # Priority queue stores (elevation, row, col)
    pq = []

    # Initialize boundary cells as drainage outlets
    for r in range(rows):
        for c in (0, cols - 1):
            if valid[r, c]:
                filled[r, c] = dem[r, c]
                heapq.heappush(pq, (dem[r, c], r, c))
            else:
                filled[r, c] = np.nan

    for c in range(cols):
        for r in (0, rows - 1):
            if valid[r, c] and np.isinf(filled[r, c]):
                filled[r, c] = dem[r, c]
                heapq.heappush(pq, (dem[r, c], r, c))
            elif not valid[r, c]:
                filled[r, c] = np.nan

    # 8 neighbors offsets
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    while pq:
        elev, r, c = heapq.heappop(pq)

        for dr, dc in neighbors:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if valid[nr, nc] and np.isinf(filled[nr, nc]):
                    # Neighbor elevation cannot be lower than current spillway elevation
                    new_elev = max(float(dem[nr, nc]), elev + epsilon)
                    filled[nr, nc] = new_elev
                    heapq.heappush(pq, (new_elev, nr, nc))

    # Restore nodata
    filled[~valid] = np.nan
    return filled
