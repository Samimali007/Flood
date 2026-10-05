"""Hydrological flow routing and terrain analysis:
- D8 Flow Direction (ESRI coding standard: 1, 2, 4, 8, 16, 32, 64, 128)
- Flow Accumulation (topologically sorted upstream pixel counting)
- Stream extraction ("likely drainage pathways (terrain-derived)")
- Drainage basins (watershed delineation)
- HAND (Height Above Nearest Drainage)
- Terrain Slope (degrees)
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Standard ESRI D8 flow directions:
# 32  64  128
# 16   X    1
#  8   4    2
D8_OFFSETS: Dict[int, Tuple[int, int]] = {
    1: (0, 1),      # East
    2: (1, 1),      # South-East
    4: (1, 0),      # South
    8: (1, -1),     # South-West
    16: (0, -1),    # West
    32: (-1, -1),   # North-West
    64: (-1, 0),    # North
    128: (-1, 1),   # North-East
}


def compute_d8_flow_direction(
    dem: np.ndarray,
    dx: float = 10.0,
    dy: float = 10.0,
) -> np.ndarray:
    """Computes D8 flow direction for each pixel in a conditioned DEM.

    Returns uint8 array with ESRI D8 direction codes (0 for pit/outlet/nodata).
    """
    rows, cols = dem.shape
    fdir = np.zeros((rows, cols), dtype=np.uint8)
    valid = np.isfinite(dem)

    dx = abs(dx)
    dy = abs(dy)
    diag_d = np.sqrt(dx**2 + dy**2)

    for r in range(rows):
        for c in range(cols):
            if not valid[r, c]:
                continue

            current_elev = dem[r, c]
            max_slope = 0.0
            best_dir = 0

            for code, (dr, dc) in D8_OFFSETS.items():
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and valid[nr, nc]:
                    dist = diag_d if (dr != 0 and dc != 0) else (dx if dr == 0 else dy)
                    drop = current_elev - dem[nr, nc]
                    slope = drop / dist
                    if slope > max_slope:
                        max_slope = slope
                        best_dir = code

            fdir[r, c] = best_dir

    return fdir


def compute_flow_accumulation(
    fdir: np.ndarray,
) -> np.ndarray:
    """Computes flow accumulation count (number of upstream contributing pixels)."""
    rows, cols = fdir.shape
    acc = np.ones((rows, cols), dtype=np.int32)
    in_degree = np.zeros((rows, cols), dtype=np.int32)

    # 1. Calculate in-degree for topological sort
    for r in range(rows):
        for c in range(cols):
            code = fdir[r, c]
            if code in D8_OFFSETS:
                dr, dc = D8_OFFSETS[code]
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    in_degree[nr, nc] += 1

    # 2. Queue all headwater cells (in-degree == 0)
    queue = deque()
    for r in range(rows):
        for c in range(cols):
            if in_degree[r, c] == 0:
                queue.append((r, c))

    # 3. Process in topological order
    while queue:
        r, c = queue.popleft()
        code = fdir[r, c]
        if code in D8_OFFSETS:
            dr, dc = D8_OFFSETS[code]
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                acc[nr, nc] += acc[r, c]
                in_degree[nr, nc] -= 1
                if in_degree[nr, nc] == 0:
                    queue.append((nr, nc))

    return acc


def extract_streams(
    accumulation: np.ndarray,
    threshold: int = 50,
) -> np.ndarray:
    """Extracts likely drainage pathways (terrain-derived) where accumulation >= threshold."""
    return accumulation >= threshold


def delineate_basins(
    fdir: np.ndarray,
    stream_mask: np.ndarray,
) -> np.ndarray:
    """Delineates drainage sub-basins by tracing down to outlets."""
    rows, cols = fdir.shape
    basin_map = np.zeros((rows, cols), dtype=np.int32)

    # Find outlets: stream pixels that either point out of bounds or point to non-stream/0
    current_basin_id = 1
    stream_points = []
    for r in range(rows):
        for c in range(cols):
            if stream_mask[r, c]:
                stream_points.append((r, c))

    # Map each stream component or outlet
    # For robust delineation, assign a unique basin ID per distinct outlet or stream stretch
    stream_to_basin = {}
    for r, c in stream_points:
        code = fdir[r, c]
        is_outlet = True
        if code in D8_OFFSETS:
            dr, dc = D8_OFFSETS[code]
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and stream_mask[nr, nc]:
                is_outlet = False
        if is_outlet:
            stream_to_basin[(r, c)] = current_basin_id
            current_basin_id += 1

    # If no explicit outlet, label connected stream components
    if not stream_to_basin and stream_points:
        for r, c in stream_points:
            stream_to_basin[(r, c)] = 1

    # Propagate upstream from labeled stream cells
    # First, trace all non-outlet stream cells down to an outlet
    def get_basin_for_stream(sr: int, sc: int, visited=None) -> int:
        if visited is None:
            visited = set()
        if (sr, sc) in stream_to_basin:
            return stream_to_basin[(sr, sc)]
        if (sr, sc) in visited:
            return 1
        visited.add((sr, sc))
        code = fdir[sr, sc]
        if code in D8_OFFSETS:
            dr, dc = D8_OFFSETS[code]
            nr, nc = sr + dr, sc + dc
            if 0 <= nr < rows and 0 <= nc < cols and stream_mask[nr, nc]:
                bid = get_basin_for_stream(nr, nc, visited)
                stream_to_basin[(sr, sc)] = bid
                return bid
        stream_to_basin[(sr, sc)] = 1
        return 1

    for r, c in stream_points:
        basin_map[r, c] = get_basin_for_stream(r, c)

    # Trace each non-stream cell down along flow direction until it hits a stream cell
    def trace_cell_to_basin(cr: int, cc: int, visited=None) -> int:
        if visited is None:
            visited = set()
        if basin_map[cr, cc] > 0:
            return basin_map[cr, cc]
        if (cr, cc) in visited:
            return 0
        visited.add((cr, cc))

        code = fdir[cr, cc]
        if code in D8_OFFSETS:
            dr, dc = D8_OFFSETS[code]
            nr, nc = cr + dr, cc + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                bid = trace_cell_to_basin(nr, nc, visited)
                basin_map[cr, cc] = bid
                return bid
        return 0

    for r in range(rows):
        for c in range(cols):
            if basin_map[r, c] == 0:
                trace_cell_to_basin(r, c)

    return basin_map


def compute_hand(
    dem: np.ndarray,
    fdir: np.ndarray,
    stream_mask: np.ndarray,
) -> np.ndarray:
    """Computes Height Above Nearest Drainage (HAND).

    Follows D8 flow direction from each cell down to the drainage stream network
    and subtracts the stream elevation from the cell elevation:
      HAND = dem[cell] - dem[drainage_stream_cell]
    """
    rows, cols = dem.shape
    hand = np.full((rows, cols), np.nan, dtype=np.float32)
    stream_elev = np.full((rows, cols), np.nan, dtype=np.float32)
    valid = np.isfinite(dem)

    # Base case: Stream cells have stream_elev = dem and HAND = 0
    stream_elev[stream_mask & valid] = dem[stream_mask & valid]
    hand[stream_mask & valid] = 0.0

    def find_stream_elevation(r: int, c: int, visited=None) -> float:
        if visited is None:
            visited = set()
        if not np.isnan(stream_elev[r, c]):
            return float(stream_elev[r, c])
        if (r, c) in visited:
            return float(dem[r, c])
        visited.add((r, c))

        code = fdir[r, c]
        if code in D8_OFFSETS:
            dr, dc = D8_OFFSETS[code]
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and valid[nr, nc]:
                se = find_stream_elevation(nr, nc, visited)
                stream_elev[r, c] = se
                return se

        # Hit boundary or pit without hitting stream
        stream_elev[r, c] = dem[r, c]
        return float(dem[r, c])

    for r in range(rows):
        for c in range(cols):
            if valid[r, c] and np.isnan(hand[r, c]):
                target_se = find_stream_elevation(r, c)
                hand[r, c] = max(0.0, float(dem[r, c] - target_se))

    return hand


def compute_slope(
    dem: np.ndarray,
    dx: float = 10.0,
    dy: float = 10.0,
) -> np.ndarray:
    """Calculates terrain slope in degrees using 2nd order central difference gradient."""
    dx = abs(dx)
    dy = abs(dy)
    gy, gx = np.gradient(dem, dy, dx)
    slope_deg = np.degrees(np.arctan(np.sqrt(gx**2 + gy**2)))
    slope_deg[~np.isfinite(dem)] = np.nan
    return slope_deg.astype(np.float32)
