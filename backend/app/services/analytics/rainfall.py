"""Rainfall precipitation data provider interfaces:
- Open-Meteo Historical Weather API (verified endpoint)
- IMERG (GPM NASA)
- IMD (India Meteorological Department) or local CSV format
"""

from __future__ import annotations

import csv
import logging
from abc import ABC, abstractmethod
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import httpx

logger = logging.getLogger(__name__)


class RainfallProvider(ABC):
    """Abstract rainfall data provider."""

    @abstractmethod
    def fetch_daily_rainfall(
        self,
        latitude: float,
        longitude: float,
        start_date: date,
        end_date: date,
    ) -> List[Dict[str, Any]]:
        """Returns a list of dicts: [{'date': 'YYYY-MM-DD', 'precipitation_mm': float}]."""
        pass


class OpenMeteoRainfallProvider(RainfallProvider):
    """Fetches observed daily precipitation from Open-Meteo Archive API.

    Endpoint: https://archive-api.open-meteo.com/v1/archive
    No API key required.
    """

    BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

    def fetch_daily_rainfall(
        self,
        latitude: float,
        longitude: float,
        start_date: date,
        end_date: date,
    ) -> List[Dict[str, Any]]:
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "daily": "precipitation_sum",
            "timezone": "UTC",
        }

        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(self.BASE_URL, params=params)
                resp.raise_for_status()
                data = resp.json()

            daily = data.get("daily", {})
            times = daily.get("time", [])
            precip = daily.get("precipitation_sum", [])

            records = []
            for t, p in zip(times, precip):
                records.append({
                    "date": t,
                    "precipitation_mm": float(p) if p is not None else 0.0,
                })
            return records
        except Exception as e:
            logger.warning(f"Failed to query Open-Meteo rainfall: {e}")
            return []


class CSVRainfallProvider(RainfallProvider):
    """Loads rainfall from local CSV file (supports IMD, IMERG, or user gauges).

    Expected CSV columns: 'date' (YYYY-MM-DD), 'precipitation_mm' or 'rainfall_mm'.
    """

    def __init__(self, csv_path: Union[str, Path]) -> None:
        self.csv_path = Path(csv_path)

    def fetch_daily_rainfall(
        self,
        latitude: float,
        longitude: float,
        start_date: date,
        end_date: date,
    ) -> List[Dict[str, Any]]:
        if not self.csv_path.exists():
            logger.warning(f"Rainfall CSV not found at {self.csv_path}")
            return []

        records = []
        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                d_str = row.get("date") or row.get("Date")
                p_val = row.get("precipitation_mm") or row.get("rainfall_mm") or row.get("rain")
                if d_str and p_val is not None:
                    try:
                        row_date = date.fromisoformat(d_str.strip())
                        if start_date <= row_date <= end_date:
                            records.append({
                                "date": row_date.isoformat(),
                                "precipitation_mm": float(p_val.strip()),
                            })
                    except Exception:
                        pass
        return records
