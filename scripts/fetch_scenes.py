"""SAR Scene acquisition and search script using asf_search.

Features:
- Configurable and verified product identifiers (Sentinel-1 GRD_HD/RTC, NISAR GCOV)
- Earthdata credentials via environment variables (EARTHDATA_USERNAME, EARTHDATA_PASSWORD) or .env, never hardcoded
- List scenes for AOI + dates (event and/or baseline)
- Group by relative orbit / track number and acquisition date
- Report detailed coverage summary BEFORE downloading
- Safe --dry-run option (default enabled unless --download is passed)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import asf_search as asf
from rich.console import Console
from rich.table import Table

# Add project root to sys.path so src imports work cleanly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import EventConfig, SensorType

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("fetch_scenes")
console = Console()

# Official verified ASF search constants / mappings
VERIFIED_DATASETS = {
    SensorType.SENTINEL1: {
        "platform": asf.PLATFORM.SENTINEL1,
        "default_processing_level": asf.PRODUCT_TYPE.GRD_HD,
        "supported_processing_levels": [asf.PRODUCT_TYPE.GRD_HD, asf.PRODUCT_TYPE.RTC],
        "default_polarization": [asf.POLARIZATION.VV, asf.POLARIZATION.VV_VH],
    },
    SensorType.NISAR: {
        "dataset": asf.DATASET.NISAR,
        "default_processing_level": asf.PRODUCT_TYPE.GCOV,
        "supported_processing_levels": [asf.PRODUCT_TYPE.GCOV],
        "default_polarization": None,  # NISAR GCOV includes all polarimetric channels
    },
}


def get_earthdata_session() -> Optional[asf.ASFSession]:
    """Authenticates an ASF session with NASA Earthdata credentials if present in env."""
    username = os.environ.get("EARTHDATA_USERNAME")
    password = os.environ.get("EARTHDATA_PASSWORD")

    if not username or not password:
        logger.warning(
            "EARTHDATA_USERNAME or EARTHDATA_PASSWORD environment variable not set. "
            "Public search will work, but downloading restricted data products requires credentials."
        )
        return None

    try:
        session = asf.ASFSession().auth_with_creds(username=username, password=password)
        logger.info("Successfully authenticated with NASA Earthdata Login.")
        return session
    except Exception as e:
        logger.error(f"Failed to authenticate with Earthdata Login: {e}")
        return None


def search_scenes_for_window(
    wkt_aoi: str,
    sensor: SensorType,
    start_date: date,
    end_date: date,
    processing_level: Optional[str] = None,
    polarization: Optional[str] = None,
    max_results: Optional[int] = None,
) -> asf.ASFSearchResults:
    """Executes verified ASF search query with proper parameters."""
    cfg = VERIFIED_DATASETS[sensor]
    search_kwargs: Dict[str, Any] = {
        "intersectsWith": wkt_aoi,
        "start": start_date.isoformat(),
        "end": end_date.isoformat(),
    }

    if max_results:
        search_kwargs["maxResults"] = max_results

    # Set platform or dataset based on verified constants
    if "platform" in cfg:
        search_kwargs["platform"] = cfg["platform"]
    if "dataset" in cfg:
        search_kwargs["dataset"] = cfg["dataset"]

    # Processing level
    lvl = processing_level or cfg["default_processing_level"]
    search_kwargs["processingLevel"] = lvl

    # Polarization filtering (primarily for Sentinel-1)
    if sensor == SensorType.SENTINEL1 and polarization:
        pol_raw = polarization.upper().strip()
        # In Sentinel-1, single channel acquisitions (e.g. 'VV') are often distributed as dual-pol 'VV+VH'
        if pol_raw in ("VV", "VH"):
            search_kwargs["polarization"] = [asf.POLARIZATION.VV, asf.POLARIZATION.VV_VH, asf.POLARIZATION.DUAL_VV]
        elif pol_raw in ("HH", "HV"):
            search_kwargs["polarization"] = [asf.POLARIZATION.HH, asf.POLARIZATION.HH_HV, asf.POLARIZATION.DUAL_HH]
        else:
            pol_formatted = pol_raw.replace("+", "_")
            search_kwargs["polarization"] = pol_formatted

    logger.info(f"Querying ASF CMR for {sensor.value} ({start_date} to {end_date})...")
    try:
        results = asf.search(**search_kwargs)
        return results
    except Exception as e:
        logger.error(f"Search query failed: {e}")
        return asf.ASFSearchResults([])


def group_and_summarize_scenes(
    results: asf.ASFSearchResults,
    period_label: str = "Window",
) -> Dict[str, Any]:
    """Groups search results by relative orbit (track) and acquisition date.

    Returns grouped data and prints a formatted summary table.
    """
    grouped: Dict[Optional[int], Dict[str, List[asf.ASFProduct]]] = defaultdict(lambda: defaultdict(list))
    total_scenes = len(results)

    table = Table(
        title=f"SAR Scene Coverage Summary: {period_label} ({total_scenes} scenes found)",
        header_style="bold cyan",
    )
    table.add_column("Relative Orbit / Track", style="bold yellow", justify="center")
    table.add_column("Acquisition Date (UTC)", style="magenta")
    table.add_column("Sensor / Platform", style="green")
    table.add_column("Polarization", style="white")
    table.add_column("Product Level", style="blue")
    table.add_column("Scene / Granule ID", style="dim")
    table.add_column("Size (MB)", justify="right")

    for product in results:
        props = product.properties
        # Relative orbit / path number
        rel_orbit = props.get("pathNumber")
        if rel_orbit is None:
            rel_orbit = props.get("orbit")

        # Start time / date
        start_time_str = props.get("startTime", "")
        acq_date_str = start_time_str.split("T")[0] if "T" in start_time_str else "Unknown"

        grouped[rel_orbit][acq_date_str].append(product)

        # Polarization info
        pol = props.get("polarization")
        if not pol:
            main_pol = props.get("mainBandPolarization")
            pol = "+".join(main_pol) if isinstance(main_pol, list) else str(main_pol)

        # File size calculation
        bytes_val = props.get("bytes", 0)
        size_mb = f"{bytes_val / (1024 * 1024):.1f}" if isinstance(bytes_val, (int, float)) and bytes_val > 0 else "N/A"

        table.add_row(
            str(rel_orbit) if rel_orbit is not None else "Unknown",
            props.get("startTime", acq_date_str),
            str(props.get("platform", "Unknown")),
            str(pol or "N/A"),
            str(props.get("processingLevel", "N/A")),
            props.get("sceneName", props.get("fileID", "Unknown"))[:36] + "...",
            size_mb,
        )

    console.print(table)
    return grouped


def download_scenes(
    results: asf.ASFSearchResults,
    output_dir: Path,
    session: Optional[asf.ASFSession] = None,
) -> List[Path]:
    """Downloads scenes to the designated directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded_paths = []
    logger.info(f"Downloading {len(results)} scenes into {output_dir}...")

    for prod in results:
        fname = prod.properties.get("fileName") or f"{prod.properties.get('sceneName')}.zip"
        dest_path = output_dir / fname
        if dest_path.exists():
            logger.info(f"Scene already exists locally: {dest_path.name}")
            downloaded_paths.append(dest_path)
            continue
        try:
            prod.download(path=str(output_dir), session=session)
            downloaded_paths.append(dest_path)
            logger.info(f"Successfully downloaded: {fname}")
        except Exception as e:
            logger.error(f"Failed to download {fname}: {e}")

    return downloaded_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search, group, and fetch SAR scenes for Flood event analysis."
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to event configuration YAML file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Perform search, grouping, and coverage summary without downloading (Default: True).",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        default=False,
        help="Execute actual download of scenes after coverage summary.",
    )
    parser.add_argument(
        "--include-baseline",
        action="store_true",
        default=False,
        help="Also fetch scenes for the baseline date window.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="data/raw_scenes",
        help="Directory to store downloaded SAR granules.",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=20,
        help="Max results limit per search window.",
    )

    args = parser.parse_args()

    # Load and validate event configuration
    config_path = Path(args.config)
    if not config_path.exists():
        console.print(f"[bold red]Config file not found: {config_path}[/bold red]")
        sys.exit(1)

    event_cfg = EventConfig.from_yaml(config_path)
    console.print(
        f"[bold green]Loaded Event Config:[/bold green] [yellow]{event_cfg.id}[/yellow] - {event_cfg.name}"
    )
    console.print(f"Sensor: [cyan]{event_cfg.sensor.value}[/cyan] | Target Pol: [cyan]{event_cfg.polarization}[/cyan]")

    wkt_aoi = event_cfg.get_aoi_wkt()
    out_dir = Path(args.output_dir)

    # 1. Search Event Window
    console.print(f"\n[bold]Searching Event Window ({event_cfg.event_date_range.start} to {event_cfg.event_date_range.end})...[/bold]")
    event_results = search_scenes_for_window(
        wkt_aoi=wkt_aoi,
        sensor=event_cfg.sensor,
        start_date=event_cfg.event_date_range.start,
        end_date=event_cfg.event_date_range.end,
        polarization=event_cfg.polarization,
        max_results=args.max_results,
    )
    group_and_summarize_scenes(event_results, period_label=f"Event Window ({event_cfg.id})")

    # 2. Search Baseline Window if requested
    baseline_results = asf.ASFSearchResults([])
    if args.include_baseline:
        console.print(f"\n[bold]Searching Baseline Window ({event_cfg.baseline_date_range.start} to {event_cfg.baseline_date_range.end})...[/bold]")
        baseline_results = search_scenes_for_window(
            wkt_aoi=wkt_aoi,
            sensor=event_cfg.sensor,
            start_date=event_cfg.baseline_date_range.start,
            end_date=event_cfg.baseline_date_range.end,
            polarization=event_cfg.polarization,
            max_results=args.max_results,
        )
        group_and_summarize_scenes(baseline_results, period_label=f"Baseline Window ({event_cfg.id})")

    # Check download action
    all_results = event_results + baseline_results
    if not args.download or args.dry_run and not args.download:
        console.print(
            "\n[bold yellow]--dry-run active.[/bold yellow] Coverage summary displayed. "
            "To download scenes, specify `--download`."
        )
        return

    if not all_results:
        console.print("[yellow]No scenes found to download.[/yellow]")
        return

    # Authenticate and download
    session = get_earthdata_session()
    download_scenes(all_results, output_dir=out_dir, session=session)
    console.print(f"[bold green]Downloads completed into {out_dir}[/bold green]")


if __name__ == "__main__":
    main()
