#!/usr/bin/env python3
"""
Gridpoint Persistence Blocking Scorer.

This scorer computes blocking scores using DG-style per-gridpoint persistence:
- A grid cell is "blocked on day t" iff day t lies within a run of >=5 consecutive
  days above the anomaly threshold.
- Score = mean area-weighted percentage of region that is blocked across the window.

This bypasses the region-level event tracking from ANO_PlaSim.py and treats each
grid point independently.

Author: AI-RES Project
"""

import json
from pathlib import Path
from typing import Optional, Union

import numpy as np
import xarray as xr

from ..base import BlockingScorer

# Default paths for climatology and threshold files
DEFAULT_CLIMATOLOGY_PATH = Path("/glade/u/home/zhil/project/AI-RES/Blocking/data/ano_climatology_thresholds.nc")
DEFAULT_THRESHOLDS_PATH = Path("/glade/u/home/zhil/project/AI-RES/Blocking/data/ano_thresholds.json")

# Predefined region boundaries (lon_min, lon_max, lat_min, lat_max)
REGION_BOUNDS = {
    "Eurasia": (30.0, 100.0, 55.0, 75.0),
    "NorthAtlantic": (-59.0625, -2.8125, 57.20397269, 73.94716987),
}


def load_climatology(clim_path: Union[str, Path]) -> xr.DataArray:
    """
    Load day-of-year Z500 climatology from NetCDF file.

    Parameters
    ----------
    clim_path : str or Path
        Path to climatology NetCDF file.

    Returns
    -------
    xr.DataArray
        Climatology with dimensions (dayofyear, lat, lon).
    """
    ds = xr.open_dataset(clim_path)
    return ds["z500_climatology"]


def load_thresholds(thresh_path: Union[str, Path]) -> dict:
    """
    Load monthly 90th percentile thresholds from JSON file.

    Parameters
    ----------
    thresh_path : str or Path
        Path to thresholds JSON file.

    Returns
    -------
    dict
        Dictionary mapping month number (int) to threshold value (float).
    """
    with open(thresh_path, "r") as f:
        data = json.load(f)

    # Convert string keys to integers
    monthly_thresholds = {
        int(k): float(v) for k, v in data["monthly_thresholds"].items()
    }
    return monthly_thresholds


def compute_anomalies_from_climatology(
    z500: xr.DataArray,
    climatology: xr.DataArray
) -> xr.DataArray:
    """
    Compute Z500 anomalies by subtracting day-of-year climatology.

    Parameters
    ----------
    z500 : xr.DataArray
        Raw Z500 data with dimensions (time, lat, lon).
    climatology : xr.DataArray
        Day-of-year climatology with dimensions (dayofyear, lat, lon).

    Returns
    -------
    xr.DataArray
        Anomalies with same dimensions as z500.
    """
    # Extract day-of-year for each timestep
    doy = z500.time.dt.dayofyear

    # Subtract climatology for each day
    # Need to align coordinates properly
    clim_for_days = climatology.sel(dayofyear=doy)

    # Handle different xarray versions:
    # Newer xarray: .sel() with DataArray indexer already replaces 'dayofyear'
    # dim with 'time', so swap_dims would fail.
    # Older xarray: 'dayofyear' may still be a dimension that needs swapping.
    if "dayofyear" in clim_for_days.dims:
        clim_for_days = clim_for_days.assign_coords(time=z500.time)
        clim_for_days = clim_for_days.swap_dims({"dayofyear": "time"})
    clim_for_days = clim_for_days.drop_vars("dayofyear", errors="ignore")

    z500_anom = z500 - clim_for_days
    z500_anom.name = "z500_anomaly"

    return z500_anom


def apply_monthly_threshold(
    z500_anom: xr.DataArray,
    monthly_thresholds: dict
) -> xr.DataArray:
    """
    Create boolean mask of grid points exceeding monthly threshold.

    Parameters
    ----------
    z500_anom : xr.DataArray
        Z500 anomalies with dimensions (time, lat, lon).
    monthly_thresholds : dict
        Monthly thresholds mapping month (int) -> threshold (float).

    Returns
    -------
    xr.DataArray
        Boolean mask (True = above threshold) with same dimensions.
    """
    # Get month for each timestep
    months = z500_anom.time.dt.month.values

    # Build threshold array for each timestep
    T = len(months)
    threshold_arr = np.empty(T, dtype=np.float32)
    for i, m in enumerate(months):
        threshold_arr[i] = monthly_thresholds.get(m, np.nan)

    # Broadcast and compare: (T,) vs (T, lat, lon)
    above_threshold = z500_anom.values > threshold_arr[:, None, None]

    result = xr.DataArray(
        above_threshold,
        coords=z500_anom.coords,
        dims=z500_anom.dims,
        name="above_threshold"
    )
    return result


def compute_dg_blocking_mask(
    above_threshold: xr.DataArray,
    min_persistence: int = 5
) -> xr.DataArray:
    """
    Compute DG-style blocking mask with persistence criterion.

    A grid cell is "blocked on day t" iff day t lies within a run of
    >= min_persistence consecutive days above threshold.

    This uses a forward-backward approach:
    1. For each day, compute consecutive above-threshold count ending at that day
    2. For each day, compute consecutive above-threshold count starting at that day
    3. A day is blocked if (backward_count + forward_count - 1) >= min_persistence

    Parameters
    ----------
    above_threshold : xr.DataArray
        Boolean mask (time, lat, lon) indicating days above threshold.
    min_persistence : int
        Minimum consecutive days required (default: 5).

    Returns
    -------
    xr.DataArray
        Boolean blocking mask with same dimensions.
    """
    data = above_threshold.values.astype(np.int8)  # (T, lat, lon)
    T, nlat, nlon = data.shape

    # Compute backward count: consecutive 1s ending at each day (inclusive)
    backward_count = np.zeros_like(data, dtype=np.int16)
    backward_count[0] = data[0]
    for t in range(1, T):
        # If above threshold, increment from previous; else reset to 0
        backward_count[t] = np.where(data[t] == 1, backward_count[t-1] + 1, 0)

    # Compute forward count: consecutive 1s starting at each day (inclusive)
    forward_count = np.zeros_like(data, dtype=np.int16)
    forward_count[-1] = data[-1]
    for t in range(T - 2, -1, -1):
        forward_count[t] = np.where(data[t] == 1, forward_count[t+1] + 1, 0)

    # Total run length containing each day = backward + forward - 1 (avoid double counting the day itself)
    # But we need to be careful: if a day is not above threshold, run length is 0
    run_length = np.where(data == 1, backward_count + forward_count - 1, 0)

    # A day is blocked if it's part of a run >= min_persistence
    blocked = run_length >= min_persistence

    result = xr.DataArray(
        blocked,
        coords=above_threshold.coords,
        dims=above_threshold.dims,
        name="dg_blocking_mask"
    )
    return result


def create_region_mask(
    z500: xr.DataArray,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float
) -> xr.DataArray:
    """
    Create a boolean mask for the specified region.

    Handles both 0-360 and -180 to 180 longitude conventions.

    Parameters
    ----------
    z500 : xr.DataArray
        Data array with lat/lon coordinates.
    lon_min, lon_max, lat_min, lat_max : float
        Region boundaries.

    Returns
    -------
    xr.DataArray
        Boolean mask (lat, lon) for the region.
    """
    lat = z500.lat
    lon = z500.lon

    # Handle longitude wrapping for regions that cross the prime meridian
    # or use different conventions
    lon_vals = lon.values

    # Check if lon_min/lon_max are in -180 to 180 convention
    # and data is in 0-360 convention (or vice versa)
    if lon_min < 0 and lon_vals.min() >= 0:
        # Convert region bounds to 0-360
        lon_min_adj = lon_min % 360
        lon_max_adj = lon_max % 360
    elif lon_min >= 0 and lon_vals.min() < 0:
        # Convert region bounds to -180 to 180
        lon_min_adj = ((lon_min + 180) % 360) - 180
        lon_max_adj = ((lon_max + 180) % 360) - 180
    else:
        lon_min_adj = lon_min
        lon_max_adj = lon_max

    # Create mask
    lat_mask = (lat >= lat_min) & (lat <= lat_max)

    if lon_min_adj <= lon_max_adj:
        lon_mask = (lon >= lon_min_adj) & (lon <= lon_max_adj)
    else:
        # Region wraps around (e.g., 350 to 10 degrees)
        lon_mask = (lon >= lon_min_adj) | (lon <= lon_max_adj)

    # Combine to 2D mask
    mask = lat_mask & lon_mask

    return mask


def compute_area_weights(lat: xr.DataArray) -> xr.DataArray:
    """
    Compute area weights proportional to cos(latitude).

    Parameters
    ----------
    lat : xr.DataArray
        Latitude coordinate array.

    Returns
    -------
    xr.DataArray
        Area weights with same shape as lat.
    """
    weights = np.cos(np.deg2rad(lat))
    return weights


class GridpointPersistenceScorer(BlockingScorer):
    """
    Blocking scorer using DG-style per-gridpoint persistence.

    This scorer computes blocking without relying on region-level event tracking.
    Instead, it treats each grid point independently:
    - A grid cell is "blocked on day t" iff day t lies within a run of >=5
      consecutive days above the anomaly threshold.
    - Score = mean area-weighted percentage of region that is blocked.

    Parameters
    ----------
    climatology_path : str or Path, optional
        Path to climatology NetCDF file. Default uses standard path.
    thresholds_path : str or Path, optional
        Path to thresholds JSON file. Default uses standard path.
    min_persistence : int, optional
        Minimum consecutive days above threshold to count as blocking.
        Default: 5. Must be >= 5.

    Attributes
    ----------
    requires_blocking_detection : bool
        Set to False because this scorer does not need pre-computed blocking.
    """

    name = "GridpointPersistenceScorer"
    description = "DG-style per-gridpoint blocking with persistence criterion"
    requires_blocking_detection = False

    def __init__(
        self,
        climatology_path: Union[str, Path] = DEFAULT_CLIMATOLOGY_PATH,
        thresholds_path: Union[str, Path] = DEFAULT_THRESHOLDS_PATH,
        min_persistence: int = 5
    ):
        if min_persistence < 5:
            raise ValueError(f"min_persistence must be >= 5, got {min_persistence}")

        self.climatology_path = Path(climatology_path)
        self.thresholds_path = Path(thresholds_path)
        self.min_persistence = min_persistence

        # Lazy loading - load on first use
        self._climatology = None
        self._thresholds = None

    @property
    def climatology(self) -> xr.DataArray:
        """Lazy-loaded climatology data."""
        if self._climatology is None:
            self._climatology = load_climatology(self.climatology_path)
        return self._climatology

    @property
    def thresholds(self) -> dict:
        """Lazy-loaded monthly thresholds."""
        if self._thresholds is None:
            self._thresholds = load_thresholds(self.thresholds_path)
        return self._thresholds

    def compute_score_from_anomalies(
        self,
        z500_anom: xr.DataArray,
        threshold_90: dict,
        onset_time_idx: int,
        duration_days: int,
        region: str = "Eurasia",
        region_lon_min: Optional[float] = None,
        region_lon_max: Optional[float] = None,
        region_lat_min: Optional[float] = None,
        region_lat_max: Optional[float] = None,
    ) -> float:
        """
        Compute score from pre-computed anomalies and thresholds.

        This method is used when anomalies and thresholds are already computed
        (e.g., in the RES experiment pipeline).

        Parameters
        ----------
        z500_anom : xr.DataArray
            Pre-computed Z500 anomalies with dimensions (time, lat, lon).
        threshold_90 : dict
            Monthly thresholds mapping month (int) -> threshold (float).
        onset_time_idx : int
            Time index for start of scoring window.
        duration_days : int
            Length of scoring window in days. Must be >= 5.
        region : str, optional
            Predefined region name ("Eurasia" or "NorthAtlantic").
        region_lon_min, region_lon_max : float, optional
            Explicit longitude bounds.
        region_lat_min, region_lat_max : float, optional
            Explicit latitude bounds.

        Returns
        -------
        float
            Mean blocked percentage (0-100) across the window.
        """
        if duration_days < 5:
            raise ValueError(
                f"duration_days must be >= 5 for DG-style blocking detection, got {duration_days}"
            )

        # Determine region bounds
        if all(v is not None for v in [region_lon_min, region_lon_max,
                                        region_lat_min, region_lat_max]):
            lon_min, lon_max = region_lon_min, region_lon_max
            lat_min, lat_max = region_lat_min, region_lat_max
        elif region in REGION_BOUNDS:
            lon_min, lon_max, lat_min, lat_max = REGION_BOUNDS[region]
        else:
            available = ", ".join(REGION_BOUNDS.keys())
            raise ValueError(f"Unknown region '{region}'. Available: {available}")

        # Compute end index
        end_time_idx = onset_time_idx + duration_days

        if end_time_idx > len(z500_anom.time):
            raise ValueError(
                f"Scoring window exceeds data length: onset_time_idx={onset_time_idx}, "
                f"duration_days={duration_days}, data length={len(z500_anom.time)}"
            )

        # Step 1: Apply monthly thresholds to get above-threshold mask
        above_threshold = apply_monthly_threshold(z500_anom, threshold_90)

        # Step 2: Compute DG-style blocking mask (uses full time series for run detection)
        blocking_mask = compute_dg_blocking_mask(above_threshold, self.min_persistence)

        # Step 3: Select the scoring window by index
        window_mask = blocking_mask.isel(time=slice(onset_time_idx, end_time_idx))

        if len(window_mask.time) == 0:
            return 0.0

        # Step 4: Create region mask and area weights
        region_mask = create_region_mask(z500_anom, lon_min, lon_max, lat_min, lat_max)
        area_weights = compute_area_weights(z500_anom.lat)

        # Step 5: Compute daily blocked percentage within region
        daily_percentages = []

        for t in range(len(window_mask.time)):
            day_mask = window_mask.isel(time=t)

            # Apply region mask
            blocked_in_region = day_mask.where(region_mask, drop=False)

            # Compute area-weighted percentage
            weights_2d = area_weights.broadcast_like(day_mask)

            # Mask to region
            region_blocked = blocked_in_region.where(region_mask, 0).values
            region_weights = weights_2d.where(region_mask, 0).values

            total_blocked_area = np.sum(region_blocked * region_weights)
            total_area = np.sum(region_weights)

            if total_area > 0:
                pct = 100.0 * total_blocked_area / total_area
            else:
                pct = 0.0

            daily_percentages.append(pct)

        # Step 6: Return mean percentage
        mean_pct = np.mean(daily_percentages)

        return float(mean_pct)

    def compute_score(
        self,
        z500: xr.DataArray,
        start_time: np.datetime64,
        duration_days: int,
        region: str = "Eurasia",
        region_lon_min: Optional[float] = None,
        region_lon_max: Optional[float] = None,
        region_lat_min: Optional[float] = None,
        region_lat_max: Optional[float] = None,
    ) -> float:
        """
        Compute mean blocked percentage for a time window.

        Parameters
        ----------
        z500 : xr.DataArray
            Raw Z500 data with dimensions (time, lat, lon).
            Should include data before start_time for accurate run detection.
        start_time : np.datetime64 or datetime-like
            Start of the scoring window.
        duration_days : int
            Length of scoring window in days. Must be >= 5.
        region : str, optional
            Predefined region name ("Eurasia" or "NorthAtlantic").
            Used only if explicit bounds are not provided.
        region_lon_min, region_lon_max : float, optional
            Explicit longitude bounds. Override region preset if provided.
        region_lat_min, region_lat_max : float, optional
            Explicit latitude bounds. Override region preset if provided.

        Returns
        -------
        float
            Mean blocked percentage (0-100) across the window.

        Raises
        ------
        ValueError
            If duration_days < 5 or region is unknown.
        """
        if duration_days < 5:
            raise ValueError(
                f"duration_days must be >= 5 for DG-style blocking detection, got {duration_days}"
            )

        # Determine region bounds
        if all(v is not None for v in [region_lon_min, region_lon_max,
                                        region_lat_min, region_lat_max]):
            lon_min, lon_max = region_lon_min, region_lon_max
            lat_min, lat_max = region_lat_min, region_lat_max
        elif region in REGION_BOUNDS:
            lon_min, lon_max, lat_min, lat_max = REGION_BOUNDS[region]
        else:
            available = ", ".join(REGION_BOUNDS.keys())
            raise ValueError(f"Unknown region '{region}'. Available: {available}")

        # Compute end time
        start_time = np.datetime64(start_time, 'ns')
        end_time = start_time + np.timedelta64(duration_days - 1, 'D')

        # Step 1: Compute anomalies from climatology
        z500_anom = compute_anomalies_from_climatology(z500, self.climatology)

        # Step 2: Apply monthly thresholds to get above-threshold mask
        above_threshold = apply_monthly_threshold(z500_anom, self.thresholds)

        # Step 3: Compute DG-style blocking mask (uses full time series for run detection)
        blocking_mask = compute_dg_blocking_mask(above_threshold, self.min_persistence)

        # Step 4: Select the scoring window [start_time, end_time]
        window_mask = blocking_mask.sel(time=slice(start_time, end_time))

        if len(window_mask.time) == 0:
            raise ValueError(
                f"No data found in window [{start_time}, {end_time}]. "
                f"Data time range: [{z500.time.values[0]}, {z500.time.values[-1]}]"
            )

        # Step 5: Create region mask and area weights
        region_mask = create_region_mask(z500, lon_min, lon_max, lat_min, lat_max)
        area_weights = compute_area_weights(z500.lat)

        # Step 6: Compute daily blocked percentage within region
        # For each day, compute area-weighted mean of blocking within region
        daily_percentages = []

        for t in range(len(window_mask.time)):
            day_mask = window_mask.isel(time=t)

            # Apply region mask
            blocked_in_region = day_mask.where(region_mask, drop=False)

            # Compute area-weighted percentage
            # numerator: sum of (blocked * weight) over region
            # denominator: sum of weights over region
            weights_2d = area_weights.broadcast_like(day_mask)

            # Mask to region
            region_blocked = blocked_in_region.where(region_mask, 0).values
            region_weights = weights_2d.where(region_mask, 0).values

            total_blocked_area = np.sum(region_blocked * region_weights)
            total_area = np.sum(region_weights)

            if total_area > 0:
                pct = 100.0 * total_blocked_area / total_area
            else:
                pct = 0.0

            daily_percentages.append(pct)

        # Step 7: Return mean percentage
        mean_pct = np.mean(daily_percentages)

        return float(mean_pct)

    def compute_event_scores(
        self,
        z500: xr.DataArray,
        event_info: dict,
        region_lon_min: float = 30.0,
        region_lon_max: float = 100.0,
        region_lat_min: float = 55.0,
        region_lat_max: float = 75.0,
        **kwargs
    ):
        """
        Compatibility method for BlockingScorer interface.

        This scorer does not use event_info. For the intended usage,
        call compute_score() directly instead.

        Raises
        ------
        NotImplementedError
            Always, because this scorer uses a different interface.
        """
        raise NotImplementedError(
            "GridpointPersistenceScorer does not use the event_info interface. "
            "Use compute_score() instead."
        )

    def get_score_columns(self):
        """Return score column names."""
        return ["blocked_pct"]

    def get_primary_score_column(self):
        """Return primary score column name."""
        return "blocked_pct"


# Convenience function for direct usage
def compute_gridpoint_blocking_score(
    z500: xr.DataArray,
    start_time,
    duration_days: int,
    region: str = "Eurasia",
    climatology_path: Union[str, Path] = DEFAULT_CLIMATOLOGY_PATH,
    thresholds_path: Union[str, Path] = DEFAULT_THRESHOLDS_PATH,
    min_persistence: int = 5,
    region_lon_min: Optional[float] = None,
    region_lon_max: Optional[float] = None,
    region_lat_min: Optional[float] = None,
    region_lat_max: Optional[float] = None,
) -> float:
    """
    Compute gridpoint persistence blocking score.

    Convenience function that creates a scorer and computes the score.

    Parameters
    ----------
    z500 : xr.DataArray
        Raw Z500 data with dimensions (time, lat, lon).
        Should include data before start_time for accurate run detection.
    start_time : datetime-like
        Start of the scoring window.
    duration_days : int
        Length of scoring window in days. Must be >= 5.
    region : str, optional
        Predefined region name ("Eurasia" or "NorthAtlantic").
    climatology_path : str or Path, optional
        Path to climatology NetCDF file.
    thresholds_path : str or Path, optional
        Path to thresholds JSON file.
    min_persistence : int, optional
        Minimum consecutive days (default: 5).
    region_lon_min, region_lon_max : float, optional
        Explicit longitude bounds.
    region_lat_min, region_lat_max : float, optional
        Explicit latitude bounds.

    Returns
    -------
    float
        Mean blocked percentage (0-100) across the window.
    """
    scorer = GridpointPersistenceScorer(
        climatology_path=climatology_path,
        thresholds_path=thresholds_path,
        min_persistence=min_persistence
    )

    return scorer.compute_score(
        z500=z500,
        start_time=start_time,
        duration_days=duration_days,
        region=region,
        region_lon_min=region_lon_min,
        region_lon_max=region_lon_max,
        region_lat_min=region_lat_min,
        region_lat_max=region_lat_max,
    )
