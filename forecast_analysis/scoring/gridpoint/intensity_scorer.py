#!/usr/bin/env python3
"""
Gridpoint Persistence Intensity Scorer.

This scorer computes a moving-average maximum intensity score using strict
per-window threshold exceedance: a grid point contributes to a window only if
it exceeds the daily threshold on all days of the window.
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import xarray as xr

from .persistence_scorer import (
    DEFAULT_CLIMATOLOGY_PATH,
    DEFAULT_THRESHOLDS_PATH,
    REGION_BOUNDS,
    GridpointPersistenceScorer,
    apply_monthly_threshold,
    compute_anomalies_from_climatology,
    create_region_mask,
)


class GridpointIntensityScorer(GridpointPersistenceScorer):
    """
    Scorer for moving-average maximum blocking intensity.

    For each candidate L-day window (L = min_persistence), a grid point is
    considered blocked only if it exceeds the threshold on all L days. The
    score is the maximum window-mean anomaly across valid blocked points.
    """

    name = "GridpointIntensityScorer"
    description = "Strict per-window threshold exceedance with moving-average max intensity"
    requires_blocking_detection = False

    @staticmethod
    def _compute_window_max_intensity(
        z500_anom_np: np.ndarray,
        above_threshold_np: np.ndarray,
        region_mask_np: np.ndarray,
        start_idx: int,
        end_idx: int,
        window_days: int,
        fallback_to_nonblocked: bool,
    ) -> float:
        """
        Compute max intensity over [start_idx, end_idx) using strict window logic.

        A grid point is valid at start t only if it is above threshold on all
        days in [t, t + window_days).
        """
        actual_duration = end_idx - start_idx
        if actual_duration < window_days:
            raise ValueError(
                f"Actual window duration ({actual_duration}) is less than "
                f"window_days ({window_days})"
            )

        max_start_idx = end_idx - window_days
        max_intensity = 0.0

        for t in range(start_idx, max_start_idx + 1):
            t_end = t + window_days

            window_mean = np.mean(z500_anom_np[t:t_end, :, :], axis=0)
            blocked_at_t = np.all(above_threshold_np[t:t_end], axis=0) & region_mask_np

            if np.any(blocked_at_t):
                max_at_t = float(np.max(window_mean[blocked_at_t]))
            elif fallback_to_nonblocked:
                if np.any(region_mask_np):
                    max_at_t = float(np.max(window_mean[region_mask_np]))
                else:
                    max_at_t = 0.0
            else:
                max_at_t = 0.0

            if max_at_t > max_intensity:
                max_intensity = max_at_t

        return float(max_intensity)

    def compute_intensity_score_from_anomalies(
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
        fallback_to_nonblocked: bool = False,
    ) -> float:
        """
        Compute maximum intensity score from pre-computed anomalies.

        Uses strict per-window threshold exceedance with window size equal to
        min_persistence.
        """
        window_days = self.min_persistence
        if duration_days < window_days:
            raise ValueError(
                f"duration_days ({duration_days}) must be >= window_days ({window_days})"
            )

        if all(v is not None for v in [region_lon_min, region_lon_max,
                                        region_lat_min, region_lat_max]):
            lon_min, lon_max = region_lon_min, region_lon_max
            lat_min, lat_max = region_lat_min, region_lat_max
        elif region in REGION_BOUNDS:
            lon_min, lon_max, lat_min, lat_max = REGION_BOUNDS[region]
        else:
            available = ", ".join(REGION_BOUNDS.keys())
            raise ValueError(f"Unknown region '{region}'. Available: {available}")

        window_end_idx = onset_time_idx + duration_days
        if window_end_idx > len(z500_anom.time):
            raise ValueError(
                f"Scoring window exceeds data length: onset_time_idx={onset_time_idx}, "
                f"duration_days={duration_days}, data length={len(z500_anom.time)}"
            )

        above_threshold = apply_monthly_threshold(z500_anom, threshold_90)
        region_mask = create_region_mask(z500_anom, lon_min, lon_max, lat_min, lat_max)

        return self._compute_window_max_intensity(
            z500_anom_np=z500_anom.values,
            above_threshold_np=above_threshold.values,
            region_mask_np=region_mask.values,
            start_idx=onset_time_idx,
            end_idx=window_end_idx,
            window_days=window_days,
            fallback_to_nonblocked=fallback_to_nonblocked,
        )

    def compute_intensity_score(
        self,
        z500: xr.DataArray,
        start_time: np.datetime64,
        duration_days: int,
        region: str = "Eurasia",
        region_lon_min: Optional[float] = None,
        region_lon_max: Optional[float] = None,
        region_lat_min: Optional[float] = None,
        region_lat_max: Optional[float] = None,
        fallback_to_nonblocked: bool = False,
    ) -> float:
        """
        Compute maximum intensity score for a time window from raw Z500.
        """
        window_days = self.min_persistence
        if duration_days < window_days:
            raise ValueError(
                f"duration_days ({duration_days}) must be >= window_days ({window_days})"
            )

        if all(v is not None for v in [region_lon_min, region_lon_max,
                                        region_lat_min, region_lat_max]):
            lon_min, lon_max = region_lon_min, region_lon_max
            lat_min, lat_max = region_lat_min, region_lat_max
        elif region in REGION_BOUNDS:
            lon_min, lon_max, lat_min, lat_max = REGION_BOUNDS[region]
        else:
            available = ", ".join(REGION_BOUNDS.keys())
            raise ValueError(f"Unknown region '{region}'. Available: {available}")

        start_time = np.datetime64(start_time, "ns")
        end_time = start_time + np.timedelta64(duration_days - 1, "D")

        z500_anom = compute_anomalies_from_climatology(z500, self.climatology)
        above_threshold = apply_monthly_threshold(z500_anom, self.thresholds)

        time_vals = z500_anom.time.values
        start_idx = np.searchsorted(time_vals, start_time)
        end_idx = np.searchsorted(time_vals, end_time, side="right")
        if start_idx >= len(time_vals) or end_idx <= start_idx:
            raise ValueError(
                f"No data found in window [{start_time}, {end_time}]. "
                f"Data time range: [{time_vals[0]}, {time_vals[-1]}]"
            )

        region_mask = create_region_mask(z500, lon_min, lon_max, lat_min, lat_max)

        return self._compute_window_max_intensity(
            z500_anom_np=z500_anom.values,
            above_threshold_np=above_threshold.values,
            region_mask_np=region_mask.values,
            start_idx=start_idx,
            end_idx=end_idx,
            window_days=window_days,
            fallback_to_nonblocked=fallback_to_nonblocked,
        )

    def get_score_columns(self):
        """Return score column names."""
        return ["max_intensity"]

    def get_primary_score_column(self):
        """Return primary score column name."""
        return "max_intensity"


def compute_gridpoint_intensity_score(
    z500: xr.DataArray,
    start_time,
    duration_days: int,
    region: str = "Eurasia",
    climatology_path: Union[str, Path] = DEFAULT_CLIMATOLOGY_PATH,
    thresholds_path: Union[str, Path] = DEFAULT_THRESHOLDS_PATH,
    min_persistence: int = 5,
    fallback_to_nonblocked: bool = False,
    region_lon_min: Optional[float] = None,
    region_lon_max: Optional[float] = None,
    region_lat_min: Optional[float] = None,
    region_lat_max: Optional[float] = None,
) -> float:
    """
    Convenience function for moving-average maximum intensity scoring.
    """
    scorer = GridpointIntensityScorer(
        climatology_path=climatology_path,
        thresholds_path=thresholds_path,
        min_persistence=min_persistence
    )
    return scorer.compute_intensity_score(
        z500=z500,
        start_time=start_time,
        duration_days=duration_days,
        region=region,
        region_lon_min=region_lon_min,
        region_lon_max=region_lon_max,
        region_lat_min=region_lat_min,
        region_lat_max=region_lat_max,
        fallback_to_nonblocked=fallback_to_nonblocked,
    )
