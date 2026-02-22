#!/usr/bin/env python3
"""
Unified ANO-based Blocking Scorer.

This module provides ANOScorer, a unified scorer that merges the functionality
of DriftPenalizedScorer and IntegratedScorer with configurable options.

Supports two event detection modes:
- "auto": Automatically find all contiguous regional events (DriftPenalizedScorer logic)
- "onset": Track specific block from given onset time (IntegratedScorer logic)

Optionally applies drift penalty for longitudinal drift.

Author: AI-RES Project
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import xarray as xr

from ..base import BlockingScorer
from .utils import (
    REGION_BOUNDS,
    create_spatial_weight_mask,
    select_event_at_onset,
    select_block_at_onset,
    track_block_through_time,
    _integrate_over_masks,
)


class ANOScorer(BlockingScorer):
    """
    Unified ANO-based blocking scorer with configurable options.

    Supports two event detection modes:
    - "auto": Automatically find all contiguous regional events
    - "onset": Track specific block from given onset time

    Optionally applies drift penalty for longitudinal drift.

    Parameters
    ----------
    mode : str, optional
        Event detection mode: "auto" or "onset". Default: "auto".
    onset_time_idx : int, optional
        Time index of blocking onset. Required if mode="onset".
    integration_days : int, optional
        Number of days to integrate from the start of each event. Default: 5.
    n_days : int, optional
        Alias for integration_days (for IntegratedScorer compatibility).
    use_drift_penalty : bool, optional
        Whether to apply drift penalty. Default: False.
    gamma : float, optional
        Drift penalty constant. Default: 5.0.
    region_lon_min, region_lon_max : float, optional
        Longitude bounds for target ROI. Default: North Atlantic (-60 to 0).
    region_lat_min, region_lat_max : float, optional
        Latitude bounds for target ROI. Default: North Atlantic (55 to 75).

    Attributes
    ----------
    name : str
        Human-readable name of the scoring method.
    description : str
        Brief description of how the scoring works.
    requires_blocking_detection : bool
        Always True for ANO-based scorers.

    Examples
    --------
    Auto mode with drift penalty (equivalent to DriftPenalizedScorer):

    >>> scorer = ANOScorer(mode="auto", use_drift_penalty=True, gamma=5.0)
    >>> df = scorer.compute_event_scores(z500, event_info)

    Onset mode without drift penalty (equivalent to IntegratedScorer):

    >>> scorer = ANOScorer(mode="onset", onset_time_idx=10)
    >>> df = scorer.compute_event_scores(z500, event_info)
    """

    name = "ANOScorer"
    description = "Unified ANO-based scorer with configurable event detection and drift penalty"
    requires_blocking_detection = True

    def __init__(
        self,
        # Event detection mode
        mode: str = "auto",
        onset_time_idx: Optional[int] = None,
        # Integration parameters
        integration_days: int = 5,
        n_days: Optional[int] = None,  # Alias for integration_days
        # Drift penalty
        use_drift_penalty: bool = False,
        gamma: float = 5.0,
        # Region bounds (defaults to North Atlantic)
        region_lon_min: float = -60.0,
        region_lon_max: float = 0.0,
        region_lat_min: float = 55.0,
        region_lat_max: float = 75.0,
    ):
        if mode not in ("auto", "onset"):
            raise ValueError(f"mode must be 'auto' or 'onset', got '{mode}'")

        if mode == "onset" and onset_time_idx is None:
            raise ValueError("onset_time_idx is required when mode='onset'")

        self.mode = mode
        self.onset_time_idx = onset_time_idx

        # Handle n_days alias for backward compatibility
        if n_days is not None:
            self.integration_days = n_days
        else:
            self.integration_days = integration_days

        self.use_drift_penalty = use_drift_penalty
        self.gamma = gamma

        self.region_lon_min = region_lon_min
        self.region_lon_max = region_lon_max
        self.region_lat_min = region_lat_min
        self.region_lat_max = region_lat_max

    def compute_event_scores(
        self,
        z500: xr.DataArray,
        event_info: Dict,
        region_lon_min: Optional[float] = None,
        region_lon_max: Optional[float] = None,
        region_lat_min: Optional[float] = None,
        region_lat_max: Optional[float] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Compute scores for blocking events.

        In "auto" mode: finds all contiguous events and scores each
        In "onset" mode: tracks and scores the block at onset_time_idx

        Parameters
        ----------
        z500 : xr.DataArray
            Input geopotential height data with dimensions (time, lat, lon).
        event_info : Dict
            Event information dictionary containing 'blocked_mask' and 'z500_anom'.
            May also contain 'event_mask' for onset mode.
        region_lon_min, region_lon_max : float, optional
            Longitude bounds for scoring region. Overrides instance defaults.
        region_lat_min, region_lat_max : float, optional
            Latitude bounds for scoring region. Overrides instance defaults.

        Returns
        -------
        pd.DataFrame
            DataFrame with scored events. Columns depend on mode and drift penalty:

            Auto mode without drift: region_event_id, start_time, end_time,
                duration_days, integration_days, P_block

            Auto mode with drift: same + P_drift, P_total, mean_lon_centroid

            Onset mode without drift: onset_time_idx, onset_lon, onset_lat,
                n_days, B_int

            Onset mode with drift: same + P_drift, B_total
        """
        # Use provided bounds or fall back to instance defaults
        lon_min = region_lon_min if region_lon_min is not None else self.region_lon_min
        lon_max = region_lon_max if region_lon_max is not None else self.region_lon_max
        lat_min = region_lat_min if region_lat_min is not None else self.region_lat_min
        lat_max = region_lat_max if region_lat_max is not None else self.region_lat_max

        if self.mode == "auto":
            return self._compute_auto_events(
                z500, event_info, lon_min, lon_max, lat_min, lat_max
            )
        else:
            return self._compute_onset_event(
                z500, event_info, lon_min, lon_max, lat_min, lat_max
            )

    def _compute_auto_events(
        self,
        z500: xr.DataArray,
        event_info: Dict,
        lon_min: float,
        lon_max: float,
        lat_min: float,
        lat_max: float,
    ) -> pd.DataFrame:
        """
        Find contiguous regional events and score each (DriftPenalizedScorer logic).
        """
        # 1. Construct Regional Mask W(lambda, phi)
        W = create_spatial_weight_mask(z500, lat_min, lat_max, lon_min, lon_max)

        # 2. Form Regional Blocking Mask I_R
        blocked_mask = event_info["blocked_mask"]  # (time, lat, lon)
        I_R = blocked_mask * W  # (time, lat, lon)

        # Define column names based on drift penalty setting
        if self.use_drift_penalty:
            empty_columns = [
                "region_event_id", "start_time", "end_time", "duration_days",
                "integration_days", "P_block", "P_drift", "P_total", "mean_lon_centroid"
            ]
        else:
            empty_columns = [
                "region_event_id", "start_time", "end_time", "duration_days",
                "integration_days", "P_block"
            ]

        # Handle case where I_R is all zeros
        if I_R.sum() == 0:
            return pd.DataFrame(columns=empty_columns)

        # 3. Define Region-Level Events
        # Area arithmetic: A_R(t) = sum(I_R * cos(lat))
        weights_lat = np.cos(np.deg2rad(z500.lat))
        weighted_mask = I_R * weights_lat
        A_R = weighted_mask.sum(dim=['lat', 'lon'])

        # Get boolean array for regional blocking existence
        is_blocked_regionally = A_R > 0

        # Find contiguous intervals
        condition = is_blocked_regionally.values.astype(int)
        condition_padded = np.pad(condition, (1, 1), mode='constant', constant_values=0)
        diff = np.diff(condition_padded)

        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]  # Exclusive index

        n_events = len(starts)

        # 4. Compute Scores
        z_anom = event_info["z500_anom"]
        z_plus_prime = xr.where(z_anom > 0, z_anom, 0)  # Z'+ = max(0, Z')

        results = []

        for i in range(n_events):
            t_start_idx = starts[i]
            t_end_idx_full = ends[i]  # Exclusive, full event end

            if t_end_idx_full <= t_start_idx:
                continue

            region_event_id = i + 1

            # Full event duration
            full_duration = t_end_idx_full - t_start_idx

            # Determine integration window based on integration_days
            if self.integration_days is None:
                # Integrate entire event
                t_end_idx = t_end_idx_full
                integration_duration = full_duration
            else:
                # Integrate first N days (or full event if shorter)
                integration_duration = min(full_duration, self.integration_days)
                t_end_idx = t_start_idx + integration_duration

            # Time slices
            times_slice = z500.time.isel(time=slice(t_start_idx, t_end_idx))
            t_start = times_slice.values[0]
            t_end = times_slice.values[-1]
            duration = integration_duration

            # --- Compute P_block ---
            I_R_slice = I_R.isel(time=slice(t_start_idx, t_end_idx))
            Z_plus_slice = z_plus_prime.isel(time=slice(t_start_idx, t_end_idx))

            integrand = I_R_slice * Z_plus_slice * weights_lat
            P_block = integrand.sum().item()

            result = {
                "region_event_id": region_event_id,
                "start_time": t_start,
                "end_time": t_end,
                "duration_days": full_duration,
                "integration_days": integration_duration,
                "P_block": P_block,
            }

            # --- Compute P_drift if enabled ---
            if self.use_drift_penalty:
                lons = z500.lon.values
                lats = z500.lat.values
                ww = weights_lat.values

                W_grid = np.broadcast_to(ww[:, None], I_R_slice.shape[1:])
                LON_grid, _ = np.meshgrid(lons, lats)

                lambda_c_list = []

                for t in range(duration):
                    mask_t = I_R_slice.isel(time=t).values > 0
                    w_i = W_grid[mask_t]
                    lam_i = LON_grid[mask_t]

                    denom = np.sum(w_i)
                    if denom == 0:
                        lambda_c = np.nan
                    else:
                        lambda_c = np.sum(w_i * lam_i) / denom

                    lambda_c_list.append(lambda_c)

                lambda_c_arr = np.array(lambda_c_list)

                # Compute event-mean longitude centroid
                lambda_mean = np.nanmean(lambda_c_arr)

                # Compute drift penalty using spread around the mean
                valid_mask = ~np.isnan(lambda_c_arr)
                if np.sum(valid_mask) > 0:
                    diffs = lambda_c_arr[valid_mask] - lambda_mean
                    P_drift = -self.gamma * np.sum(diffs**2)
                else:
                    P_drift = 0.0

                P_total = P_block + P_drift

                result["P_drift"] = P_drift
                result["P_total"] = P_total
                result["mean_lon_centroid"] = lambda_mean

            results.append(result)

        # Sort and return
        if results:
            df = pd.DataFrame(results)
            sort_col = "P_total" if self.use_drift_penalty else "P_block"
            return df.sort_values(by=sort_col, ascending=False)
        else:
            return pd.DataFrame(columns=empty_columns)

    def _compute_onset_event(
        self,
        z500: xr.DataArray,
        event_info: Dict,
        lon_min: float,
        lon_max: float,
        lat_min: float,
        lat_max: float,
    ) -> pd.DataFrame:
        """
        Track and score block from onset time (IntegratedScorer logic).
        """
        z500_anom = event_info['z500_anom']
        event_mask = event_info.get('event_mask', None)
        blocked_mask = event_info.get('blocked_mask', None)

        result = self._compute_integrated_score_core(
            z500_anom=z500_anom,
            onset_time_idx=self.onset_time_idx,
            n_days=self.integration_days,
            region_lon_min=lon_min,
            region_lon_max=lon_max,
            region_lat_min=lat_min,
            region_lat_max=lat_max,
            event_mask=event_mask,
            blocked_mask=blocked_mask,
        )

        row = {
            'onset_time_idx': self.onset_time_idx,
            'onset_lon': result['onset_lon'],
            'onset_lat': result['onset_lat'],
            'n_days': self.integration_days,
            'B_int': result['B_int'],
        }

        # Add drift penalty if enabled
        if self.use_drift_penalty and result['tracked_masks']:
            P_drift = self._compute_drift_penalty_from_masks(
                z500_anom, result['tracked_masks'], self.onset_time_idx
            )
            row['P_drift'] = P_drift
            row['B_total'] = result['B_int'] + P_drift

        return pd.DataFrame([row])

    def _compute_integrated_score_core(
        self,
        z500_anom: xr.DataArray,
        onset_time_idx: int,
        n_days: int,
        region_lon_min: float,
        region_lon_max: float,
        region_lat_min: float,
        region_lat_max: float,
        event_mask: Optional[np.ndarray] = None,
        blocked_mask: Optional[xr.DataArray] = None,
    ) -> Dict:
        """
        Unified core for computing the integrated anomaly score.

        If *event_mask* is provided (preferred), uses ANO event tracking.
        Otherwise falls back to raw *blocked_mask* with overlap-based tracking.

        Returns a dict with B_int = 0.0 when no qualifying block is found or
        the block dissipates before the integration window ends.
        """
        _ZERO_RESULT = {
            'B_int': 0.0,
            'onset_lon': np.nan,
            'onset_lat': np.nan,
            'n_days': n_days,
            'tracked_masks': [],
            'daily_contributions': [],
        }

        lats = z500_anom.lat.values
        lons = z500_anom.lon.values

        # --- Path A: event_mask from identify_blocking_events (preferred) ------
        if event_mask is not None:
            result = select_event_at_onset(
                event_mask=event_mask,
                lats=lats,
                lons=lons,
                onset_time_idx=onset_time_idx,
                region_lon_min=region_lon_min,
                region_lon_max=region_lon_max,
                region_lat_min=region_lat_min,
                region_lat_max=region_lat_max,
                n_days=n_days,
            )
            if result is None:
                return _ZERO_RESULT

            event_id, onset_info = result

            # Build per-timestep masks from the event_mask
            tracked_masks = []
            for offset in range(n_days):
                t = onset_time_idx + offset
                tracked_masks.append(event_mask[t] == event_id)

            B_int, daily_contributions = _integrate_over_masks(
                z500_anom, tracked_masks, onset_time_idx,
            )

            return {
                'B_int': B_int,
                'onset_lon': onset_info['centroid_lon'],
                'onset_lat': onset_info['centroid_lat'],
                'n_days': n_days,
                'tracked_masks': tracked_masks,
                'daily_contributions': daily_contributions,
            }

        # --- Path B: raw blocked_mask fallback ---------------------------------
        if blocked_mask is None:
            raise ValueError(
                "Either event_mask or blocked_mask must be provided."
            )

        selection = select_block_at_onset(
            blocked_mask=blocked_mask,
            onset_time_idx=onset_time_idx,
            region_lon_min=region_lon_min,
            region_lon_max=region_lon_max,
            region_lat_min=region_lat_min,
            region_lat_max=region_lat_max,
        )
        if selection is None:
            return _ZERO_RESULT

        initial_mask, onset_info = selection

        tracked_masks = track_block_through_time(
            blocked_mask=blocked_mask,
            onset_time_idx=onset_time_idx,
            n_days=n_days,
            initial_component_mask=initial_mask,
        )
        if tracked_masks is None:
            return _ZERO_RESULT

        B_int, daily_contributions = _integrate_over_masks(
            z500_anom, tracked_masks, onset_time_idx,
        )

        return {
            'B_int': B_int,
            'onset_lon': onset_info['centroid_lon'],
            'onset_lat': onset_info['centroid_lat'],
            'n_days': n_days,
            'tracked_masks': tracked_masks,
            'daily_contributions': daily_contributions,
        }

    def _compute_drift_penalty_from_masks(
        self,
        z500_anom: xr.DataArray,
        tracked_masks: List[np.ndarray],
        onset_time_idx: int,
    ) -> float:
        """
        Compute P_drift = -gamma * sum((lon - mean_lon)^2) from tracked masks.
        """
        lats = z500_anom.lat.values
        lons = z500_anom.lon.values
        weights_lat = np.cos(np.deg2rad(lats))

        W_grid = np.broadcast_to(weights_lat[:, None], (len(lats), len(lons)))
        LON_grid, _ = np.meshgrid(lons, lats)

        lambda_c_list = []

        for mask in tracked_masks:
            mask_bool = mask > 0
            w_i = W_grid[mask_bool]
            lam_i = LON_grid[mask_bool]

            denom = np.sum(w_i)
            if denom == 0:
                lambda_c = np.nan
            else:
                lambda_c = np.sum(w_i * lam_i) / denom

            lambda_c_list.append(lambda_c)

        lambda_c_arr = np.array(lambda_c_list)

        # Compute event-mean longitude centroid
        lambda_mean = np.nanmean(lambda_c_arr)

        # Compute drift penalty using spread around the mean
        valid_mask = ~np.isnan(lambda_c_arr)
        if np.sum(valid_mask) > 0:
            diffs = lambda_c_arr[valid_mask] - lambda_mean
            P_drift = -self.gamma * np.sum(diffs**2)
        else:
            P_drift = 0.0

        return P_drift

    def get_score_columns(self) -> List[str]:
        """Return score column names based on mode and drift penalty setting."""
        if self.mode == "auto":
            if self.use_drift_penalty:
                return ["P_block", "P_drift", "P_total"]
            else:
                return ["P_block"]
        else:  # onset mode
            if self.use_drift_penalty:
                return ["B_int", "P_drift", "B_total"]
            else:
                return ["B_int"]

    def get_primary_score_column(self) -> str:
        """Return primary score column name."""
        if self.mode == "auto":
            return "P_total" if self.use_drift_penalty else "P_block"
        else:
            return "B_total" if self.use_drift_penalty else "B_int"


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def compute_blocking_scores(
    z500: xr.DataArray,
    event_info: Dict,
    region: str = "Eurasia",
    gamma: float = 5.0,
    integration_days: int = 5,
    region_lon_min: Optional[float] = None,
    region_lon_max: Optional[float] = None,
    region_lat_min: Optional[float] = None,
    region_lat_max: Optional[float] = None,
) -> pd.DataFrame:
    """
    Compute blocking scores using auto mode with drift penalty.

    This function provides backward compatibility with the original
    compute_blocking_scores interface from DriftPenalizedScorer.

    Parameters
    ----------
    z500 : xr.DataArray
        Input geopotential height data (time, lat, lon).
    event_info : Dict
        Event information dictionary from blocking detection.
    region : str, optional
        Name of predefined region ("Eurasia" or "NorthAtlantic").
        Used only if explicit bounds are not provided.
    gamma : float, optional
        Drift penalty constant. Default: 5.0.
    integration_days : int or None, optional
        Number of days to integrate from the start of each event.
        Default: 5. If None, integrate the entire event duration.
    region_lon_min, region_lon_max : float, optional
        Explicit longitude bounds. Override region preset if provided.
    region_lat_min, region_lat_max : float, optional
        Explicit latitude bounds. Override region preset if provided.

    Returns
    -------
    pd.DataFrame
        Scored events sorted by P_total descending.
    """
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

    scorer = ANOScorer(
        mode="auto",
        use_drift_penalty=True,
        gamma=gamma,
        integration_days=integration_days,
        region_lon_min=lon_min,
        region_lon_max=lon_max,
        region_lat_min=lat_min,
        region_lat_max=lat_max,
    )
    return scorer.compute_event_scores(z500=z500, event_info=event_info)


def compute_integrated_score(
    z500_anom: xr.DataArray,
    blocked_mask: Optional[xr.DataArray] = None,
    onset_time_idx: int = 0,
    region_lon_min: float = -60.0,
    region_lon_max: float = 0.0,
    region_lat_min: float = 55.0,
    region_lat_max: float = 75.0,
    n_days: int = 5,
    event_mask: Optional[np.ndarray] = None,
) -> Dict:
    """
    Convenience function to compute integrated anomaly score directly.

    Provides backward compatibility with the original compute_integrated_score
    interface from IntegratedScorer.

    Parameters
    ----------
    z500_anom : xr.DataArray
        Z500 anomaly data with dimensions (time, lat, lon).
    blocked_mask : xr.DataArray, optional
        Binary blocking mask with dimensions (time, lat, lon).
        Used as fallback when *event_mask* is not provided.
    onset_time_idx : int
        Time index of blocking onset.
    region_lon_min, region_lon_max : float, optional
        Longitude bounds for target ROI. Default: Atlantic (-60-0 E).
    region_lat_min, region_lat_max : float, optional
        Latitude bounds for target ROI. Default: Atlantic (55-75 N).
    n_days : int, optional
        Number of days to integrate. Default: 5.
    event_mask : np.ndarray, optional
        3-D array (time, lat, lon) of integer event IDs from
        identify_blocking_events. Preferred over blocked_mask.

    Returns
    -------
    result : dict
        Dictionary containing:
        - 'B_int': Integrated anomaly score (0.0 if no qualifying block)
        - 'onset_time_idx': Input onset time index
        - 'onset_lon': Longitude of block centroid at onset (NaN if none)
        - 'onset_lat': Latitude of block centroid at onset (NaN if none)
        - 'n_days': Number of days integrated
        - 'tracked_masks': List of component masks for each day
        - 'daily_contributions': List of daily spatial integrals
    """
    scorer = ANOScorer(
        mode="onset",
        onset_time_idx=onset_time_idx,
        n_days=n_days,
        use_drift_penalty=False,
        region_lon_min=region_lon_min,
        region_lon_max=region_lon_max,
        region_lat_min=region_lat_min,
        region_lat_max=region_lat_max,
    )

    event_info = {'z500_anom': z500_anom}
    if event_mask is not None:
        event_info['event_mask'] = event_mask
    if blocked_mask is not None:
        event_info['blocked_mask'] = blocked_mask

    result = scorer._compute_integrated_score_core(
        z500_anom=z500_anom,
        onset_time_idx=onset_time_idx,
        n_days=n_days,
        region_lon_min=region_lon_min,
        region_lon_max=region_lon_max,
        region_lat_min=region_lat_min,
        region_lat_max=region_lat_max,
        event_mask=event_mask,
        blocked_mask=blocked_mask,
    )
    result['onset_time_idx'] = onset_time_idx
    return result
