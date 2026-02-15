#!/usr/bin/env python3
"""
RMSE-based Blocking Scorer.

This scorer computes the RMSE between emulator and truth Z500 anomalies
at a fixed 3x3 grid patch centered on the truth blocking onset centroid.

The score evaluates forecast skill by comparing:
- Emulator (Pangu-PlaSim) Z500 anomalies
- Truth (PlaSim) Z500 anomalies

at the same spatial location (determined by truth) and time (blocking onset).

Author: AI-RES Project
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from ..base import BlockingScorer
from .utils import (
    compute_blocking_centroid,
    compute_unweighted_centroid,
    extract_3x3_patch,
)


def compute_rmse(emulator_patch: np.ndarray, truth_patch: np.ndarray) -> float:
    """
    Compute RMSE between emulator and truth patches.

    Parameters
    ----------
    emulator_patch : np.ndarray
        Flattened emulator Z500 anomaly values (9 values for 3x3).
    truth_patch : np.ndarray
        Flattened truth Z500 anomaly values (9 values for 3x3).

    Returns
    -------
    rmse : float
        Root Mean Square Error in the same units as the input (typically meters).
    """
    diff = emulator_patch - truth_patch
    rmse = np.sqrt(np.mean(diff ** 2))
    return float(rmse)


def compute_onset_centroid_from_event_info(
    event_info: Dict,
    event_id: int = 1
) -> Tuple[float, float, int]:
    """
    Compute the unweighted centroid at blocking onset for a specific event.

    Parameters
    ----------
    event_info : Dict
        Event information dictionary from blocking detection.
        Must contain 'blocked_mask' and event timing information.
    event_id : int, optional
        Which event to use (1-indexed). Default: 1 (first event).

    Returns
    -------
    lon_centroid : float
        Longitude of onset centroid.
    lat_centroid : float
        Latitude of onset centroid.
    onset_time_idx : int
        Time index of the blocking onset.

    Raises
    ------
    ValueError
        If the specified event does not exist.
    """
    blocked_mask = event_info['blocked_mask']

    # Find the first time with blocking (onset)
    # For simplicity, we use the first timestep where any blocking occurs
    blocking_exists = blocked_mask.sum(dim=['lat', 'lon']) > 0
    onset_indices = np.where(blocking_exists.values)[0]

    if len(onset_indices) == 0:
        raise ValueError("No blocking events found in event_info")

    # Use the onset of the specified event
    # For now, we use the first blocking onset
    onset_time_idx = int(onset_indices[0])

    lon_centroid, lat_centroid = compute_unweighted_centroid(
        blocked_mask, onset_time_idx
    )

    return lon_centroid, lat_centroid, onset_time_idx


class RMSEScorer(BlockingScorer):
    """
    RMSE-based blocking scorer comparing emulator vs truth Z500 anomalies.

    This scorer evaluates forecast skill by computing the RMSE between
    emulator and truth Z500 anomalies at a 3x3 grid patch centered on
    the truth blocking onset centroid.

    The truth blocking location and onset time are fixed inputs - the emulator
    does not need to identify or track blocking events.

    Parameters
    ----------
    truth_z500_anom : xr.DataArray
        Truth (PlaSim) Z500 anomaly data with dimensions (time, lat, lon).
    truth_onset_lon : float, optional
        Longitude of truth blocking onset centroid. If None, computed from
        truth_blocked_mask.
    truth_onset_lat : float, optional
        Latitude of truth blocking onset centroid. If None, computed from
        truth_blocked_mask.
    truth_onset_time_idx : int, optional
        Time index of truth blocking onset. If None, computed from
        truth_blocked_mask.
    truth_blocked_mask : xr.DataArray, optional
        Truth blocking mask, used to compute centroid if not provided directly.

    Notes
    -----
    Either provide (truth_onset_lon, truth_onset_lat, truth_onset_time_idx)
    directly, OR provide truth_blocked_mask to compute them automatically.

    This scorer sets requires_blocking_detection = False because it only needs
    Z500 anomalies from the emulator, not blocking event detection.
    """

    name = "RMSEScorer"
    description = "RMSE between emulator and truth Z500 anomalies at onset centroid"
    requires_blocking_detection = False

    def __init__(
        self,
        truth_z500_anom: xr.DataArray,
        truth_onset_lon: Optional[float] = None,
        truth_onset_lat: Optional[float] = None,
        truth_onset_time_idx: Optional[int] = None,
        truth_blocked_mask: Optional[xr.DataArray] = None
    ):
        self.truth_z500_anom = truth_z500_anom

        # Determine onset location and time
        if all(v is not None for v in [truth_onset_lon, truth_onset_lat, truth_onset_time_idx]):
            self.truth_onset_lon = truth_onset_lon
            self.truth_onset_lat = truth_onset_lat
            self.truth_onset_time_idx = truth_onset_time_idx
        elif truth_blocked_mask is not None:
            # Compute from blocking mask
            event_info = {'blocked_mask': truth_blocked_mask}
            self.truth_onset_lon, self.truth_onset_lat, self.truth_onset_time_idx = \
                compute_onset_centroid_from_event_info(event_info)
        else:
            raise ValueError(
                "Must provide either (truth_onset_lon, truth_onset_lat, truth_onset_time_idx) "
                "or truth_blocked_mask to compute them."
            )

        # Pre-extract the truth patch for efficiency
        self.truth_patch = extract_3x3_patch(
            self.truth_z500_anom,
            self.truth_onset_lon,
            self.truth_onset_lat,
            self.truth_onset_time_idx
        )

    def compute_event_scores(
        self,
        z500: xr.DataArray,
        event_info: Dict,
        region_lon_min: float = 30.0,
        region_lon_max: float = 100.0,
        region_lat_min: float = 55.0,
        region_lat_max: float = 75.0,
        **kwargs
    ) -> pd.DataFrame:
        """
        Compute RMSE score for emulator forecast against truth.

        This method extracts the emulator Z500 anomalies at the same location
        and time as the truth (provided during initialization) and computes
        the RMSE.

        Parameters
        ----------
        z500 : xr.DataArray
            Emulator Z500 data (not directly used, anomalies come from event_info).
        event_info : Dict
            Event information dictionary containing 'z500_anom' (emulator anomalies).
        region_lon_min, region_lon_max : float
            Longitude bounds (not used for RMSE, kept for interface compatibility).
        region_lat_min, region_lat_max : float
            Latitude bounds (not used for RMSE, kept for interface compatibility).

        Returns
        -------
        pd.DataFrame
            Single-row DataFrame with columns:
            - truth_onset_lon: Longitude of truth centroid
            - truth_onset_lat: Latitude of truth centroid
            - truth_onset_time_idx: Time index of truth onset
            - rmse: Root Mean Square Error (meters)
        """
        emulator_z500_anom = event_info['z500_anom']

        # Extract emulator patch at the SAME location and time as truth
        try:
            emulator_patch = extract_3x3_patch(
                emulator_z500_anom,
                self.truth_onset_lon,
                self.truth_onset_lat,
                self.truth_onset_time_idx
            )
        except (ValueError, IndexError) as e:
            # Handle case where time index or location is out of bounds
            return pd.DataFrame([{
                'truth_onset_lon': self.truth_onset_lon,
                'truth_onset_lat': self.truth_onset_lat,
                'truth_onset_time_idx': self.truth_onset_time_idx,
                'rmse': np.nan,
                'error': str(e)
            }])

        # Compute RMSE
        rmse = compute_rmse(emulator_patch, self.truth_patch)

        return pd.DataFrame([{
            'truth_onset_lon': self.truth_onset_lon,
            'truth_onset_lat': self.truth_onset_lat,
            'truth_onset_time_idx': self.truth_onset_time_idx,
            'rmse': rmse
        }])

    def get_score_columns(self) -> List[str]:
        """Return score column names."""
        return ['rmse']

    def get_primary_score_column(self) -> str:
        """Return primary score column name."""
        return 'rmse'


def compute_rmse_score(
    emulator_z500_anom: xr.DataArray,
    truth_z500_anom: xr.DataArray,
    truth_onset_lon: float,
    truth_onset_lat: float,
    truth_onset_time_idx: int
) -> Dict:
    """
    Convenience function to compute RMSE score directly.

    This function provides a simple interface for computing the RMSE between
    emulator and truth Z500 anomalies without using the BlockingScorer framework.

    Parameters
    ----------
    emulator_z500_anom : xr.DataArray
        Emulator (Pangu-PlaSim) Z500 anomaly data with dimensions (time, lat, lon).
    truth_z500_anom : xr.DataArray
        Truth (PlaSim) Z500 anomaly data with dimensions (time, lat, lon).
    truth_onset_lon : float
        Longitude of truth blocking onset centroid (degrees).
    truth_onset_lat : float
        Latitude of truth blocking onset centroid (degrees).
    truth_onset_time_idx : int
        Time index of truth blocking onset.

    Returns
    -------
    result : dict
        Dictionary containing:
        - 'rmse': RMSE value (meters)
        - 'truth_onset_lon': Input longitude
        - 'truth_onset_lat': Input latitude
        - 'truth_onset_time_idx': Input time index
        - 'truth_patch': Truth Z500 anomaly values (9 values)
        - 'emulator_patch': Emulator Z500 anomaly values (9 values)
    """
    # Extract truth patch
    truth_patch = extract_3x3_patch(
        truth_z500_anom,
        truth_onset_lon,
        truth_onset_lat,
        truth_onset_time_idx
    )

    # Extract emulator patch at the SAME location and time
    emulator_patch = extract_3x3_patch(
        emulator_z500_anom,
        truth_onset_lon,
        truth_onset_lat,
        truth_onset_time_idx
    )

    # Compute RMSE
    rmse = compute_rmse(emulator_patch, truth_patch)

    return {
        'rmse': rmse,
        'truth_onset_lon': truth_onset_lon,
        'truth_onset_lat': truth_onset_lat,
        'truth_onset_time_idx': truth_onset_time_idx,
        'truth_patch': truth_patch,
        'emulator_patch': emulator_patch
    }
