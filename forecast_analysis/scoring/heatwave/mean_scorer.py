#!/usr/bin/env python3
"""
Heatwave Mean Scorer.

This scorer computes the spatiotemporal mean of near-surface temperature
(T_2m / tas) over an L-day window within a regional bounding box.

Unlike blocking scorers, this scorer:
- Does NOT require anomaly fields (works on raw temperature)
- Does NOT require blocking detection
- Uses ``compute_score_from_field()`` instead of event-based scoring

Author: AI-RES Project
"""

import xarray as xr

from ..base import BlockingScorer


class HeatwaveMeanScorer(BlockingScorer):
    """Spatiotemporal mean of T_2m over an L-day regional box.

    Parameters
    ----------
    n_days : int
        Number of days for the averaging window (default: 7).

    Attributes
    ----------
    requires_blocking_detection : bool
        Always ``False``; no blocking mask needed.
    requires_anomaly : bool
        ``False``; operates on raw temperature fields.
    required_variable : str
        ``"tas"`` (near-surface air temperature).
    allowed_regions : tuple
        ``("Chicago", "France")``; regions where this scorer is valid.
    """

    name = "HeatwaveMeanScorer"
    description = "Spatiotemporal mean of T_2m over L-day regional box"
    requires_blocking_detection = False
    requires_anomaly = False
    required_variable = "tas"
    allowed_regions = ("Chicago", "France")

    def __init__(self, n_days: int = 7):
        self.n_days = n_days

    def compute_event_scores(self, z500, event_info, **kwargs):
        """Required by BlockingScorer ABC. Not used for this scorer."""
        raise NotImplementedError(
            "HeatwaveMeanScorer does not use event-based scoring. "
            "Use compute_score_from_field() via compute_res_score() dispatch."
        )

    def compute_score_from_field(
        self,
        field_data,           # xr.DataArray: raw tas, dims (time, lat, lon)
        onset_time_idx: int,
        region_bounds: dict,  # {lon_min, lon_max, lat_min, lat_max}
    ) -> float:
        """Compute A_L(t) = spatiotemporal mean of field over region and n_days.

        Parameters
        ----------
        field_data : xr.DataArray
            Raw near-surface temperature with dimensions ``(time, lat, lon)``.
        onset_time_idx : int
            Starting time index for the averaging window.
        region_bounds : dict
            Bounding box with keys ``lon_min``, ``lon_max``, ``lat_min``,
            ``lat_max``.

        Returns
        -------
        float
            Plain mean temperature in Kelvin.
        """
        window = field_data.isel(time=slice(onset_time_idx, onset_time_idx + self.n_days))
        regional = window.sel(
            lon=slice(region_bounds['lon_min'], region_bounds['lon_max']),
            lat=slice(region_bounds['lat_min'], region_bounds['lat_max']),
        )
        return float(regional.mean().values)
