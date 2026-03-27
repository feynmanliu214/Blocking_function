#!/usr/bin/env python3
"""
Heatwave Mean Scorer.

This scorer computes the spatiotemporal mean of near-surface temperature
(T_2m / tas) over an L-day window within a regional selector.

Unlike blocking scorers, this scorer:
- Does NOT require anomaly fields (works on raw temperature)
- Does NOT require blocking detection
- Uses ``compute_score_from_field()`` instead of event-based scoring

Author: AI-RES Project
"""

import xarray as xr

from ..base import BlockingScorer

KELVIN_OFFSET = 273.15


class HeatwaveMeanScorer(BlockingScorer):
    """Spatiotemporal mean of T_2m over an L-day regional selector.

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
    canonical_region_points = {
        # Keep in sync with AI-RES/RES/regions.json
        "France": {
            "lon": [0.0, 2.8125, 5.625],
            "lat": [48.83524097, 46.04472663, 43.25419467],
        },
        "Chicago": {
            "lon": [270.0, 272.8125, 275.625],
            "lat": [37.67308963, 40.46364818, 43.25419467],
        },
    }

    def __init__(self, n_days: int = 7):
        self.n_days = n_days

    @classmethod
    def resolve_region_points(cls, region: str) -> dict:
        """Return canonical point lists for supported heatwave regions."""
        if region not in cls.canonical_region_points:
            raise ValueError(
                f"Unknown heatwave region '{region}'. "
                f"Known regions: {sorted(cls.canonical_region_points)}"
            )
        points = cls.canonical_region_points[region]
        return {"lon": list(points["lon"]), "lat": list(points["lat"])}

    @classmethod
    def resolve_region_bounds(cls, region: str) -> dict:
        """Return derived bounds for supported heatwave regions."""
        points = cls.resolve_region_points(region)
        return {
            "lon_min": float(min(points["lon"])),
            "lon_max": float(max(points["lon"])),
            "lat_min": float(min(points["lat"])),
            "lat_max": float(max(points["lat"])),
        }

    @staticmethod
    def _select_region(field_data: xr.DataArray, region_selector: dict) -> xr.DataArray:
        """Select either explicit region points or a legacy bounding box."""
        if "lon" in region_selector and "lat" in region_selector:
            return field_data.sel(
                lon=region_selector["lon"],
                lat=region_selector["lat"],
                method="nearest",
            )
        return field_data.sel(
            lon=slice(region_selector["lon_min"], region_selector["lon_max"]),
            lat=slice(region_selector["lat_min"], region_selector["lat_max"]),
        )

    def compute_event_scores(self, z500, event_info, **kwargs):
        """Required by BlockingScorer ABC. Not used for this scorer."""
        raise NotImplementedError(
            "HeatwaveMeanScorer does not use event-based scoring. "
            "Use compute_score_from_field() via score_single_member() in forecast_analysis.scoring.pipeline."
        )

    def compute_score_from_field(
        self,
        field_data,           # xr.DataArray: raw tas, dims (time, lat, lon)
        onset_time_idx: int,
        region_bounds: dict,  # {lon, lat} or {lon_min, lon_max, lat_min, lat_max}
    ) -> float:
        """Compute A_L(t) = spatiotemporal mean of field over region and n_days.

        Parameters
        ----------
        field_data : xr.DataArray
            Raw near-surface temperature with dimensions ``(time, lat, lon)``.
        onset_time_idx : int
            Starting time index for the averaging window.
        region_bounds : dict
            Either explicit region point lists with keys ``lon`` and ``lat``,
            or a legacy bounding box with keys ``lon_min``, ``lon_max``,
            ``lat_min``, ``lat_max``.

        Returns
        -------
        float
            Mean temperature in Celsius.
        """
        window = field_data.isel(time=slice(onset_time_idx, onset_time_idx + self.n_days))
        regional = self._select_region(window, region_bounds)
        spatial_mean = regional.mean(dim=["lon", "lat"])
        return float(spatial_mean.mean(dim="time").values - KELVIN_OFFSET)
