#!/usr/bin/env python3
"""
Abstract Base Class for Blocking Scorers.

This module defines the interface that all blocking scorers must implement,
enabling extensible scoring methods for blocking event analysis.

To create a new scoring method:
1. Inherit from BlockingScorer
2. Implement compute_event_scores()
3. Optionally override get_score_columns() if you have custom columns

Example:
    class MyCustomScorer(BlockingScorer):
        def compute_event_scores(self, z500, event_info, **kwargs):
            # Your scoring logic here
            return pd.DataFrame(...)

Author: AI-RES Project
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import pandas as pd
import xarray as xr


class BlockingScorer(ABC):
    """
    Abstract base class for blocking event scorers.

    All scoring methods should inherit from this class and implement
    the compute_event_scores method.

    Attributes
    ----------
    name : str
        Human-readable name of the scoring method.
    description : str
        Brief description of how the scoring works.
    requires_blocking_detection : bool
        If True (default), blocking detection (ANO method) must be run on the
        emulator data before scoring. If False, the scorer only needs Z500
        anomalies and can skip the computationally expensive blocking detection.
        Scorers like RMSEScorer that compare against truth at fixed locations
        set this to False.
    requires_anomaly : bool
        If True (default), the scorer operates on anomaly fields (e.g. Z500
        anomalies). If False, the scorer works on raw variable fields (e.g.
        raw surface temperature) and uses ``compute_score_from_field()``
        instead of the event-based ``compute_event_scores()`` pathway.
    required_variable : str
        The climate variable this scorer expects (e.g. ``"z500"``, ``"tas"``).
        Default ``"z500"`` for backward compatibility with blocking scorers.
    allowed_regions : tuple or None
        Tuple of region name strings that this scorer is valid for.
        ``None`` means the scorer can be used with any region.
    """

    name: str = "BaseScorer"
    description: str = "Abstract base scorer"
    requires_blocking_detection: bool = True
    requires_anomaly: bool = True
    required_variable: str = "z500"
    allowed_regions: tuple = None

    @abstractmethod
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
        Compute scores for blocking events within a specified region.

        Parameters
        ----------
        z500 : xr.DataArray
            Input geopotential height data with dimensions (time, lat, lon).
        event_info : Dict
            Event information dictionary from blocking detection.
            Must contain at least 'blocked_mask' and 'z500_anom'.
        region_lon_min, region_lon_max : float
            Longitude bounds for the scoring region.
        region_lat_min, region_lat_max : float
            Latitude bounds for the scoring region.
        **kwargs
            Additional scorer-specific parameters.

        Returns
        -------
        pd.DataFrame
            DataFrame containing scored events with at least these columns:
            - region_event_id: Unique identifier for the regional event
            - start_time: Event start timestamp
            - end_time: Event end timestamp
            - duration_days: Event duration in days
            - score: The primary score value (name may vary by scorer)
        """
        pass

    def get_score_columns(self) -> List[str]:
        """
        Return the list of score-related column names produced by this scorer.

        Override this method if your scorer produces different columns
        than the default set.

        Returns
        -------
        List[str]
            Column names for score-related fields.
        """
        return ["score"]

    def get_primary_score_column(self) -> str:
        """
        Return the name of the primary score column used for ranking.

        Override this method if your primary score column has a different name.

        Returns
        -------
        str
            Name of the primary score column.
        """
        return "score"

    def score_from_anomaly(
        self,
        z500_anom: "xr.DataArray",
        event_info: dict,
        region_bounds: dict,
        onset_time_idx: int,
        threshold_90: Optional[dict] = None,
        scorer_params: Optional[dict] = None,
    ) -> float:
        """Compute scalar score from a pre-computed anomaly field.

        scorer_params carries config-defined values that are NOT stored as
        scorer object attributes (e.g. n_days/duration_days for gridpoint
        scorers, fallback_to_nonblocked for GridpointIntensityScorer).
        These must be passed through explicitly because the scorer constructors
        store only their own init-time parameters (e.g. min_persistence).

        Default: raises NotImplementedError.  Anomaly-path scorers must
        override this.  Non-anomaly scorers (HeatwaveMeanScorer) do not
        override it — score_single_member routes them to
        compute_score_from_field() instead.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement score_from_anomaly(). "
            "Use compute_score_from_field() for non-anomaly scorers."
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
