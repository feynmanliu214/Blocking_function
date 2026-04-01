#!/usr/bin/env python3
"""
Heatwave Scorers Subpackage.

This subpackage provides heatwave-related scoring methods:

- HeatwaveMeanScorer: Spatiotemporal mean of surface temperature over a regional box

Usage:
    from forecast_analysis.scoring.heatwave import HeatwaveMeanScorer

    scorer = HeatwaveMeanScorer(n_days=7)
    score = scorer.compute_score_from_field(field_data, onset_time_idx=5, region_bounds={...})

Author: AI-RES Project
"""

from .mean_scorer import HeatwaveMeanScorer

__all__ = [
    "HeatwaveMeanScorer",
]
