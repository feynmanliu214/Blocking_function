# Blocking Scoring Reference (Skill)

This file is a compact, task-focused guide for choosing and calling scorers.

## Use This Scorer

| Need | Scorer | Primary Output |
|------|--------|----------------|
| Event-based blocking score with optional drift penalty | `ANOScorer` | `P_total` (auto) or `B_int`/`B_total` (onset) |
| Event-based verification RMSE at onset centroid | `RMSEScorer` | `rmse` |
| Gridpoint persistence mean blocked area | `GridpointPersistenceScorer` | `blocked_pct` |
| Gridpoint strict-window max intensity | `GridpointIntensityScorer` | `max_intensity` |

## Gridpoint Scorers (Current Split)

### Mean Blocked Area

- Module: `forecast_analysis/scoring/gridpoint/persistence_scorer.py`
- Class: `GridpointPersistenceScorer`
- Convenience function: `compute_gridpoint_blocking_score(...)`
- Methods:
  - `compute_score(...)` from raw `z500`
  - `compute_score_from_anomalies(...)` from `z500_anom` + `threshold_90`
- Returns: mean area-weighted blocked percentage over the scoring window (`blocked_pct`, 0-100)

Minimal usage:

```python
from forecast_analysis.scoring.gridpoint import GridpointPersistenceScorer

scorer = GridpointPersistenceScorer(min_persistence=5)
blocked_pct = scorer.compute_score(
    z500=z500,
    start_time="2000-01-01",
    duration_days=10,
    region="Eurasia",
)
```

### Strict-Window Max Intensity

- Module: `forecast_analysis/scoring/gridpoint/intensity_scorer.py`
- Class: `GridpointIntensityScorer`
- Convenience function: `compute_gridpoint_intensity_score(...)`
- Methods:
  - `compute_intensity_score(...)` from raw `z500`
  - `compute_intensity_score_from_anomalies(...)` from `z500_anom` + `threshold_90`
- Returns: max window-mean anomaly over blocked grid points in the window (`max_intensity`)
  using strict per-window exceedance with `window_days = min_persistence`.
- `fallback_to_nonblocked` controls the blocking condition:
  - `false` (default): only blocked grid points contribute; 0 if none exist.
  - `true`: fall back to regional max anomaly if no blocked points in a window.
  - `"always"`: always use regional max anomaly, skip blocking detection entirely (most efficient).

Minimal usage:

```python
from forecast_analysis.scoring.gridpoint import GridpointIntensityScorer

scorer = GridpointIntensityScorer(min_persistence=5)

# Blocked-only (default)
score = scorer.compute_intensity_score(
    z500=z500,
    start_time="2000-01-01",
    duration_days=10,
    region="Eurasia",
    fallback_to_nonblocked=False,
)

# Fallback to regional max when no blocked points
score = scorer.compute_intensity_score(z500=z500, ..., fallback_to_nonblocked=True)

# Always use regional max (ignores blocking, fastest)
score = scorer.compute_intensity_score(z500=z500, ..., fallback_to_nonblocked="always")
```

## Registry and Factory Notes

- `get_scorer()` / `SCORER_REGISTRY` currently include:
  - `ANOScorer`
  - `GridpointPersistenceScorer`
  - `GridpointIntensityScorer`
- `GridpointIntensityScorer` is available via:
  - `forecast_analysis.scoring.gridpoint.GridpointIntensityScorer`
  - `forecast_analysis.scoring.gridpoint.compute_gridpoint_intensity_score`

## Practical Reminders

- `min_persistence` for gridpoint scorers must be `>= 5`.
- Do not pass `running_mean_days` for `GridpointIntensityScorer`; it was removed.
- For anomaly-based methods, pass monthly thresholds as `threshold_90`.
- Use preset regions (`"Eurasia"`, `"NorthAtlantic"`) or explicit lon/lat bounds.
- Source-of-truth implementation files:
  - `forecast_analysis/scoring/gridpoint/persistence_scorer.py`
  - `forecast_analysis/scoring/gridpoint/intensity_scorer.py`
  - `forecast_analysis/scoring/__init__.py`
