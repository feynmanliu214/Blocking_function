# Return Curve Fixed-Window Return-Period Design

## Summary

Update the AI-RES return-curve plotting script so the AI+RES x-axis return period is computed from the fixed experiment window exceedance probability directly:

- `RP_years = 1 / p_window`

This removes the current season-rate conversion (`season_days / target_duration`), which was interpreting the AI-RES window probability as a DJF-wide rate and is considered mathematically wrong for the current experiment workflow.

## Approved Decisions

- Keep AI+RES x-axis as `return period (years)`.
- Remove season-scaled conversion entirely (no legacy option).
- Keep DNS overlay plotting behavior unchanged.
- Keep "return period" wording for backward compatibility.
- Add `12/24–12/30` in the title only.
- Hard break is acceptable for obsolete CLI options (`--season_days`, `--target_duration`).

## Problem Statement

The current plotting script estimates a weighted AI-RES exceedance probability `p` and converts it to years using:

- `windows_per_season = season_days / target_duration`
- `RP_years = 1 / (p * windows_per_season)`

This assumes the event probability is a per-window rate that should be scaled across DJF. For the current AI-RES experiment, the target window is a specific fixed annual window (`12/24–12/30`), so the correct interpretation is one opportunity per year:

- `RP_years = 1 / p_window`

## Scope

### In Scope

- `script/plot_return_curve_blocking_area_pct.py`
  - Remove AI+RES season-rate conversion.
  - Compute AI+RES x-values as `1/p`.
  - Remove obsolete CLI args/help text for `--season_days` and `--target_duration`.
  - Update logging to describe fixed-window annual conversion.
  - Add `12/24–12/30` to plot title.
- `.skills/plot/plot_return_curve.md`
  - Update runbook documentation to match the new fixed-window return-period interpretation.
  - Remove references implying AI+RES x-axis depends on `season_days/target_duration`.

### Out of Scope

- Recomputing or reformatting DNS ground-truth NPZ files.
- Changing DNS overlay x-values or visualization style.
- Renaming the script or changing "return period" terminology globally.

## Design Details

### AI+RES Return-Period Formula

Current:

- `rp_exp = 1 / (p * (season_days / target_duration))`

New:

- `rp_exp = 1 / p`

Interpretation:

- `p` is the estimated exceedance probability for the experiment's fixed window (`12/24–12/30`) in a given year.
- The reciprocal therefore has units of years because the window occurs once per year.

### Title and Labels

- Keep x-axis label wording compatible (still "return period (years)").
- Keep existing general naming ("return-period curve").
- Update the plot title to explicitly include the fixed window, e.g. `12/24–12/30 window`.

### CLI and Backward Compatibility

- Remove `--season_days` and `--target_duration` from the script CLI.
- Remove code paths and logs that reference the removed conversion.
- This is an intentional breaking change because the prior behavior is being removed, not deprecated.

## Risks and Mitigations

### Risk: Existing shell scripts still pass removed flags

Mitigation:

- Update the runbook examples and flag lists.
- Document the breaking change in help text/runbook wording.

### Risk: Users confuse DNS overlay window with AI+RES x-axis interpretation

Mitigation:

- Keep and clarify documentation that DNS overlay file selection is separate from AI+RES x-axis computation.
- Explicitly note that AI+RES now uses fixed-window annual return period (`1/p_window`).

## Validation Plan (Post-Implementation)

- Run the plotting script on a known experiment and confirm logs show fixed-window conversion (`RP = 1/p`) with no `season_days/target_duration` message.
- Verify the output figure title includes `12/24–12/30`.
- Confirm DNS overlay still renders.
- Spot-check a few points: new AI+RES x-values should equal old x-values multiplied by the removed factor (`season_days / target_duration`) from previous behavior.

## File Targets (Expected)

- `script/plot_return_curve_blocking_area_pct.py`
- `.skills/plot/plot_return_curve.md`
- Optional test/doc fixtures if any exist for script CLI help snapshots (none identified yet)
