#!/usr/bin/env python3
"""
Score Aggregation and Ranking Functions.

This module provides functions to aggregate blocking scores across
ensemble members or lead times and rank them.

Author: AI-RES Project
"""

from typing import Dict, Optional, Union

import pandas as pd

from .base import BlockingScorer
from .ano import ANOScorer, compute_blocking_scores


def compute_member_scores(
    results: Dict[str, Dict],
    scorer: Optional[BlockingScorer] = None,
    region_lon_min: float = 30.0,
    region_lon_max: float = 100.0,
    region_lat_min: float = 55.0,
    region_lat_max: float = 75.0,
    gamma: float = 5.0
) -> pd.DataFrame:
    """
    Compute blocking scores for all forecast members/lead times.

    Parameters
    ----------
    results : dict
        Results dictionary from analyze_ensemble_forecast() or
        analyze_deterministic_forecast().
    scorer : BlockingScorer, optional
        Scorer instance to use. If None, uses ANOScorer with auto mode and drift penalty.
    region_lon_min, region_lon_max : float, optional
        Longitude bounds for scoring region. Default: 30-100 E (Eurasia).
    region_lat_min, region_lat_max : float, optional
        Latitude bounds for scoring region. Default: 55-75 N.
    gamma : float, optional
        Drift penalty constant (used if scorer is None). Default: 5.0.

    Returns
    -------
    df_scores : pd.DataFrame
        DataFrame with columns:
        - member: Member/lead time name
        - n_events: Number of regional blocking events
        - sum_P_total: Sum of all event scores
        - sum_P_block: Sum of blocking components
        - sum_P_drift: Sum of drift penalties
        - max_P_total: Maximum single event score
    """
    # Use default scorer if none provided
    if scorer is None:
        scorer = ANOScorer(
            mode="auto",
            use_drift_penalty=True,
            gamma=gamma,
            region_lon_min=region_lon_min,
            region_lon_max=region_lon_max,
            region_lat_min=region_lat_min,
            region_lat_max=region_lat_max,
        )

    primary_col = scorer.get_primary_score_column()
    score_cols = scorer.get_score_columns()

    member_scores = []

    for member_name, result in results.items():
        if 'error' in result:
            continue

        # Check for required keys - support both 'event_info' and 'stats' keys
        event_info_key = 'event_info' if 'event_info' in result else 'stats'
        if event_info_key not in result or 'z500' not in result:
            print(f"Warning: Skipping {member_name} - missing required keys")
            continue

        try:
            # Compute regional blocking scores
            df_events = scorer.compute_event_scores(
                z500=result['z500'],
                event_info=result[event_info_key],
                region_lon_min=region_lon_min,
                region_lon_max=region_lon_max,
                region_lat_min=region_lat_min,
                region_lat_max=region_lat_max
            )

            # Build score summary
            score_summary = {
                'member': member_name,
                'n_events': len(df_events),
            }

            # Add aggregations for each score column
            for col in score_cols:
                if col in df_events.columns:
                    score_summary[f'sum_{col}'] = df_events[col].sum() if len(df_events) > 0 else 0
                    score_summary[f'max_{col}'] = df_events[col].max() if len(df_events) > 0 else 0

            member_scores.append(score_summary)

        except Exception as e:
            print(f"Warning: Error computing scores for {member_name}: {e}")
            continue

    # Return DataFrame with proper columns even if empty
    if not member_scores:
        columns = ['member', 'n_events']
        for col in score_cols:
            columns.extend([f'sum_{col}', f'max_{col}'])
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(member_scores)


def rank_ensemble_scores(
    results: Dict[str, Dict],
    threshold_score: Optional[float] = None,
    scorer: Optional[BlockingScorer] = None,
    region_lon_min: float = 30.0,
    region_lon_max: float = 100.0,
    region_lat_min: float = 55.0,
    region_lat_max: float = 75.0,
    gamma: float = 5.0,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Compute, rank, and report blocking scores for ensemble members.

    Parameters
    ----------
    results : dict
        Results dictionary from analyze_ensemble_forecast().
    threshold_score : float, optional
        If provided, highlight members with primary score sum above threshold.
    scorer : BlockingScorer, optional
        Scorer instance to use. If None, uses ANOScorer with auto mode and drift penalty.
    region_lon_min, region_lon_max : float, optional
        Longitude bounds for scoring region. Default: 30-100 E (Eurasia).
    region_lat_min, region_lat_max : float, optional
        Latitude bounds for scoring region. Default: 55-75 N.
    gamma : float, optional
        Drift penalty constant (used if scorer is None). Default: 5.0.
    verbose : bool, optional
        Print ranking tables. Default: True.

    Returns
    -------
    df_ranked : pd.DataFrame
        DataFrame with ranked members including:
        - rank: Ranking (1 = highest score)
        - member: Member name
        - n_events: Number of events
        - Score columns and aggregations
    """
    # Use default scorer if none provided
    if scorer is None:
        scorer = ANOScorer(
            mode="auto",
            use_drift_penalty=True,
            gamma=gamma,
            region_lon_min=region_lon_min,
            region_lon_max=region_lon_max,
            region_lat_min=region_lat_min,
            region_lat_max=region_lat_max,
        )

    primary_col = scorer.get_primary_score_column()
    sum_primary_col = f'sum_{primary_col}'

    if verbose:
        print("=" * 70)
        print(f"COMPUTING BLOCKING SCORES ({scorer.name})")
        print(f"Region: {region_lon_min} -{region_lon_max} E, {region_lat_min} -{region_lat_max} N")
        print("=" * 70)

    # Compute scores for all members
    df_all = compute_member_scores(
        results,
        scorer=scorer,
        region_lon_min=region_lon_min,
        region_lon_max=region_lon_max,
        region_lat_min=region_lat_min,
        region_lat_max=region_lat_max,
        gamma=gamma
    )

    # Handle empty DataFrame
    if len(df_all) == 0:
        if verbose:
            print("\n No valid members to rank. All members may have errors.")
        columns = ['rank', 'member', 'n_events', sum_primary_col]
        return pd.DataFrame(columns=columns)

    # Sort by primary score sum (descending)
    df_all = df_all.sort_values(sum_primary_col, ascending=False).reset_index(drop=True)
    df_all['rank'] = df_all.index + 1

    # Reorder columns to put rank first
    cols = df_all.columns.tolist()
    cols.remove('rank')
    df_all = df_all[['rank'] + cols]

    if verbose:
        # Display full ranking
        print("\n" + "=" * 90)
        print("ALL MEMBERS - FULL RANKING (Highest -> Lowest)")
        print("=" * 90)
        print(df_all.to_string(index=False))
        print("=" * 90)

        # Display selected members if threshold provided
        if threshold_score is not None:
            df_selected = df_all[df_all[sum_primary_col] > threshold_score]
            print("\n" + "=" * 90)
            print(f"SELECTED MEMBERS ({sum_primary_col} > {threshold_score:.2f})")
            print("=" * 90)
            if len(df_selected) == 0:
                print(f"\n No members above threshold.\n")
            else:
                print(df_selected.to_string(index=False))
                print(f"\n Selected {len(df_selected)} out of {len(df_all)} members")

        # Summary statistics
        print("\n" + "=" * 90)
        print("SUMMARY")
        print("=" * 90)
        print(f"  Max score:   {df_all[sum_primary_col].max():.2f}  ({df_all.iloc[0]['member']})")
        print(f"  Min score:   {df_all[sum_primary_col].min():.2f}  ({df_all.iloc[-1]['member']})")
        print(f"  Mean:        {df_all[sum_primary_col].mean():.2f}")
        if threshold_score is not None:
            print(f"  Threshold:   {threshold_score:.2f}")
        print("=" * 90)

    return df_all
