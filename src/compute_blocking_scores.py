#!/usr/bin/env python3
"""
Compute Blocking Scores for Eurasian Winter Events using ANO Method.

This module provides a function to compute blocking scores from the output
of ano_blocking_complete.

Usage in Jupyter Notebook:
    # Cell 1: Run blocking detection
    from ANO_PlaSim import ano_blocking_complete
    blocking_freq, event_info = ano_blocking_complete(z500, ...)

    # Cell 2: Compute scores
    from compute_blocking_scores import compute_blocking_scores
    df = compute_blocking_scores(z500, event_info, ...)

Author: Antigravity
"""

import numpy as np
import xarray as xr
import pandas as pd
from typing import Dict


def create_spatial_weight_mask(ds_coords: xr.DataArray,
                               lat_min: float, lat_max: float,
                               lon_min: float, lon_max: float) -> xr.DataArray:
    """
    Construct the 2D spatial weight mask W(lambda, phi).
    
    W = 1 if lon in [lon_min, lon_max] and lat in [lat_min, lat_max], else 0.
    """
    mask = (
        (ds_coords.lat >= lat_min) & 
        (ds_coords.lat <= lat_max) & 
        (ds_coords.lon >= lon_min) & 
        (ds_coords.lon <= lon_max)
    )
    return mask.astype(float)


def compute_blocking_scores(
    z500: xr.DataArray,
    event_info: Dict,
    region_lon_min: float = 30.0,
    region_lon_max: float = 100.0,
    region_lat_min: float = 55.0,
    region_lat_max: float = 75.0,
    gamma: float = 5.0
) -> pd.DataFrame:
    """
    Compute blocking scores from the output of ano_blocking_complete.
    
    Parameters:
    -----------
    z500 : xr.DataArray
        Input geopotential height data (time, lat, lon).
    event_info : Dict
        The event_info dictionary returned by ano_blocking_complete.
        Must contain keys: 'blocked_mask', 'z500_anom'.
    region_lon_min, region_lon_max : float
        Longitude bounds for the target region (default: 30-100 for Eurasia).
    region_lat_min, region_lat_max : float
        Latitude bounds for the target region (default: 55-75 for Eurasia).
    gamma : float
        Drift penalty constant (default: 5.0).
        The drift penalty is computed as: P_drift = -gamma * sum((lambda_c(k) - lambda_mean)^2)
        where lambda_mean is the event-mean longitude centroid.
        Note: This uses "spread-around-mean" rather than consecutive-day increments.
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing scored events, sorted by P_total descending.
        Columns: region_event_id, start_time, end_time, duration_days,
                 P_block, P_drift, P_total, mean_lon_centroid.
    """
    
    # 1. Construct Regional Mask W(lambda, phi)
    print(f"Constructing regional mask: Lon[{region_lon_min}, {region_lon_max}], Lat[{region_lat_min}, {region_lat_max}]")
    W = create_spatial_weight_mask(z500, region_lat_min, region_lat_max, region_lon_min, region_lon_max)
    
    # 2. Form Regional Blocking Mask I_R
    blocked_mask = event_info["blocked_mask"]  # (time, lat, lon)
    I_R = blocked_mask * W  # (time, lat, lon)
    
    # Handle case where I_R is all zeros
    if I_R.sum() == 0:
        print("No blocking detected in the specified region.")
        return pd.DataFrame(columns=[
            "region_event_id", "start_time", "end_time", "duration_days", 
            "P_block", "P_drift", "P_total", "mean_lon_centroid"
        ])

    # 3. Define Region-Level Events
    print("Identifying regional blocking events...")
    
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
    print(f"Found {n_events} regional blocking events.")
    
    # 4. Compute Scores
    z_anom = event_info["z500_anom"]
    z_plus_prime = xr.where(z_anom > 0, z_anom, 0)  # Z'+ = max(0, Z')
    
    results = []
    
    for i in range(n_events):
        t_start_idx = starts[i]
        t_end_idx = ends[i]  # Exclusive
        
        if t_end_idx <= t_start_idx:
            continue
            
        region_event_id = i + 1
        
        # Time slices
        times_slice = z500.time.isel(time=slice(t_start_idx, t_end_idx))
        t_start = times_slice.values[0]
        t_end = times_slice.values[-1]
        duration = t_end_idx - t_start_idx
        
        # --- Compute P_block ---
        I_R_slice = I_R.isel(time=slice(t_start_idx, t_end_idx))
        Z_plus_slice = z_plus_prime.isel(time=slice(t_start_idx, t_end_idx))
        
        integrand = I_R_slice * Z_plus_slice * weights_lat
        P_block = integrand.sum().item()
        
        # --- Compute P_drift ---
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
        # lambda_mean = (1/N) * sum_{k=0}^{N-1} lambda_c(k)
        lambda_mean = np.nanmean(lambda_c_arr)
        
        # Compute drift penalty using spread around the mean
        # P_drift = -gamma * sum_{k=0}^{N-1} (lambda_c(k) - lambda_mean)^2
        # This replaces the old increment-based penalty: -gamma * sum(delta_lambda^2)
        # Note: gamma may need re-tuning after switching from increment-based to spread-based
        valid_mask = ~np.isnan(lambda_c_arr)
        if np.sum(valid_mask) > 0:
            diffs = lambda_c_arr[valid_mask] - lambda_mean
            P_drift = -gamma * np.sum(diffs**2)
        else:
            P_drift = 0.0
            
        P_total = P_block + P_drift
        mean_lon = lambda_mean  # Already computed above
        
        results.append({
            "region_event_id": region_event_id,
            "start_time": t_start,
            "end_time": t_end,
            "duration_days": duration,
            "P_block": P_block,
            "P_drift": P_drift,
            "P_total": P_total,
            "mean_lon_centroid": mean_lon
        })

    # Sort and return
    if results:
        df = pd.DataFrame(results)
        return df.sort_values(by="P_total", ascending=False)
    else:
        return pd.DataFrame(columns=[
            "region_event_id", "start_time", "end_time", "duration_days", 
            "P_block", "P_drift", "P_total", "mean_lon_centroid"
        ])
