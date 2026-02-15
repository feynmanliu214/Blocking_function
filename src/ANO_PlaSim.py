import numpy as np
import xarray as xr
from typing import Tuple, Optional, Dict, List, Union
from scipy import ndimage
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


# Universal constant
EARTH_RADIUS_KM = 6371.0  # Earth radius in kilometers


def compute_anomalies(z500: xr.DataArray) -> xr.DataArray:
    """
    Compute daily geopotential height anomalies using day-of-year climatology.
    
    Anomalies are computed by subtracting the climatological mean for each 
    specific calendar day (day-of-year), following Woollings et al. (2018):
    
    Z'(t, φ, λ) = Z(t, φ, λ) - Z̄_clim(d(t), φ, λ)
    
    where Z̄_clim(d, φ, λ) is the multi-year average for calendar day d.
    
    Parameters
    ----------
    z500 : xarray.DataArray
        Daily geopotential height field with dimensions (time, lat, lon).
        Units: meters
    
    Returns
    -------
    z500_anom : xarray.DataArray
        Daily anomalies with the same shape as input.
        Units: meters
    
    Notes
    -----
    The day-of-year climatology is computed by:
    1. Extracting the day-of-year (1-366) for each timestep
    2. Grouping all days with the same day-of-year across all years
    3. Computing the mean for each day-of-year at each grid point
    4. Subtracting the appropriate climatological value from each day
    
    This approach accounts for the seasonal cycle and is the standard method
    for blocking detection (Woollings et al., 2018).
    
    Examples
    --------
    >>> z500_anom = compute_anomalies(z500_djf_100years)
    """
    # Extract day-of-year for each timestep
    try:
        dayofyear = z500.time.dt.dayofyear
    except AttributeError:
        raise ValueError("Time coordinate must be datetime-like with .dt.dayofyear accessor")
    
    # Compute climatology: mean for each day-of-year across all years
    # Group by day-of-year and compute mean
    # CRITICAL FIX: Compute this immediately to break the dask graph dependency
    # Otherwise, accessing any slice of anomalies triggers a full re-read of all data
    print("   Computing climatology into memory (breaking dask graph)...")
    z500_clim = z500.groupby(dayofyear).mean(dim='time').compute()
    
    # Subtract the climatology from each day
    # xarray automatically aligns based on the dayofyear coordinate
    z500_anom = z500.groupby(dayofyear) - z500_clim
    
    # The result has an extra 'dayofyear' coordinate dimension, drop it
    z500_anom = z500_anom.drop_vars('dayofyear')
    
    # Add metadata
    z500_anom.name = 'z500_anomaly'
    z500_anom.attrs['description'] = 'Daily geopotential height anomalies (day-of-year climatology)'
    z500_anom.attrs['units'] = 'm'
    z500_anom.attrs['long_name'] = 'Z500 anomaly (deviation from day-of-year climatology)'
    z500_anom.attrs['method'] = 'ANO blocking detection (Woollings et al., 2018)'
    z500_anom.attrs['climatology'] = 'Day-of-year climatology (multi-year mean for each calendar day)'
    
    return z500_anom


def compute_threshold_percentile(z500_anom: xr.DataArray, 
                                  lat_min: float = 50.0, 
                                  lat_max: float = 80.0,
                                  percentile: float = 90.0) -> float:
    """
    Compute the Nth percentile threshold of anomalies over a latitude band.
    
    Uses AREA-WEIGHTED percentile to avoid over-representing polar grid cells
    on regular lat-lon grids (weights proportional to cos(latitude)).
    
    Parameters
    ----------
    z500_anom : xarray.DataArray
        Daily geopotential height anomalies with dimensions (time, lat, lon).
        Units: meters
    lat_min : float, optional
        Minimum latitude for threshold computation (default: 50.0°N)
    lat_max : float, optional
        Maximum latitude for threshold computation (default: 80.0°N)
    percentile : float, optional
        Percentile threshold to compute (default: 90.0)
        Must be between 0 and 100.
    
    Returns
    -------
    threshold : float
        The area-weighted Nth percentile value (scalar) in meters.
    
    Notes
    -----
    The function:
    1. Selects all grid points within [lat_min, lat_max]
    2. Computes area weights proportional to cos(latitude)
    3. Computes the area-weighted Nth percentile
    
    Area-weighting prevents polar grid cells from being over-represented
    in the percentile calculation, which would lead to thresholds that are
    too low and spurious blocking detections at high latitudes.
    
    Examples
    --------
    >>> threshold = compute_threshold_percentile(z500_anom, lat_min=50, lat_max=80, percentile=90)
    >>> print(f"90th percentile threshold: {threshold:.2f} m")
    """
    # Select the latitude band using boolean indexing (robust to lat ordering)
    lat_mask = (z500_anom.lat >= lat_min) & (z500_anom.lat <= lat_max)
    z500_region = z500_anom.sel(lat=lat_mask)
    
    # Compute area weights (proportional to cos(latitude))
    lats = z500_region.lat.values
    weights = np.cos(np.deg2rad(lats))  # Area weight for each latitude
    
    # Broadcast weights to (lat,) and let xarray handle the broadcasting
    # We want to weight by latitude only (same weight for all longitudes and times)
    # Flatten the data and apply weights
    data = z500_region.values  # (time, lat, lon)
    
    # Reshape to (time * lon, lat) for easier weighting
    nt, nlat, nlon = data.shape
    data_reshaped = data.transpose(1, 0, 2).reshape(nlat, -1)  # (lat, time*lon)
    
    # Flatten and create weighted sample
    data_flat = data_reshaped.ravel()  # All values
    weights_flat = np.repeat(weights, nt * nlon)  # Repeat weights for each time/lon
    
    # Remove NaN values
    valid = ~np.isnan(data_flat)
    data_flat = data_flat[valid]
    weights_flat = weights_flat[valid]
    
    # Compute weighted percentile using sorting and cumulative weights
    if len(data_flat) == 0:
        return np.nan
    
    # Sort data and weights together
    sorted_indices = np.argsort(data_flat)
    sorted_data = data_flat[sorted_indices]
    sorted_weights = weights_flat[sorted_indices]
    
    # Compute cumulative weights
    cumsum = np.cumsum(sorted_weights)
    cumsum = cumsum / cumsum[-1]  # Normalize to [0, 1]
    
    # Find the Nth percentile by interpolation
    threshold = float(np.interp(percentile / 100.0, cumsum, sorted_data))
    
    return threshold


# Backward compatibility alias
def compute_threshold_90(z500_anom: xr.DataArray, 
                         lat_min: float = 50.0, 
                         lat_max: float = 80.0) -> float:
    """Backward compatibility alias for compute_threshold_percentile with percentile=90."""
    return compute_threshold_percentile(z500_anom, lat_min, lat_max, percentile=90.0)


def _weighted_quantile(z_window: np.ndarray, weights_1d: np.ndarray, percentile: float = 90.0) -> float:
    """
    Compute area-weighted Nth percentile with guaranteed weight-data alignment.
    
    Transposes data to (lat, time, lon) layout, then broadcasts weights to match
    that exact layout before raveling. This guarantees one-to-one alignment
    regardless of memory layout (handles non-C-contiguous and dask-backed arrays).
    
    Parameters
    ----------
    z_window : np.ndarray
        Anomaly data for the 3-month window, shape (n_times, nlat, nlon)
    weights_1d : np.ndarray
        Area weights for each latitude, shape (nlat,)
        Typically cos(latitude) values
    percentile : float, optional
        Percentile threshold to compute (default: 90.0)
        Must be between 0 and 100.
    
    Returns
    -------
    threshold : float
        Area-weighted Nth percentile value
    """
    # Arrange as (lat, time, lon) to make weights broadcasting trivial
    # Transpose from (n_times, nlat, nlon) to (nlat, n_times, nlon)
    data_ltl = np.transpose(z_window, (1, 0, 2))
    
    # Build matching weights with identical layout, then flatten both
    # weights_1d[:, None, None] creates shape (nlat, 1, 1)
    # np.broadcast_to expands to (nlat, n_times, nlon) without copying memory
    w_ltl = np.broadcast_to(weights_1d[:, None, None], data_ltl.shape)
    
    # Flatten both in the same order - guarantees alignment
    data_flat = data_ltl.ravel()  # 1-D, same order as w_ltl.ravel()
    w_flat = w_ltl.ravel()
    
    # Drop NaNs synchronously
    valid = ~np.isnan(data_flat)
    data_flat = data_flat[valid]
    w_flat = w_flat[valid]
    
    # Guard: no data or all weights zero
    if data_flat.size == 0 or np.all(w_flat == 0.0):
        return np.nan
    
    # Sort data and reorder weights the same way
    idx = np.argsort(data_flat)
    sd = data_flat[idx]
    sw = w_flat[idx]
    
    # Compute cumulative weights and normalize to [0, 1]
    cw = np.cumsum(sw, dtype=np.float64)
    cw /= cw[-1]
    
    # Find Nth percentile using linear interpolation
    # This matches the original implementation and is more robust
    threshold = float(np.interp(percentile / 100.0, cw, sd))
    
    return threshold


# Backward compatibility alias
def _weighted_q90(z_window: np.ndarray, weights_1d: np.ndarray) -> float:
    """Backward compatibility alias for _weighted_quantile with percentile=90."""
    return _weighted_quantile(z_window, weights_1d, percentile=90.0)


def _compute_threshold_for_month(z500_region: xr.DataArray,
                                  months: np.ndarray,
                                  month: int,
                                  percentile: float = 90.0) -> Tuple[int, float]:
    """
    Helper function to compute AREA-WEIGHTED Nth percentile threshold for a single month.
    Used for parallel processing.
    
    Parameters
    ----------
    z500_region : xarray.DataArray
        Z500 anomalies already subset to latitude band (time, lat, lon)
    months : np.ndarray
        Array of month numbers for each time step
    month : int
        Month number (1-12) to compute threshold for
    percentile : float, optional
        Percentile threshold to compute (default: 90.0)
    
    Returns
    -------
    (month, threshold) : tuple
        Month number and computed area-weighted threshold value
    """
    # Define 3-month window centered on this month
    # month-1, month, month+1 (with wraparound for Jan and Dec)
    prev_month = month - 1 if month > 1 else 12
    next_month = month + 1 if month < 12 else 1
    window_months = [prev_month, month, next_month]
    
    # Select data from these three months
    month_mask = np.isin(months, window_months)
    z500_window = z500_region.isel(time=month_mask)
    
    # Compute AREA-WEIGHTED Nth percentile
    if len(z500_window.time) > 0:
        data = z500_window.values  # (time, lat, lon)
        
        # Compute area weights (proportional to cos(latitude))
        lats = z500_window.lat.values
        weights = np.cos(np.deg2rad(lats))
        
        # Use the same helper function that guarantees alignment
        threshold = _weighted_quantile(data, weights, percentile)
        
        # Sanity check: compare weighted vs unweighted quantile
        # They should have the same sign and similar magnitude
        q_unw = float(np.nanpercentile(data, percentile))
        if np.sign(threshold) != np.sign(q_unw):
            raise RuntimeError(
                f"Weighted Q{percentile:.0f} sign mismatch for month {month} "
                f"(weighted={threshold:.2f}, unweighted={q_unw:.2f}). "
                "Weights/data alignment likely broken."
            )
        
        if np.isnan(threshold):
            return (int(month), np.nan)
        else:
            return (int(month), threshold)
    else:
        return (int(month), np.nan)


def compute_monthly_thresholds_rolling(z500_anom: xr.DataArray,
                                       lat_min: float = 50.0,
                                       lat_max: float = 80.0,
                                       n_workers: int = 20,
                                       percentile: float = 90.0) -> Dict[int, float]:
    """
    Compute month-specific Nth percentile thresholds using 3-month rolling windows.
    PARALLELIZED VERSION: Processes each month in parallel using ThreadPoolExecutor.
    
    Following Woollings et al. (2018), each month uses a 3-month window centered
    on that month. For example:
    - December: uses Nov-Dec-Jan (NDJ)
    - January: uses Dec-Jan-Feb (DJF)
    - June: uses May-Jun-Jul (MJJ)
    
    This accounts for seasonal variations in blocking characteristics.
    
    Parameters
    ----------
    z500_anom : xarray.DataArray
        Daily geopotential height anomalies with dimensions (time, lat, lon).
        Must have a decodable time coordinate with month information.
        Units: meters
    lat_min : float, optional
        Minimum latitude for threshold computation (default: 50.0°N)
    lat_max : float, optional
        Maximum latitude for threshold computation (default: 80.0°N)
    n_workers : int, optional
        Number of parallel workers for processing months (default: 20)
    percentile : float, optional
        Percentile threshold to compute (default: 90.0)
        Must be between 0 and 100.
    
    Returns
    -------
    monthly_thresholds : dict
        Dictionary mapping month number (1-12) to Nth percentile threshold (m).
        Only months present in the input data will have entries.
    
    Notes
    -----
    - For each month, uses data from (month-1, month, month+1) with wraparound
    - Only computes thresholds for months that are present in the data
    - Handles year boundaries automatically (Dec→Jan, Aug→Sep, etc.)
    - Parallelized: processes all months concurrently using ThreadPoolExecutor
    
    Examples
    --------
    >>> # For DJF data (months 12, 1, 2)
    >>> thresholds = compute_monthly_thresholds_rolling(z500_anom)
    >>> print(thresholds)
    {12: 150.2, 1: 148.5, 2: 152.1}  # Example values in meters
    """
    # Extract month information from time coordinate
    try:
        months = z500_anom.time.dt.month.values
    except AttributeError:
        raise ValueError("Time coordinate must be datetime-like with .dt.month accessor")
    
    # Get unique months present in the data
    unique_months = np.unique(months)
    
    # Select latitude band (do this once before parallel processing)
    lat_mask = (z500_anom.lat >= lat_min) & (z500_anom.lat <= lat_max)
    z500_region = z500_anom.sel(lat=lat_mask)
    
    # Load data into memory if it's a dask array to avoid serialization issues
    # This is done once before parallel processing to avoid repeated computation
    if hasattr(z500_region.data, 'compute'):
        print(f"   Loading latitude subset into memory (dask array)...")
        load_start = time.time()
        z500_region = z500_region.compute()
        print(f"   Data loaded in {time.time() - load_start:.2f} seconds")
    
    # Compute threshold for each month in parallel
    monthly_thresholds = {}
    
    print(f"   Processing {len(unique_months)} months in parallel ({n_workers} workers)...")
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        # Submit all month computations
        future_to_month = {
            executor.submit(_compute_threshold_for_month, z500_region, months, month, percentile): month
            for month in unique_months
        }
        
        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_month):
            month, threshold = future.result()
            completed += 1
            if not np.isnan(threshold):
                monthly_thresholds[month] = threshold
            if completed % max(1, len(unique_months) // 4) == 0:
                print(f"      Progress: {completed}/{len(unique_months)} months completed...")
    
    return monthly_thresholds


def compute_monthly_thresholds_rolling_fast(z500_anom: xr.DataArray,
                                             lat_min: float = 50.0,
                                             lat_max: float = 80.0,
                                             percentile: float = 90.0,
                                             lon_min: Optional[float] = None,
                                             lon_max: Optional[float] = None) -> Dict[int, float]:
    """
    Fast vectorized version of compute_monthly_thresholds_rolling with AREA-WEIGHTING.
    
    Uses pure NumPy operations for maximum speed. No threading overhead.
    Optimized for large datasets on high-memory nodes.
    
    Parameters
    ----------
    z500_anom : xarray.DataArray
        Daily geopotential height anomalies with dimensions (time, lat, lon).
        Units: meters
    lat_min : float, optional
        Minimum latitude for threshold computation (default: 50.0°N)
    lat_max : float, optional
        Maximum latitude for threshold computation (default: 80.0°N)
    percentile : float, optional
        Percentile threshold to compute (default: 90.0)
        Must be between 0 and 100.
    lon_min : float, optional
        Minimum longitude for threshold computation (default: None = all longitudes).
        Uses degrees East (0-360). Handles wraparound if lon_min > lon_max.
    lon_max : float, optional
        Maximum longitude for threshold computation (default: None = all longitudes).
        Uses degrees East (0-360). Handles wraparound if lon_min > lon_max.
    
    Returns
    -------
    monthly_thresholds : dict
        Dictionary mapping month number (1-12) to area-weighted Nth percentile threshold (m).
        Only months present in the input data will have entries.
    
    Notes
    -----
    This is a vectorized, single-threaded version optimized for speed.
    Uses float32 for memory efficiency and area-weighted percentile to avoid
    over-representing polar grid cells.
    
    The weighted percentile computation uses 3D broadcasting to guarantee
    weight-data alignment, which is critical for non-C-contiguous or
    dask-backed arrays.
    """
    months = z500_anom.time.dt.month.values
    lat_mask = (z500_anom.lat >= lat_min) & (z500_anom.lat <= lat_max)
    
    # Get subset: latitude first
    z500_subset = z500_anom.sel(lat=lat_mask)
    
    # Optionally subset by longitude
    if lon_min is not None and lon_max is not None:
        if lon_min <= lon_max:
            lon_mask = (z500_subset.lon >= lon_min) & (z500_subset.lon <= lon_max)
        else:
            # Wraparound case (e.g., 350° to 10°)
            lon_mask = (z500_subset.lon >= lon_min) | (z500_subset.lon <= lon_max)
        z500_subset = z500_subset.sel(lon=lon_mask)
    z = z500_subset.values.astype("float32", copy=False)  # (time, lat, lon)
    nt, nlat, nlon = z.shape
    
    # Compute area weights (proportional to cos(latitude))
    lats = z500_subset.lat.values
    weights = np.cos(np.deg2rad(lats)).astype("float32")
    
    # Only compute thresholds for months that are actually present in the data
    unique_months = np.unique(months)
    
    thresholds = {}
    for m in unique_months:  # Only iterate over months present in data
        pm = 12 if m == 1 else m - 1
        nm = 1 if m == 12 else m + 1
        # Only include months that are in the data
        mask = np.zeros(nt, dtype=bool)
        if pm in unique_months:
            mask |= (months == pm)
        if m in unique_months:
            mask |= (months == m)
        if nm in unique_months:
            mask |= (months == nm)
        
        if not mask.any():
            continue
        
        # Get data for this 3-month window
        z_window = z[mask]  # (n_times, nlat, nlon)
        
        # Arrange as (lat, time, lon) to make weights broadcasting trivial
        data_ltl = np.transpose(z_window, (1, 0, 2))  # (nlat, n_times, nlon)
        
        # Build matching weights with identical layout, then flatten both
        w_ltl = np.broadcast_to(weights[:, None, None], data_ltl.shape)
        data_flat = data_ltl.ravel()  # 1-D, same order as w_ltl.ravel()
        w_flat = w_ltl.ravel()
        
        # Drop NaNs synchronously
        valid = ~np.isnan(data_flat)
        data_flat = data_flat[valid]
        w_flat = w_flat[valid]
        
        # Guard: no weights or all zero
        if data_flat.size == 0 or np.all(w_flat == 0.0):
            continue
        
        # Sort data and reorder weights the same way
        idx = np.argsort(data_flat)
        sd = data_flat[idx]
        sw = w_flat[idx]
        
        # Compute cumulative weights and normalize to [0, 1]
        cw = np.cumsum(sw, dtype=np.float64)
        cw /= cw[-1]
        
        # Find Nth percentile using linear interpolation
        thr = float(np.interp(percentile / 100.0, cw, sd))
        thresholds[m] = thr
        
        # Quick consistency check: compare weighted vs unweighted quantile
        q_unw = float(np.percentile(data_flat, percentile))
        print(f"Month {m:02d}: weighted={thr:.2f}, unweighted={q_unw:.2f}")
        
        if np.sign(thr) != np.sign(q_unw):
            raise RuntimeError(
                f"Q{percentile:.0f} sign mismatch for month {m}: weighted={thr:.2f}, unweighted={q_unw:.2f}"
            )
    
    return thresholds


def create_blocking_mask(z500_anom: xr.DataArray,
                         threshold_90: Union[float, Dict[int, float]],
                         n_workers: int = 20) -> xr.DataArray:
    """
    Create a daily blocking mask based on anomaly exceedance.
    MEMORY-SAFE VERSION: Uses uint8 and lazy xarray operations to avoid kernel crashes.
    
    Grid points are marked as "blocked" (1) if the anomaly exceeds the 
    threshold. All other points are marked as "not blocked" (0).
    
    IMPORTANT: The threshold should be computed from a specific latitude band
    (e.g., 50-80°N using compute_threshold_90), but this function applies 
    that threshold to ALL grid points in the input data (typically the entire
    Northern Hemisphere). This is the correct ANO method approach.
    
    Parameters
    ----------
    z500_anom : xarray.DataArray
        Daily geopotential height anomalies with dimensions (time, lat, lon).
        Units: meters
    threshold_90 : float or dict
        Either a single 90th percentile threshold value in meters, or
        a dictionary mapping month numbers (1-12) to month-specific thresholds.
        When dict is provided, each day uses the threshold for its month.
    n_workers : int, optional
        Number of parallel workers (kept for API compatibility, not used in memory-safe version).
        Default: 20
    
    Returns
    -------
    blocked_mask : xarray.DataArray
        Binary mask with same shape as z500_anom.
        Values: 1 = blocked, 0 = not blocked
        Blocking is detected at all grid points where anomaly exceeds threshold.
        Uses uint8 dtype for memory efficiency.
    
    Notes
    -----
    The mask is created by comparing z500_anom to threshold_90 at each grid point.
    The threshold is typically computed over 50-80°N but applied everywhere in NH.
    This follows the ANO method specification (Woollings et al., 2018).
    
    When using monthly thresholds, each day is compared against the threshold
    computed for its month using a 3-month rolling window (Woollings et al., 2018).
    
    This memory-safe version:
    - Uses uint8 dtype instead of float64 (8x memory reduction)
    - Uses lazy xarray operations (works with dask arrays)
    - Avoids materializing large numpy arrays in memory
    - Processes months sequentially (avoids memory pressure from parallelization)
    
    Examples
    --------
    >>> # Single threshold for all data
    >>> threshold_90 = compute_threshold_90(z500_anom, lat_min=50, lat_max=80)
    >>> blocked_mask = create_blocking_mask(z500_anom, threshold_90)
    
    >>> # Month-specific thresholds
    >>> monthly_thresholds = compute_monthly_thresholds_rolling(z500_anom)
    >>> blocked_mask = create_blocking_mask(z500_anom, monthly_thresholds)
    >>> print(f"Blocked grid points: {blocked_mask.sum().values}")
    """
    # Single global threshold
    if not isinstance(threshold_90, dict):
        blocked_mask = (z500_anom > threshold_90).astype("uint8")
        blocked_mask.name = "blocking_mask"
        blocked_mask.attrs.update({
            'description': 'Daily blocking mask (ANO method)',
            'method': 'ANO blocking detection (Woollings et al., 2018)',
            'threshold_info': f'Single threshold: {float(threshold_90):.2f} m',
            'note': 'Threshold computed from 50-80°N but applied to all NH grid points',
            'values': '1=blocked, 0=not blocked',
            'long_name': 'Atmospheric blocking occurrence',
        })
        return blocked_mask
    
    # Month-specific thresholds
    # Use lazy xarray operations - works with dask arrays
    try:
        months = z500_anom.time.dt.month
    except AttributeError:
        raise ValueError("Time coordinate must be datetime-like for monthly thresholds")
    
    # Start with all zeros (lazy if z500_anom is dask-backed)
    blocked_mask = xr.zeros_like(z500_anom, dtype="uint8")
    
    # Process each month sequentially (memory-safe, avoids parallelization overhead)
    for m, thr in threshold_90.items():
        # Boolean mask over time dimension, auto-broadcast over lat/lon
        cond = (months == m)
        if cond.any():
            # Use xr.where with lazy evaluation
            blocked_mask = xr.where(cond & (z500_anom > thr), 1, blocked_mask)
    
    blocked_mask.name = "blocking_mask"
    blocked_mask.attrs.update({
        'description': 'Daily blocking mask (ANO method)',
        'method': 'ANO blocking detection (Woollings et al., 2018)',
        'threshold_info': f'Month-specific scalar thresholds (3-month rolling windows)',
        'note': 'Thresholds computed from 50-80°N but applied to all NH grid points',
        'values': '1=blocked, 0=not blocked',
        'long_name': 'Atmospheric blocking occurrence',
    })
    
    return blocked_mask


def create_blocking_mask_fast(z500_anom: xr.DataArray,
                              threshold_90: Union[float, Dict[int, float]]) -> xr.DataArray:
    """
    Fast vectorized version of create_blocking_mask.
    
    Uses one big broadcasted comparison instead of per-month slicing.
    Optimized for large datasets on high-memory nodes.
    
    Parameters
    ----------
    z500_anom : xarray.DataArray
        Daily geopotential height anomalies with dimensions (time, lat, lon).
        Units: meters (should be float32 for best performance)
    threshold_90 : float or dict
        Either a single 90th percentile threshold value in meters, or
        a dictionary mapping month numbers (1-12) to month-specific thresholds.
    
    Returns
    -------
    blocked_mask : xarray.DataArray
        Binary mask with same shape as z500_anom.
        Values: 1 = blocked, 0 = not blocked
        Uses uint8 dtype for memory efficiency.
    
    Notes
    -----
    This version materializes the data array once and does a single
    broadcasted comparison, which is much faster than per-month operations.
    """
    data = z500_anom.values  # (T, Y, X), should be float32
    T = data.shape[0]
    
    if isinstance(threshold_90, dict):
        months = z500_anom.time.dt.month.values
        thr_t = np.empty(T, dtype=data.dtype)
        for m in range(1, 13):
            thr = threshold_90.get(m, np.nan)
            thr_t[months == m] = thr
        
        # Broadcast: (T,1,1) vs (T,Y,X)
        blocked = (data > thr_t[:, None, None]).astype("uint8")
        thr_info = "Month-specific thresholds (3-month rolling windows)"
    else:
        blocked = (data > float(threshold_90)).astype("uint8")
        thr_info = f"Single threshold: {float(threshold_90):.2f} m"
    
    blocked_mask = xr.DataArray(
        blocked,
        coords=z500_anom.coords,
        dims=z500_anom.dims,
        name="blocking_mask",
    )
    blocked_mask.attrs.update({
        'description': 'Daily blocking mask (ANO method)',
        'method': 'ANO blocking detection (Woollings et al., 2018)',
        'threshold_info': thr_info,
        'note': 'Threshold computed from 50-80°N but applied to all NH grid points',
        'values': '1=blocked, 0=not blocked',
        'long_name': 'Atmospheric blocking occurrence',
    })
    return blocked_mask


def detect_ano_blocking(z500: xr.DataArray,
                        lat_min: float = 50.0,
                        lat_max: float = 80.0,
                        use_monthly_thresholds: bool = True,
                        return_intermediates: bool = False,
                        percentile: float = 90.0) -> Tuple[xr.DataArray, ...]:
    """
    Complete ANO blocking detection pipeline.
    
    This is a convenience function that combines all steps of the ANO
    blocking detection method:
    1. Compute anomalies
    2. Compute Nth percentile threshold(s) (from lat_min to lat_max)
    3. Create blocking mask (applies threshold to all NH grid points)
    
    Parameters
    ----------
    z500 : xarray.DataArray
        Daily geopotential height field with dimensions (time, lat, lon).
        Units: meters
    lat_min : float, optional
        Minimum latitude for THRESHOLD computation (default: 50.0°N)
    lat_max : float, optional
        Maximum latitude for THRESHOLD computation (default: 80.0°N)
        NOTE: The threshold is computed from this band but applied to all NH
    use_monthly_thresholds : bool, optional
        If True, use month-specific thresholds with 3-month rolling windows
        (Woollings et al., 2018). If False, use single threshold for all data.
        Default: True
    return_intermediates : bool, optional
        If True, return (blocked_mask, z500_anom, threshold).
        If False, return only blocked_mask (default: False)
    percentile : float, optional
        Percentile threshold to compute (default: 90.0)
        Must be between 0 and 100.
    
    Returns
    -------
    blocked_mask : xarray.DataArray
        Binary blocking mask (1=blocked, 0=not blocked)
        Blocking is detected at all grid points in the input data
    z500_anom : xarray.DataArray (optional)
        Anomalies, returned only if return_intermediates=True
    threshold : float or dict (optional)
        Threshold value(s), returned only if return_intermediates=True
    
    Examples
    --------
    >>> # Get only the blocking mask (monthly thresholds)
    >>> blocked_mask = detect_ano_blocking(z500_djf_1years)
    
    >>> # Use single threshold (old method)
    >>> blocked_mask = detect_ano_blocking(z500_djf_1years, use_monthly_thresholds=False)
    
    >>> # Get all intermediate results with custom percentile
    >>> blocked_mask, z500_anom, threshold = detect_ano_blocking(
    ...     z500_djf_1years, return_intermediates=True, percentile=85
    ... )
    >>> print(f"Thresholds: {threshold}")
    >>> print(f"Total blocked occurrences: {blocked_mask.sum().values}")
    """
    # Step 1: Compute anomalies
    z500_anom = compute_anomalies(z500)
    
    # Step 2: Compute Nth percentile threshold(s)
    if use_monthly_thresholds:
        threshold = compute_monthly_thresholds_rolling(z500_anom, lat_min, lat_max, percentile=percentile)
    else:
        threshold = compute_threshold_percentile(z500_anom, lat_min, lat_max, percentile=percentile)
    
    # Step 3: Create blocking mask (applies threshold to ALL grid points)
    blocked_mask = create_blocking_mask(z500_anom, threshold)
    
    if return_intermediates:
        return blocked_mask, z500_anom, threshold
    else:
        return blocked_mask


# Convenience function for backward compatibility
def ano_blocking_detection(z500_djf_1years: xr.DataArray,
                          use_monthly_thresholds: bool = True,
                          percentile: float = 90.0) -> Tuple[xr.DataArray, xr.DataArray, Union[float, Dict[int, float]]]:
    """
    Convenience wrapper matching the exact specification in the request.
    
    Uses correct ANO method: computes threshold from 50-80°N but applies it
    to all Northern Hemisphere grid points. By default uses month-specific
    thresholds with 3-month rolling windows (Woollings et al., 2018).
    
    Parameters
    ----------
    z500_djf_1years : xarray.DataArray
        Daily geopotential height field (time=90, lat=32, lon=128)
        Units: meters, Northern Hemisphere only
    use_monthly_thresholds : bool, optional
        If True, use month-specific thresholds (default).
        If False, use single threshold for all data.
    percentile : float, optional
        Percentile threshold to compute (default: 90.0)
        Must be between 0 and 100.
    
    Returns
    -------
    z500_anom : xarray.DataArray
        Daily anomalies
    blocked_mask : xarray.DataArray
        Binary blocking mask (applied to all NH grid points)
    threshold : float or dict
        Nth percentile threshold value(s) (computed from 50-80°N)
        Dict if use_monthly_thresholds=True, float otherwise
    """
    z500_anom = compute_anomalies(z500_djf_1years)
    
    if use_monthly_thresholds:
        threshold = compute_monthly_thresholds_rolling(z500_anom, lat_min=50.0, lat_max=80.0, percentile=percentile)
    else:
        threshold = compute_threshold_percentile(z500_anom, lat_min=50.0, lat_max=80.0, percentile=percentile)
    
    blocked_mask = create_blocking_mask(z500_anom, threshold)
    
    return z500_anom, blocked_mask, threshold


def calculate_grid_cell_areas(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """
    Calculate the area of each grid cell in km².
    
    Uses spherical Earth approximation with area formula:
    Area = R² * |Δλ| * |sin(φ₁) - sin(φ₂)|
    where R is Earth radius, Δλ is longitude spacing, φ is latitude.
    
    Parameters
    ----------
    lat : np.ndarray
        1D array of latitude values in degrees
    lon : np.ndarray
        1D array of longitude values in degrees
    
    Returns
    -------
    areas : np.ndarray
        2D array of grid cell areas in km², shape (len(lat), len(lon))
    
    Notes
    -----
    - Assumes uniform spacing in latitude and longitude
    - Uses midpoint approximation for cell boundaries
    - Uses Earth radius of 6371 km
    
    Examples
    --------
    >>> areas = calculate_grid_cell_areas(lat, lon)
    >>> print(f"Total area: {areas.sum():.2e} km²")
    """
    # Calculate latitude spacing (assumes uniform)
    if len(lat) > 1:
        dlat = np.abs(np.diff(lat).mean())
    else:
        dlat = 1.0  # fallback
    
    # Calculate longitude spacing (assumes uniform)
    if len(lon) > 1:
        dlon = np.abs(np.diff(lon).mean())
    else:
        dlon = 1.0  # fallback
    
    # Convert to radians
    lat_rad = np.deg2rad(lat)
    dlon_rad = np.deg2rad(dlon)
    
    # Calculate latitude boundaries (edges of cells)
    lat_edges = np.zeros(len(lat) + 1)
    lat_edges[1:-1] = (lat_rad[:-1] + lat_rad[1:]) / 2
    lat_edges[0] = lat_rad[0] - np.deg2rad(dlat) / 2
    lat_edges[-1] = lat_rad[-1] + np.deg2rad(dlat) / 2
    
    # Calculate area for each latitude band
    # Area = R² * |Δλ| * |sin(φ₁) - sin(φ₂)|
    lat_band_areas = EARTH_RADIUS_KM**2 * dlon_rad * np.abs(
        np.sin(lat_edges[1:]) - np.sin(lat_edges[:-1])
    )
    
    # Broadcast to (lat, lon) grid
    # Each cell in a latitude band has the same area
    areas = np.tile(lat_band_areas[:, np.newaxis], (1, len(lon)))
    
    return areas


def identify_spatial_regions(blocked_mask_2d: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Identify connected regions in a 2D binary mask with CORRECT PERIODIC BOUNDARY HANDLING.
    
    Uses 4-connectivity (N-S-E-W neighbors) with periodic boundary in longitude.
    Handles the case where input might already have a cyclic point (last column = first column).
    
    Parameters
    ----------
    blocked_mask_2d : np.ndarray
        2D binary array (lat, lon) with 1=blocked, 0=not blocked
    
    Returns
    -------
    labeled : np.ndarray
        2D array with unique integer labels for each connected region
    num_features : int
        Number of distinct regions found
    
    Notes
    -----
    Longitude dimension is treated as periodic (wraps around).
    If the input array has a duplicated column (cyclic point), it treats the
    first and last columns as the same location and merges labels accordingly.
    """
    rows, cols = blocked_mask_2d.shape
    
    # Check if input already has a cyclic point (duplicate column)
    has_cyclic_point = False
    if cols > 1:
        # Check if first and last columns are identical
        if np.array_equal(blocked_mask_2d[:, 0], blocked_mask_2d[:, -1]):
            has_cyclic_point = True
    
    # Prepare data for labeling
    # We want to work with the unique columns only for the main labeling
    if has_cyclic_point:
        mask_unique = blocked_mask_2d[:, :-1]
    else:
        mask_unique = blocked_mask_2d
    
    # To handle periodicity, we append the first column to the end
    # (This effectively recreates the cyclic point structure if it wasn't there,
    # or restores it if we removed it. We do this to let ndimage connect across the seam.)
    mask_periodic = np.concatenate([mask_unique, mask_unique[:, :1]], axis=1)
    
    # Define structure for 4-connectivity
    structure = np.array([[0, 1, 0],
                          [1, 1, 1],
                          [0, 1, 0]], dtype=bool)
    
    # Label connected components on the periodic array
    # ndimage will connect the last column (copy of first) to its neighbors
    labeled_periodic, num_features = ndimage.label(mask_periodic, structure=structure)
    
    # Now we need to merge labels that wrap around
    # ndimage sees column 0 and column -1 as spatially far apart
    # But column -1 is a copy of column 0.
    # If a component spans the boundary, it will have one label at [r, 0]
    # and possibly the SAME label at [r, -1] if it was connected via the interior.
    # BUT if the component is ONLY connected via the boundary (e.g. a block sitting on the seam),
    # ndimage might assign Label A to the left part and Label B to the right part
    # (Wait, no, if we append the column, ndimage connects the rightmost part to the appended column.
    #  The appended column IS the start. So the right part is connected to the start copy.
    #  BUT the start copy is not connected to the start original by ndimage.
    #  So we have Label A at [r,0] and Label B at [r,-1]. We need to merge them.)
    
    # Find equivalences: labels that appear at the same row in col 0 and col -1
    # These represent the same physical location, so their labels must be merged
    label_map = {}  # old_label -> new_label
    
    # Get labels at the seam
    labels_start = labeled_periodic[:, 0]
    labels_end = labeled_periodic[:, -1]
    
    # Find rows where both are non-zero (part of a block)
    connected_rows = (labels_start > 0) & (labels_end > 0)
    
    if np.any(connected_rows):
        # Get the pairs of labels that need merging
        pairs = np.column_stack((labels_start[connected_rows], labels_end[connected_rows]))
        
        # Use a simple Union-Find approach or just graph connected components of labels
        # Build an adjacency graph of labels
        adj = {}
        for l1, l2 in pairs:
            if l1 == l2: continue
            if l1 not in adj: adj[l1] = set()
            if l2 not in adj: adj[l2] = set()
            adj[l1].add(l2)
            adj[l2].add(l1)
        
        # Find connected components of labels
        visited = set()
        for label in adj:
            if label in visited: continue
            
            # BFS/DFS to find all connected labels
            component = set()
            stack = [label]
            while stack:
                curr = stack.pop()
                if curr in visited: continue
                visited.add(curr)
                component.add(curr)
                if curr in adj:
                    stack.extend(adj[curr])
            
            # Map all labels in this component to the smallest label
            target_label = min(component)
            for l in component:
                if l != target_label:
                    label_map[l] = target_label
    
    # Apply the label merging
    if label_map:
        # Vectorized mapping is faster for large arrays
        # Create a mapping array
        max_label = num_features
        map_array = np.arange(max_label + 1)
        for old, new in label_map.items():
            map_array[old] = new
        
        # Apply mapping
        labeled_periodic = map_array[labeled_periodic]
        
        # Re-number labels to be consecutive (optional but good for cleanliness)
        # unique_labels = np.unique(labeled_periodic)
        # if len(unique_labels) < max_label + 1:
        #    ... (skip for performance unless strictly needed)
    
    # Extract the result
    # If input had cyclic point, we return 0..-1 (full width including cyclic)
    # If input was unique, we return 0..-1 (excluding the appended column)
    
    if has_cyclic_point:
        # Input had cyclic point (N columns). We worked on N-1 + 1 = N columns.
        # labeled_periodic has N columns.
        # But we need to make sure the last column matches the first column (after merging)
        # The merging step ensured consistency between 0 and -1.
        labeled = labeled_periodic
    else:
        # Input was unique (N columns). We worked on N+1 columns.
        # Return the first N columns.
        labeled = labeled_periodic[:, :-1]
        
    # Recalculate number of features (merging might have reduced count)
    num_features = len(np.unique(labeled)) - (1 if 0 in labeled else 0)
    
    return labeled, num_features


def calculate_overlap(region1: np.ndarray, region2: np.ndarray, grid_areas: np.ndarray) -> float:
    """
    Calculate the area-weighted spatial overlap between two binary regions.
    
    Overlap is defined as the intersection area divided by the smaller region area:
    overlap = Area(A ∩ B) / min(Area(A), Area(B))
    
    This uses proper area weighting to account for varying grid cell sizes at
    different latitudes, which is more accurate than simple grid cell counting.
    
    Parameters
    ----------
    region1 : np.ndarray
        Binary mask of first region (2D boolean array)
    region2 : np.ndarray
        Binary mask of second region (2D boolean array)
    grid_areas : np.ndarray
        Array of grid cell areas in km² with shape (lat, lon)
    
    Returns
    -------
    overlap : float
        Area-weighted overlap fraction in range [0, 1]
    
    Notes
    -----
    Using area weighting is important because Earth's grid cells shrink toward
    the poles. A grid cell at 70°N has much smaller area than one at 30°N.
    
    Examples
    --------
    >>> overlap = calculate_overlap(region1_mask, region2_mask, grid_areas)
    """
    # Calculate area-weighted intersection and sizes
    area_intersection = np.sum(grid_areas[region1 & region2])
    area1 = np.sum(grid_areas[region1])
    area2 = np.sum(grid_areas[region2])
    
    if area1 == 0.0 or area2 == 0.0:
        return 0.0
    
    min_area = min(area1, area2)
    overlap = area_intersection / min_area
    
    return overlap



def identify_blocking_events(blocked_mask: xr.DataArray,
                             min_area: float = 2e6,  # km²
                             min_duration: int = 5,  # days
                             min_overlap: float = 0.5) -> Tuple[xr.DataArray, Dict]:
    """
    Identify individual blocking events with area, duration, and overlap criteria.
    
    Parameters
    ----------
    blocked_mask : xarray.DataArray
        Binary blocking mask with dimensions (time, lat, lon)
        Values: 1=blocked, 0=not blocked
    min_area : float, optional
        Minimum area threshold in km² (default: 2×10⁶ km²)
    min_duration : int, optional
        Minimum duration in consecutive days (default: 5 days)
    min_overlap : float, optional
        Minimum overlap fraction between consecutive days (default: 0.5 = 50%)
    
    Returns
    -------
    blocking_frequency : xarray.DataArray
        Frequency of blocking at each location (fraction of time blocked)
        Shape: (lat, lon)
    event_info : dict
        Dictionary containing:
        - 'num_events': Total number of identified events
        - 'event_mask': 3D array (time, lat, lon) with unique event IDs
        - 'event_durations': List of durations for each event
        - 'event_areas': List of mean areas for each event
    
    Notes
    -----
    The algorithm:
    1. Calculates grid cell areas based on latitude (using Earth radius = 6371 km)
    2. For each day, identifies connected spatial regions
    3. Filters regions by minimum area
    4. Tracks regions through time using overlap criterion
    5. Identifies events lasting ≥ min_duration days
    6. Calculates blocking frequency from valid events
    
    Examples
    --------
    >>> freq, info = identify_blocking_events(blocked_mask)
    >>> print(f"Found {info['num_events']} blocking events")
    >>> print(f"Mean blocking frequency: {float(freq.mean())*100:.2f}%")
    """
    # Extract coordinates
    lat = blocked_mask.lat.values
    lon = blocked_mask.lon.values
    n_times = len(blocked_mask.time)
    n_lats = len(lat)
    n_lons = len(lon)
    
    # Calculate grid cell areas
    grid_areas = calculate_grid_cell_areas(lat, lon)
    
    # Initialize tracking structures
    event_mask = np.zeros((n_times, n_lats, n_lons), dtype=int)
    current_event_id = 1
    
    # Track regions from previous day
    previous_regions = {}  # region_id -> {'mask', 'area', 'event_id', 'duration'}
    
    # Event information
    event_durations = {}  # event_id -> duration
    event_areas = {}  # event_id -> list of daily areas
    
    # Process each time step
    for t in range(n_times):
        # Get blocking mask for this day
        mask_2d = blocked_mask.isel(time=t).values.astype(bool)
        
        # Identify spatial regions
        labeled, num_features = identify_spatial_regions(mask_2d)
        
        # Process each region found today
        current_regions = {}
        
        for region_id in range(1, num_features + 1):
            # Extract this region's mask
            # Check if the region ID actually exists in the labeled array
            # (It might have been merged away if we renumbered, but num_features implies 1..N)
            # If we didn't renumber, we should iterate over unique labels
            pass # Logic is handled below by iterating labels present? No, usually range(1, num_features+1) is unsafe if sparse.
                 # Better to iterate unique values.
        
        unique_labels = np.unique(labeled)
        unique_labels = unique_labels[unique_labels > 0] # Exclude background
        
        for region_id in unique_labels:
            # Extract this region's mask
            region_mask = (labeled == region_id)
            
            # Calculate area
            region_area = np.sum(grid_areas[region_mask])
            
            # Skip if below minimum area
            if region_area < min_area:
                continue
            
            # Check for overlap with previous day's regions
            matched_event_id = None
            best_overlap = 0.0
            
            if t > 0 and previous_regions:
                for prev_id, prev_info in previous_regions.items():
                    overlap = calculate_overlap(region_mask, prev_info['mask'], grid_areas)
                    
                    if overlap >= min_overlap and overlap > best_overlap:
                        best_overlap = overlap
                        matched_event_id = prev_info['event_id']
            
            # Assign event ID
            if matched_event_id is not None:
                # Continue existing event
                event_id = matched_event_id
                event_durations[event_id] = event_durations.get(event_id, 0) + 1
            else:
                # Start new event
                event_id = current_event_id
                event_durations[event_id] = 1
                event_areas[event_id] = []
                current_event_id += 1
            
            # Record area
            event_areas[event_id].append(region_area)
            
            # Store in event mask
            event_mask[t, region_mask] = event_id
            
            # Save for next iteration
            current_regions[region_id] = {
                'mask': region_mask,
                'area': region_area,
                'event_id': event_id,
                'duration': event_durations[event_id]
            }
        
        # Update previous regions for next iteration
        previous_regions = current_regions
    
    # Filter events by minimum duration
    valid_event_ids = {eid for eid, dur in event_durations.items() if dur >= min_duration}
    
    # Create filtered event mask
    filtered_event_mask = np.where(
        np.isin(event_mask, list(valid_event_ids)),
        event_mask,
        0
    )
    
    # Calculate blocking frequency (fraction of time each location was blocked)
    blocking_frequency_data = (filtered_event_mask > 0).sum(axis=0) / n_times
    
    blocking_frequency = xr.DataArray(
        blocking_frequency_data,
        coords={'lat': lat, 'lon': lon},
        dims=['lat', 'lon'],
        name='blocking_frequency',
        attrs={
            'description': 'Blocking frequency (fraction of time blocked)',
            'method': 'ANO blocking with event tracking',
            'min_area_km2': min_area,
            'min_duration_days': min_duration,
            'min_overlap': min_overlap,
            'units': 'fraction',
            'long_name': 'Blocking frequency'
        }
    )
    
    # Prepare output info
    event_info = {
        'num_events': len(valid_event_ids),
        'event_mask': filtered_event_mask,
        'event_durations': {eid: event_durations[eid] for eid in valid_event_ids},
        'event_areas': {eid: event_areas[eid] for eid in valid_event_ids},
        'all_event_ids': list(valid_event_ids)
    }
    
    return blocking_frequency, event_info


def ano_blocking_complete(z500: xr.DataArray,
                          lat_min: float = 50.0,
                          lat_max: float = 80.0,
                          min_area: float = 2e6,
                          min_duration: int = 5,
                          min_overlap: float = 0.5,
                          use_monthly_thresholds: bool = True,
                          n_workers: int = 20,
                          anomaly_scale: float = 1.0,
                          restrict_to_core_season: bool = True,
                          percentile: float = 90.0) -> Tuple[xr.DataArray, Dict]:
    """
    Complete ANO blocking detection from raw z500 data to blocking frequency.
    
    This is a single wrapper function that performs all steps:
    1. Compute anomalies
    2. Calculate 90th percentile threshold(s) (from lat_min to lat_max)
    3. Create blocking mask (applies threshold to all NH grid points)
    4. Identify individual events
    5. Calculate blocking frequency
    
    Parameters
    ----------
    z500 : xarray.DataArray
        Daily geopotential height field with dimensions (time, lat, lon).
        Units: meters, Northern Hemisphere recommended
    lat_min : float, optional
        Minimum latitude for THRESHOLD computation (default: 50.0°N)
    lat_max : float, optional
        Maximum latitude for THRESHOLD computation (default: 80.0°N)
        NOTE: Threshold is computed from this band but applied to all NH
    min_area : float, optional
        Minimum area threshold in km² (default: 2×10⁶ km²)
    min_duration : int, optional
        Minimum duration in consecutive days (default: 5 days)
    min_overlap : float, optional
        Minimum overlap fraction between consecutive days (default: 0.5 = 50%)
    use_monthly_thresholds : bool, optional
        If True, use month-specific thresholds with 3-month rolling windows
        (Woollings et al., 2018). If False, use single threshold. Default: True
    n_workers : int, optional
        Number of parallel workers for computing monthly thresholds (default: 20).
        Only used when use_monthly_thresholds=True.
    anomaly_scale : float, optional
        Multiplicative scaling factor applied to the Z500 anomaly field after
        computing anomalies but before threshold computation and blocking mask
        construction (default: 1.0). A value of 1.0 reproduces the original
        unscaled behavior. This parameter is useful for variance calibration
        of emulator forecasts, where the emulator may have systematically
        under- or over-estimated variance. For example, if an emulator's Z500
        variance is 80% of reanalysis, setting anomaly_scale=1.25 (≈1/0.8)
        would rescale the anomalies to match the reanalysis variance before
        blocking detection.
    restrict_to_core_season : bool, optional
        If True (default), restrict blocking event detection to core season months
        only, even when data includes extended shoulder months for threshold
        computation. This follows the Woollings et al. (2018) methodology where
        thresholds use 3-month rolling windows (requiring shoulder months) but
        blocking events are only reported for the core season:
        - NDJFM data → events detected only in DJF (Dec, Jan, Feb)
        - MJJAS data → events detected only in JJA (Jun, Jul, Aug)
        If False, events are detected in all months present in the data.
    percentile : float, optional
        Percentile threshold to compute (default: 90.0).
        Must be between 0 and 100. Higher values are more restrictive
        (fewer blocking events detected).
    
    Returns
    -------
    blocking_frequency : xarray.DataArray
        Frequency of blocking at each location (fraction of time blocked)
        Shape: (lat, lon)
    event_info : dict
        Dictionary containing:
        - 'num_events': Total number of identified events
        - 'event_mask': 3D array (time, lat, lon) with unique event IDs
        - 'event_durations': Dict of durations for each event
        - 'event_areas': Dict of mean areas for each event
        - 'all_event_ids': List of valid event IDs
        - 'z500_anom': Anomaly field (scaled, for reference)
        - 'blocked_mask': Binary blocking mask (for reference)
        - 'threshold_90': Threshold value(s) (for reference)
    
    Examples
    --------
    >>> # Simple usage with defaults (monthly thresholds, core season only)
    >>> blocking_frequency, event_info = ano_blocking_complete(z500_djf_1years)
    >>> print(f"Found {event_info['num_events']} blocking events")
    >>> print(f"Mean frequency: {float(blocking_frequency.mean())*100:.2f}%")
    
    >>> # Use single threshold (old method)
    >>> freq, info = ano_blocking_complete(z500_djf_1years, use_monthly_thresholds=False)
    
    >>> # Custom parameters
    >>> freq, info = ano_blocking_complete(
    ...     z500_djf_1years,
    ...     lat_min=55.0,
    ...     lat_max=75.0,
    ...     min_area=1.5e6,
    ...     min_duration=3,
    ...     min_overlap=0.4
    ... )
    
    >>> # Variance calibration for emulator with 80% of reanalysis variance
    >>> freq, info = ano_blocking_complete(z500_emulator, anomaly_scale=1.25)
    
    >>> # Detect events in all months (including shoulder months Nov, Mar)
    >>> freq, info = ano_blocking_complete(z500_ndjfm, restrict_to_core_season=False)
    
    Notes
    -----
    This function combines all ANO blocking detection steps into a single call.
    By default, uses month-specific thresholds with 3-month rolling windows
    (e.g., December uses NDJ, January uses DJF, February uses JFM). This
    follows Woollings et al. (2018) and accounts for seasonal variations.
    The threshold is computed from lat_min to lat_max (default 50-80°N) but 
    is applied to all grid points in the Northern Hemisphere.
    The intermediate results (anomalies, mask, threshold) are included in 
    event_info for reference and validation.
    Uses Earth radius of 6371 km for area calculations.
    
    When anomaly_scale != 1.0, the scaling is applied consistently to the
    anomaly field before both threshold computation and blocking mask
    construction, ensuring internal consistency of the detection algorithm.
    
    When restrict_to_core_season=True (default), the shoulder months are used
    only for threshold computation but blocking events are detected only in
    the core season (DJF for winter, JJA for summer). This prevents events
    that start in November or March from being counted as DJF events.
    """
    print("\n" + "="*70)
    print("ANO BLOCKING DETECTION - STEP-BY-STEP PROGRESS")
    print("="*70)
    
    # Print input data info
    print(f"\n📊 Input Data:")
    print(f"   Shape: {z500.shape}")
    print(f"   Dimensions: {dict(z500.sizes)}")
    print(f"   Time range: {z500.time.values[0]} to {z500.time.values[-1]}")
    print(f"   Latitude range: {float(z500.lat.min()):.1f}°N to {float(z500.lat.max()):.1f}°N")
    print(f"   Longitude range: {float(z500.lon.min()):.1f}°E to {float(z500.lon.max()):.1f}°E")
    
    # Convert to float32 for memory efficiency (blocking doesn't need float64 precision)
    print(f"\n   Converting z500 to float32 for memory efficiency...")
    z500 = z500.astype("float32")
    
    total_start = time.time()
    
    # Step 1: Compute anomalies
    print(f"\n[Step 1/4] Computing anomalies (day-of-year climatology)...")
    step_start = time.time()
    z500_anom = compute_anomalies(z500)
    print("Mean z500 over 50-80N:", float(z500.sel(lat=slice(50, 80)).mean()))
    print("Mean anomaly over 50-80N:", float(z500_anom.sel(lat=slice(50, 80)).mean()))
    print("90th pct of anomalies over 50-80N:",
          float(z500_anom.sel(lat=slice(50, 80)).quantile(0.9)))
    step_time = time.time() - step_start
    print(f"   ✅ Completed in {step_time:.2f} seconds")
    print(f"   Output shape: {z500_anom.shape}")
    # Quick diagnostics for sanity checking
    lat_band = slice(50.0, 80.0)
    mean_z500_band = float(z500.sel(lat=lat_band).mean())
    mean_anom_band = float(z500_anom.sel(lat=lat_band).mean())
    pct90_anom_band = float(z500_anom.sel(lat=lat_band).quantile(0.9))
    print(f"   Mean z500 over 50-80N: {mean_z500_band:.2f} m")
    print(f"   Mean anomaly over 50-80N: {mean_anom_band:.2f} m")
    print(f"   90th pct of anomalies over 50-80N: {pct90_anom_band:.2f} m")
    
    # Apply anomaly scaling (for variance calibration of emulator forecasts)
    if anomaly_scale != 1.0:
        print(f"\n   📐 Applying anomaly scaling factor: {anomaly_scale:.4f}")
        z500_anom = z500_anom * anomaly_scale
        # Report scaled diagnostics
        mean_anom_scaled = float(z500_anom.sel(lat=lat_band).mean())
        pct90_anom_scaled = float(z500_anom.sel(lat=lat_band).quantile(0.9))
        print(f"   Scaled mean anomaly over 50-80N: {mean_anom_scaled:.2f} m")
        print(f"   Scaled 90th pct of anomalies over 50-80N: {pct90_anom_scaled:.2f} m")
    
    # Step 2: Compute Nth percentile threshold(s)
    print(f"\n[Step 2/4] Computing AREA-WEIGHTED {percentile:.0f}th percentile threshold(s)...")
    step_start = time.time()
    if use_monthly_thresholds:
        print(f"   Method: 3-month rolling window per month (Woollings et al., 2018)")
        print(f"   Threshold computed from: {lat_min}°N to {lat_max}°N")
        print(f"   Using fast vectorized computation with area-weighting (cos φ)")
        threshold_90 = compute_monthly_thresholds_rolling_fast(z500_anom, lat_min, lat_max, percentile=percentile)
        threshold_method = f'3-month rolling window per month (Woollings et al., 2018, area-weighted, {percentile:.0f}th percentile)'
    else:
        print(f"   Method: Single threshold for all data (area-weighted)")
        print(f"   Threshold computed from: {lat_min}°N to {lat_max}°N")
        threshold_90 = compute_threshold_percentile(z500_anom, lat_min, lat_max, percentile=percentile)
        threshold_method = f'Single threshold for all data (area-weighted, {percentile:.0f}th percentile)'
    step_time = time.time() - step_start
    print(f"   ✅ Completed in {step_time:.2f} seconds")
    if isinstance(threshold_90, dict):
        print(f"   Monthly thresholds computed: {len(threshold_90)} months")
        for month, thresh in sorted(threshold_90.items()):
            month_name = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][month-1]
            print(f"      {month_name} ({month}): {thresh:.2f} m")
    elif hasattr(threshold_90, 'shape'):
        print(f"   Threshold shape: {threshold_90.shape}")
    else:
        print(f"   Threshold value: {threshold_90:.2f} m")
    
    # Step 3: Create blocking mask (applies threshold to ALL grid points)
    print(f"\n[Step 3/4] Creating blocking mask (applying threshold to all NH grid points)...")
    if isinstance(threshold_90, dict):
        print(f"   Using fast vectorized broadcasted comparison (monthly thresholds)")
    step_start = time.time()
    blocked_mask = create_blocking_mask_fast(z500_anom, threshold_90)
    step_time = time.time() - step_start
    print(f"   ✅ Completed in {step_time:.2f} seconds")
    print(f"   Mask shape: {blocked_mask.shape}")
    n_blocked = int(blocked_mask.sum().values)
    total_points = blocked_mask.size
    print(f"   Blocked points: {n_blocked:,}/{total_points:,} ({100*n_blocked/total_points:.2f}%)")
    
    # Step 3.5: Restrict to core season if requested
    # This masks out shoulder months (Nov, Mar for DJF; May, Sep for JJA)
    # while keeping them in the anomaly/threshold computation
    detection_season_info = "all months in data"
    if restrict_to_core_season:
        months_in_data = set(blocked_mask.time.dt.month.values)
        
        # Determine core season based on months present
        # NDJFM (11,12,1,2,3) -> DJF (12,1,2)
        # MJJAS (5,6,7,8,9) -> JJA (6,7,8)
        if months_in_data == {11, 12, 1, 2, 3} or months_in_data == {12, 1, 2}:
            core_months = {12, 1, 2}
            detection_season_info = "DJF only (Dec, Jan, Feb)"
        elif months_in_data == {5, 6, 7, 8, 9} or months_in_data == {6, 7, 8}:
            core_months = {6, 7, 8}
            detection_season_info = "JJA only (Jun, Jul, Aug)"
        elif 12 in months_in_data or 1 in months_in_data or 2 in months_in_data:
            # Winter-like data, restrict to DJF
            core_months = {12, 1, 2}
            detection_season_info = "DJF only (Dec, Jan, Feb)"
        elif 6 in months_in_data or 7 in months_in_data or 8 in months_in_data:
            # Summer-like data, restrict to JJA
            core_months = {6, 7, 8}
            detection_season_info = "JJA only (Jun, Jul, Aug)"
        else:
            # Unknown season pattern, don't restrict
            core_months = months_in_data
            detection_season_info = "all months (unknown season pattern)"
        
        # Create mask for core season months
        core_season_mask = np.isin(blocked_mask.time.dt.month.values, list(core_months))
        
        # Zero out blocking in shoulder months (keep the timesteps but mark as non-blocked)
        # This ensures event tracking doesn't start/continue in shoulder months
        blocked_mask_for_events = blocked_mask.copy()
        blocked_mask_for_events.values[~core_season_mask, :, :] = 0
        
        n_blocked_core = int(blocked_mask_for_events.sum().values)
        n_shoulder_removed = n_blocked - n_blocked_core
        print(f"\n   🎯 Restricting event detection to core season: {detection_season_info}")
        print(f"   Shoulder months masked out: {n_shoulder_removed:,} blocked points removed")
        print(f"   Core season blocked points: {n_blocked_core:,}")
    else:
        blocked_mask_for_events = blocked_mask
        print(f"\n   ℹ️ Event detection in: {detection_season_info}")
    
    # Step 4: Identify events and calculate frequency
    print(f"\n[Step 4/4] Identifying blocking events (spatial extent + temporal persistence)...")
    print(f"   Criteria: min_area={min_area/1e6:.1f}×10⁶ km², min_duration={min_duration} days, min_overlap={min_overlap:.1%}")
    step_start = time.time()
    blocking_frequency, event_info = identify_blocking_events(
        blocked_mask_for_events,  # Use filtered mask (core season only if restrict_to_core_season=True)
        min_area=min_area,
        min_duration=min_duration,
        min_overlap=min_overlap
    )
    step_time = time.time() - step_start
    print(f"   ✅ Completed in {step_time:.2f} seconds")
    print(f"   Detected {event_info['num_events']} blocking events")
    print(f"   Blocking frequency shape: {blocking_frequency.shape}")
    print(f"   Mean blocking frequency: {float(blocking_frequency.mean())*100:.2f}%")
    print(f"   Max blocking frequency: {float(blocking_frequency.max())*100:.2f}%")
    
    total_time = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"✅ ANO BLOCKING DETECTION COMPLETE")
    print(f"   Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"{'='*70}\n")
    
    # Add intermediate results to event_info for reference
    event_info['z500_anom'] = z500_anom
    event_info['blocked_mask'] = blocked_mask_for_events  # Use filtered mask (core season only)
    event_info['blocked_mask_full'] = blocked_mask  # Keep original for reference if needed
    event_info['threshold_90'] = threshold_90
    event_info['parameters'] = {
        'lat_min': lat_min,
        'lat_max': lat_max,
        'threshold_computed_from': f'{lat_min}-{lat_max}°N',
        'threshold_method': threshold_method,
        'use_monthly_thresholds': use_monthly_thresholds,
        'percentile': percentile,
        'blocking_detected': 'All NH grid points',
        'min_area_km2': min_area,
        'min_duration_days': min_duration,
        'min_overlap': min_overlap,
        'earth_radius_km': EARTH_RADIUS_KM,
        'anomaly_scale': anomaly_scale,
        'restrict_to_core_season': restrict_to_core_season,
        'detection_season': detection_season_info
    }
    
    return blocking_frequency, event_info
