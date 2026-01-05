import numpy as np
import xarray as xr
from pathlib import Path
from typing import List, Union, Optional
from scipy import ndimage
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import standardized preprocessing from Process module
from .Process import standardize_coordinates


def prepare_for_blocking_analysis(z500_data, lat_min=15.0, lat_max=86.0):
    """
    Prepare z500 data for blocking analysis by applying coordinate standardization
    and then subsetting to the blocking analysis domain.
    
    This function:
    1. Applies standardized coordinate preprocessing (lat orientation, lon 0-360, cyclic point)
       via standardize_coordinates() from Process module
    2. Subsets latitude to the specified blocking analysis range (default: 15°N to 86°N)
    
    Note: If z500_data comes from process_multiple_files(), it will already have
    standardized coordinates. In that case, this function primarily performs
    the latitude subsetting for the blocking domain.
    
    Parameters
    ----------
    z500_data : xarray.DataArray
        Input z500 data with dimensions (time, lat, lon).
        Can be raw or already preprocessed by process_multiple_files().
    lat_min : float, optional
        Minimum latitude in degrees North for blocking domain (default: 15.0)
    lat_max : float, optional
        Maximum latitude in degrees North for blocking domain (default: 86.0)
    
    Returns
    -------
    z500_prepared : xarray.DataArray
        Prepared z500 data with:
        - Standardized coordinates (lat northward, lon 0-360, cyclic point)
        - Latitude filtered to [lat_min, lat_max]
        Shape: (time, n_lats_subset, n_lons+1)
    
    Examples
    --------
    >>> z500_combined = process_multiple_files(range(1000, 1010), season='DJF')
    >>> z500_ready = prepare_for_blocking_analysis(z500_combined)
    >>> print(z500_ready.shape)  # (900, ~20, 129) for 10 years DJF
    
    >>> # Can also be used with raw data
    >>> z500_raw = xr.open_dataset('raw_data.nc')['z500']
    >>> z500_ready = prepare_for_blocking_analysis(z500_raw)
    """
    
    # Step 1: Apply standardized coordinate preprocessing
    # (This is idempotent - safe to call on already-standardized data)
    z500_standardized = standardize_coordinates(z500_data)
    
    # Step 2: Subset latitude to blocking analysis domain
    lat_mask = (z500_standardized.lat >= lat_min) & (z500_standardized.lat <= lat_max)
    z500_prepared = z500_standardized.sel(lat=lat_mask)
    
    # Update metadata
    z500_prepared.attrs['lat_range'] = f'{lat_min}°N to {lat_max}°N'
    z500_prepared.attrs['processing'] = 'Prepared for blocking analysis (domain subset)'
    
    return z500_prepared


def compute_meridional_gradients(z500_data, delta_lat=15.0):
    """
    Compute the three meridional geopotential-height gradients for ABS blocking detection.
    GLOBAL VERSION: Computes gradients for all valid central latitudes where surrounding
    points (φ-30°, φ-15°, φ+15°) exist in the data.
    
    The ABS (Atmospheric Blocking System) method uses three north-south gradients:
    1. GHGN (Northern gradient): [Z(φ+Δ) - Z(φ)] / Δ
    2. GHGS (Southern gradient): [Z(φ) - Z(φ-Δ)] / Δ
    3. GHGS2 (Secondary southern gradient): [Z(φ-Δ) - Z(φ-2Δ)] / Δ
    
    Parameters
    ----------
    z500_data : xarray.DataArray
        Geopotential height data at 500 hPa with dimensions (time, lat, lon).
    delta_lat : float, optional
        Latitude spacing for gradient computation in degrees (default: 15.0).
    
    Returns
    -------
    gradients : xarray.Dataset
        Dataset containing GHGN, GHGS, GHGS2 (time, lat, lon) in m/degree
        Latitude range will be all valid central latitudes where φ-30° and φ+15° exist.
    """
    
    data_lat_min = float(z500_data.lat.min())
    data_lat_max = float(z500_data.lat.max())
    
    # Determine valid central latitudes automatically
    # Need φ-30° for GHGS2, φ-15° for GHGS, φ+15° for GHGN
    # So valid central latitudes are where all three surrounding points exist
    central_lat_min = data_lat_min + 2 * delta_lat  # Need φ-30° to exist
    central_lat_max = data_lat_max - delta_lat      # Need φ+15° to exist
    
    print(f"\n🌍 GLOBAL ABS: Computing gradients for all valid latitudes")
    print(f"   Data range: [{data_lat_min:.1f}°N, {data_lat_max:.1f}°N]")
    print(f"   Valid central latitudes: [{central_lat_min:.1f}°N, {central_lat_max:.1f}°N]")
    print(f"   (Requires φ-30° and φ+15° to exist)")
    
    # Select all valid central latitudes for output
    central_lat_mask = (z500_data.lat >= central_lat_min) & (z500_data.lat <= central_lat_max)
    central_lats = z500_data.lat.sel(lat=central_lat_mask)
    
    # Initialize output arrays
    n_times = len(z500_data.time)
    n_central_lats = len(central_lats)
    n_lons = len(z500_data.lon)
    
    # Create output DataArrays
    dims = ['time', 'lat', 'lon']
    coords = {
        'time': z500_data.time,
        'lat': central_lats,
        'lon': z500_data.lon
    }
    
    GHGN = xr.DataArray(
        np.full((n_times, n_central_lats, n_lons), np.nan),
        dims=dims,
        coords=coords,
        name='GHGN'
    )
    
    GHGS = xr.DataArray(
        np.full((n_times, n_central_lats, n_lons), np.nan),
        dims=dims,
        coords=coords,
        name='GHGS'
    )
    
    GHGS2 = xr.DataArray(
        np.full((n_times, n_central_lats, n_lons), np.nan),
        dims=dims,
        coords=coords,
        name='GHGS2'
    )
    
    # Compute gradients for each central latitude
    for i, center_lat in enumerate(central_lats.values):
        # Find the nearest latitude values to the required points
        lat_north = center_lat + delta_lat
        lat_south = center_lat - delta_lat
        lat_south2 = center_lat - 2 * delta_lat
        
        # Use nearest neighbor selection with larger tolerance for Gaussian grids
        # T42 grid spacing is ~2.8°, so use tolerance of 5° to be safe
        try:
            z_center = z500_data.sel(lat=center_lat, method='nearest', tolerance=5.0)
            z_north = z500_data.sel(lat=lat_north, method='nearest', tolerance=5.0)
            z_south = z500_data.sel(lat=lat_south, method='nearest', tolerance=5.0)
            z_south2 = z500_data.sel(lat=lat_south2, method='nearest', tolerance=5.0)
            
            # Get actual latitude values used (for accurate gradient calculation)
            actual_lat_center = float(z_center.lat)
            actual_lat_north = float(z_north.lat)
            actual_lat_south = float(z_south.lat)
            actual_lat_south2 = float(z_south2.lat)
            
            # Compute gradients using actual latitude differences (in m/degree)
            delta_north = actual_lat_north - actual_lat_center
            delta_south = actual_lat_center - actual_lat_south
            delta_south2 = actual_lat_south - actual_lat_south2
            
            # Avoid division by zero
            if abs(delta_north) > 0.1:
                GHGN[:, i, :] = (z_north - z_center) / delta_north
            if abs(delta_south) > 0.1:
                GHGS[:, i, :] = (z_center - z_south) / delta_south
            if abs(delta_south2) > 0.1:
                GHGS2[:, i, :] = (z_south - z_south2) / delta_south2
            
        except (KeyError, ValueError) as e:
            # If any latitude is not found, leave as NaN
            continue
    
    # Create output dataset
    gradients = xr.Dataset({
        'GHGN': GHGN,
        'GHGS': GHGS,
        'GHGS2': GHGS2
    })
    
    gradients.attrs['delta_lat'] = delta_lat
    gradients.attrs['central_lat_range'] = f'{float(central_lats.min()):.1f}°N to {float(central_lats.max()):.1f}°N'
    gradients.attrs['method'] = 'ABS meridional gradients (Global, Gaussian grid compatible)'
    
    return gradients

def apply_longitudinal_smoothing(data, window_degrees=7.5, n_lon=128):
    """
    Apply longitudinal averaging to gradient data as required by ABS method.
    
    The ABS method requires averaging over Δ/2 = 7.5° in longitude before
    applying threshold criteria.
    
    Parameters
    ----------
    data : xarray.DataArray
        Gradient data with longitude dimension
    window_degrees : float, optional
        Smoothing window width in degrees (default: 7.5)
    n_lon : int, optional
        Number of longitude points (default: 128 for T42)
    
    Returns
    -------
    smoothed_data : xarray.DataArray
        Longitudinally averaged data
    """
    
    # Calculate window size in grid points
    lon_spacing = 360.0 / n_lon
    window_size = int(np.round(window_degrees / lon_spacing))
    if window_size < 1:
        window_size = 1
    
    # Apply rolling mean with center=True
    # Note: This assumes the data has been prepared with a cyclic longitude point
    smoothed = data.rolling(lon=window_size, center=True, min_periods=1).mean()
    
    return smoothed


def detect_abs_blocking(gradients, z500_data, 
                        ghgn_threshold=-10.0, 
                        ghgs_threshold=0.0, 
                        ghgs2_threshold=-5.0,
                        min_area_km2=5e5,
                        min_duration_days=5,
                        n_lon=128):
    """
    Detect atmospheric blocking events using the ABS (Atmospheric Blocking System) method.
    
    This implements the full ABS criteria:
    0) Longitudinal averaging: Smooth gradients over Δ/2 = 7.5° in longitude
    A) Gradient reversal: GHGN < -10 and GHGS > 0 (applied to smoothed gradients)
    B) Subtropical filter: GHGS2 < -5 (applied to smoothed gradients)
    C) Spatial extent: Contiguous regions >= 5×10⁵ km²
    D) Temporal persistence: >= 5 consecutive days with overlap
    
    Parameters
    ----------
    gradients : xarray.Dataset
        Dataset with GHGN, GHGS, GHGS2 from compute_meridional_gradients()
    z500_data : xarray.DataArray
        Original z500 data (needed for spatial information)
    ghgn_threshold : float, optional
        GHGN threshold in m/degree (default: -10.0)
    ghgs_threshold : float, optional
        GHGS threshold in m/degree (default: 0.0)
    ghgs2_threshold : float, optional
        GHGS2 threshold in m/degree (default: -5.0)
    min_area_km2 : float, optional
        Minimum area for blocking region in km² (default: 5e5)
    min_duration_days : int, optional
        Minimum persistence in days (default: 5)
    n_lon : int, optional
        Number of longitude points (default: 128 for T42 grid)
    
    Returns
    -------
    blocking_results : dict
        Dictionary containing:
        - 'instantaneous_mask': Boolean array (time, lat, lon) of blocking points
        - 'spatial_filtered_mask': After applying area criterion
        - 'persistent_events': List of blocking events that meet persistence criterion
        - 'final_blocking_mask': Final mask with only persistent events
        - 'blocking_frequency': Fraction of time each grid point is blocked
        - 'event_count': Number of blocking events detected
        - 'smoothed_gradients': Dict with zonally-averaged GHGN, GHGS, GHGS2
    
    Examples
    --------
    >>> gradients = compute_meridional_gradients(z500_ready)
    >>> results = detect_abs_blocking(gradients, z500_ready)
    >>> print(f"Detected {results['event_count']} blocking events")
    """
    
    # Step 0: Apply longitudinal averaging (Δ/2 = 7.5°) before thresholds
    #print("Step 0: Applying longitudinal averaging (Δ/2 = 7.5°)...")
    
    lon_spacing = 360.0 / n_lon
    window_degrees = 7.5
    window_size = int(np.round(window_degrees / lon_spacing))
    
    # print(f"   Longitude spacing: {lon_spacing:.3f}°")
    # print(f"   Averaging window: {window_degrees}° ({window_size} grid points)")
    
    # Apply zonal (longitudinal) smoothing to all three gradients
    GHGN_smoothed = apply_longitudinal_smoothing(gradients['GHGN'], window_degrees=7.5, n_lon=n_lon)
    GHGS_smoothed = apply_longitudinal_smoothing(gradients['GHGS'], window_degrees=7.5, n_lon=n_lon)
    GHGS2_smoothed = apply_longitudinal_smoothing(gradients['GHGS2'], window_degrees=7.5, n_lon=n_lon)
    
    # # Report smoothing effect
    # print(f"   Smoothing reduces GHGN std by {100*(1 - float(GHGN_smoothed.std())/float(gradients['GHGN'].std())):.1f}%")
    
    # Step 1: Apply Criteria A + B (instantaneous blocking points) to smoothed gradients
    print("\nStep 1: Applying gradient criteria (A + B) to smoothed gradients...")
    print(f"   Thresholds: GHGN<{ghgn_threshold}, GHGS>{ghgs_threshold}, GHGS2<{ghgs2_threshold}")
    
    criterion_A = (GHGN_smoothed < ghgn_threshold) & (GHGS_smoothed > ghgs_threshold)
    criterion_B = (GHGS2_smoothed < ghgs2_threshold)
    
    instantaneous_mask = criterion_A & criterion_B
    
    n_blocked_points = int(instantaneous_mask.sum())
    total_points = instantaneous_mask.size
    print(f"   {n_blocked_points}/{total_points} points meet gradient criteria "
          f"({100*n_blocked_points/total_points:.2f}%)")
    
    # Step 2: Apply Criterion C (spatial extent)
    print("\nStep 2: Applying spatial extent criterion (C)...")
    
    # Detect if cyclic point is present
    lon_vals = gradients['lon'].values
    n_lon_total = len(lon_vals)
    if n_lon_total > 1:
        lon_spacing = lon_vals[1] - lon_vals[0]
        expected_last = lon_vals[0] + (n_lon_total - 1) * lon_spacing
        has_cyclic = abs(lon_vals[-1] - expected_last) > lon_spacing * 0.5
        n_lon_unique = n_lon_total - 1 if has_cyclic else n_lon_total
    else:
        n_lon_unique = n_lon_total
    
    print(f"   Grid: {n_lon_unique} unique longitudes" + 
          (f" (+ 1 cyclic point)" if n_lon_total > n_lon_unique else ""))
    print(f"   Minimum area threshold: {min_area_km2/1e5:.1f}×10⁵ km²")
    
    spatial_filtered_mask = apply_spatial_extent_criterion(
        instantaneous_mask, 
        gradients['lat'].values,
        gradients['lon'].values,
        min_area_km2=min_area_km2
    )
    
    n_spatial_filtered = int(spatial_filtered_mask.sum())
    print(f"   {n_spatial_filtered}/{total_points} points meet spatial criterion "
          f"({100*n_spatial_filtered/total_points:.2f}%)")
    
    # Step 3: Apply Criterion D (temporal persistence)
    print("\nStep 3: Applying temporal persistence criterion (D)...")
    
    persistent_events = track_persistent_blocking(
        spatial_filtered_mask,
        min_duration_days=min_duration_days
    )
    
    print(f"   Detected {len(persistent_events)} persistent blocking events")
    
    # Calculate blocking frequency (fraction of time blocked after all criteria)
    final_mask = create_final_blocking_mask(spatial_filtered_mask, persistent_events)
    blocking_frequency = final_mask.mean(dim='time')
    
    # Prepare results
    results = {
        'instantaneous_mask': instantaneous_mask,
        'spatial_filtered_mask': spatial_filtered_mask,
        'persistent_events': persistent_events,
        'final_blocking_mask': final_mask,
        'blocking_frequency': blocking_frequency,
        'event_count': len(persistent_events),
        'smoothed_gradients': {
            'GHGN': GHGN_smoothed,
            'GHGS': GHGS_smoothed,
            'GHGS2': GHGS2_smoothed
        }
    }
    
    return results


def apply_spatial_extent_criterion(mask, latitudes, lon_values, min_area_km2=5e5):
    """
    Filter blocking mask by spatial extent using connected component analysis.
    
    Parameters
    ----------
    mask : xarray.DataArray
        Boolean mask (time, lat, lon) of candidate blocking points
    latitudes : numpy.ndarray
        Array of latitude values
    lon_values : numpy.ndarray
        Array of longitude values (will automatically handle cyclic points)
    min_area_km2 : float
        Minimum area in km²
    
    Returns
    -------
    filtered_mask : xarray.DataArray
        Boolean mask with small regions removed
    """
    
    filtered_mask = mask.copy()
    n_times = len(mask.time)
    
    # Process each time step independently
    for t in range(n_times):
        mask_2d = mask.isel(time=t).values
        
        if not mask_2d.any():
            continue
        
        # Label connected components (8-connectivity to include diagonals)
        labeled, n_features = ndimage.label(mask_2d, structure=np.ones((3,3)))
        
        # Check each region
        for region_id in range(1, n_features + 1):
            region_mask = (labeled == region_id)
            
            # Calculate area of this region using proper spherical geometry
            region_area = calculate_region_area(region_mask, latitudes, lon_values)
            
            # Remove if too small
            if region_area < min_area_km2:
                filtered_mask.values[t, region_mask] = False
    
    return filtered_mask


def calculate_region_area(region_mask, latitudes, lon_values):
    """
    Calculate the area of a region on the sphere using proper spherical geometry.
    
    Uses the accurate formula for Gaussian grids with non-uniform latitude spacing:
    A(φ) = R² × Δλ × [sin(φ + Δφ/2) - sin(φ - Δφ/2)]
    
    Parameters
    ----------
    region_mask : numpy.ndarray
        2D boolean array (lat, lon) marking the region
    latitudes : numpy.ndarray
        Array of latitude values (in degrees)
    lon_values : numpy.ndarray
        Array of longitude values (in degrees). If cyclic point is present,
        it will be detected and excluded from area calculation.
    
    Returns
    -------
    area_km2 : float
        Total area of the region in km²
    """
    R_earth = 6371.0  # km
    
    # Determine actual number of unique longitude points
    # Check if last longitude is cyclic (wraps to 0°/360°)
    n_lon_total = len(lon_values)
    has_cyclic = False
    
    if n_lon_total > 1:
        # Check if last point wraps around (e.g., last ≈ 360° and first ≈ 0°)
        lon_spacing = lon_values[1] - lon_values[0]
        expected_last = lon_values[0] + (n_lon_total - 1) * lon_spacing
        
        # If the last longitude is approximately 360° or wraps to the first
        if abs(lon_values[-1] - expected_last) > lon_spacing * 0.5:
            has_cyclic = True
            n_lon_unique = n_lon_total - 1
        else:
            n_lon_unique = n_lon_total
    else:
        n_lon_unique = n_lon_total
    
    # Longitude spacing in radians (uniform for unique points)
    delta_lon_rad = 2.0 * np.pi / n_lon_unique
    
    total_area = 0.0
    n_lats = len(latitudes)
    
    # Calculate cell area for each latitude band
    for i in range(n_lats):
        # Count cells at this latitude in the region
        # If cyclic point exists, don't count it (it's a duplicate)
        if has_cyclic:
            n_cells_at_lat = region_mask[i, :-1].sum()
        else:
            n_cells_at_lat = region_mask[i, :].sum()
        
        if n_cells_at_lat == 0:
            continue
        
        lat_deg = latitudes[i]
        
        # Calculate latitude bounds (midpoints to adjacent latitudes)
        if i == 0:
            # First latitude: use distance to next point, extrapolate backward
            lat_spacing = latitudes[1] - latitudes[0]
            lat_south = lat_deg - lat_spacing / 2.0
            lat_north = lat_deg + lat_spacing / 2.0
        elif i == n_lats - 1:
            # Last latitude: use distance to previous point, extrapolate forward
            lat_spacing = latitudes[i] - latitudes[i-1]
            lat_south = lat_deg - lat_spacing / 2.0
            lat_north = lat_deg + lat_spacing / 2.0
        else:
            # Middle latitudes: use actual midpoints to neighbors
            lat_south = (lat_deg + latitudes[i-1]) / 2.0
            lat_north = (lat_deg + latitudes[i+1]) / 2.0
        
        # Convert to radians
        lat_south_rad = np.deg2rad(lat_south)
        lat_north_rad = np.deg2rad(lat_north)
        
        # Calculate area of a single cell at this latitude using proper spherical formula
        # A(φ) = R² × Δλ × [sin(φ_north) - sin(φ_south)]
        cell_area = (R_earth ** 2) * delta_lon_rad * (np.sin(lat_north_rad) - np.sin(lat_south_rad))
        
        total_area += cell_area * n_cells_at_lat
    
    return total_area


def track_persistent_blocking(mask, min_duration_days=5):
    """
    Track blocking events through time and filter by persistence.
    
    Parameters
    ----------
    mask : xarray.DataArray
        Boolean mask (time, lat, lon) after spatial filtering
    min_duration_days : int
        Minimum number of consecutive days
    
    Returns
    -------
    persistent_events : list of dict
        Each event contains:
        - 'start_time': Start time index
        - 'end_time': End time index
        - 'duration': Duration in days
        - 'masks': List of 2D masks for each day
        - 'center_lat': Mean latitude of event
        - 'center_lon': Mean longitude of event
    """
    
    n_times = len(mask.time)
    events = []
    
    # Track regions day by day
    current_events = []  # List of ongoing events
    
    for t in range(n_times):
        mask_today = mask.isel(time=t).values
        
        if not mask_today.any():
            # No blocking today - close all current events
            for event in current_events:
                if event['duration'] >= min_duration_days:
                    events.append(event)
            current_events = []
            continue
        
        # Label regions today
        labeled_today, n_regions_today = ndimage.label(mask_today, structure=np.ones((3,3)))
        
        # Match with ongoing events
        matched = [False] * len(current_events)
        used_regions = [False] * n_regions_today
        
        for i, event in enumerate(current_events):
            prev_mask = event['masks'][-1]
            
            # Check overlap with each region today
            for region_id in range(1, n_regions_today + 1):
                if used_regions[region_id - 1]:
                    continue
                
                region_mask_today = (labeled_today == region_id)
                
                # Check if there's overlap
                if (prev_mask & region_mask_today).any():
                    # Extend this event
                    event['end_time'] = t
                    event['duration'] += 1
                    event['masks'].append(region_mask_today)
                    matched[i] = True
                    used_regions[region_id - 1] = True
                    break
        
        # Close unmatched events
        for i, was_matched in enumerate(matched):
            if not was_matched:
                event = current_events[i]
                if event['duration'] >= min_duration_days:
                    events.append(event)
        
        # Remove closed events
        current_events = [e for i, e in enumerate(current_events) if matched[i]]
        
        # Start new events for unmatched regions
        for region_id in range(1, n_regions_today + 1):
            if not used_regions[region_id - 1]:
                region_mask = (labeled_today == region_id)
                
                # Calculate center
                lat_indices, lon_indices = np.where(region_mask)
                center_lat = mask.lat.values[lat_indices].mean()
                center_lon = mask.lon.values[lon_indices].mean()
                
                new_event = {
                    'start_time': t,
                    'end_time': t,
                    'duration': 1,
                    'masks': [region_mask],
                    'center_lat': center_lat,
                    'center_lon': center_lon
                }
                current_events.append(new_event)
    
    # Close any remaining events at the end
    for event in current_events:
        if event['duration'] >= min_duration_days:
            events.append(event)
    
    return events


def create_final_blocking_mask(spatial_mask, persistent_events):
    """
    Create final blocking mask containing only persistent events.
    
    Parameters
    ----------
    spatial_mask : xarray.DataArray
        Spatial filtered mask (template for output)
    persistent_events : list
        List of persistent events from track_persistent_blocking()
    
    Returns
    -------
    final_mask : xarray.DataArray
        Boolean mask with only persistent blocking events
    """
    
    final_mask = xr.zeros_like(spatial_mask, dtype=bool)
    
    for event in persistent_events:
        for t_offset, mask_2d in enumerate(event['masks']):
            t = event['start_time'] + t_offset
            final_mask.values[t, mask_2d] = True
    
    return final_mask


def abs_blocking_frequency(z500_data,
                           ghgn_threshold=-10.0,
                           ghgs_threshold=0.0,
                           ghgs2_threshold=-5.0,
                           min_area_km2=5e5,
                           min_duration_days=5,
                           lat_min=15.0,
                           lat_max=90.0,
                           delta_lat=15.0):
    """
    Perform complete ABS (Atmospheric Blocking System) blocking detection workflow.
    GLOBAL VERSION: Computes blocking at all valid latitudes automatically.
    
    This is a high-level wrapper that orchestrates all steps of the ABS method:
    1. Data preparation (add cyclic point, trim latitudes to lat_min-lat_max)
    2. Compute meridional gradients (GHGN, GHGS, GHGS2) for ALL valid central latitudes
       where φ-30° and φ+15° exist in the data (~45-73°N for PlaSim, ~45-75°N for ERA5)
    3. Apply longitudinal averaging (Δ/2 = 7.5°)
    4. Apply ABS criteria globally (A: gradient reversal, B: subtropical filter, 
                                     C: spatial extent, D: temporal persistence)
    5. Calculate blocking frequency across all valid latitudes
    
    Parameters
    ----------
    z500_data : xarray.DataArray
        Raw 500 hPa geopotential height data with dimensions (time, lat, lon).
        Should contain Northern Hemisphere data, will be trimmed to lat_min-lat_max.
    ghgn_threshold : float, optional
        Northern gradient threshold in m/degree (default: -10.0)
    ghgs_threshold : float, optional
        Southern gradient threshold in m/degree (default: 0.0)
    ghgs2_threshold : float, optional
        Secondary southern gradient threshold in m/degree (default: -5.0)
    min_area_km2 : float, optional
        Minimum blocking region area in km² (default: 5e5 = 5×10⁵ km²)
    min_duration_days : int, optional
        Minimum persistence duration in days (default: 5)
    lat_min : float, optional
        Minimum latitude for data preparation in degrees N (default: 15.0)
    lat_max : float, optional
        Maximum latitude for data preparation in degrees N (default: 90.0)
        Set to 90.0 to use all available data up to the pole
    delta_lat : float, optional
        Latitude spacing for gradient computation in degrees (default: 15.0)
    
    Returns
    -------
    blocking_frequency : xarray.DataArray
        2-D array (lat, lon) showing the fraction of time each grid point 
        experiences blocking (0 = never blocked, 1 = always blocked).
        Covers ALL valid central latitudes where gradients can be computed.
        Typical range: ~45°N to ~73°N (PlaSim) or ~45°N to ~75°N (ERA5).
    
    Notes
    -----
    - Valid central latitudes are automatically determined based on data availability
    - Central lat range = [data_lat_min + 30°, data_lat_max - 15°]
    - n_lon is automatically inferred from the input data
    - Blocking regions are detected globally across all valid latitudes
    - The function prints progress information during execution
    
    Examples
    --------
    >>> # Global blocking detection in DJF season
    >>> z500_djf = process_multiple_files(range(1000, 1010), season='DJF')
    >>> blocking_freq = abs_blocking_frequency(z500_djf)
    >>> print(f"Latitude range: {float(blocking_freq.lat.min()):.1f}-{float(blocking_freq.lat.max()):.1f}°N")
    >>> print(f"Max blocking frequency: {float(blocking_freq.max()):.3f}")
    >>> 
    >>> # Subset to specific region after computation
    >>> blocking_35_80N = blocking_freq.sel(lat=slice(35, 80))
    >>> 
    >>> # With custom thresholds
    >>> blocking_freq = abs_blocking_frequency(
    ...     z500_djf, 
    ...     ghgn_threshold=-8.0,
    ...     min_area_km2=6e5,
    ...     min_duration_days=7
    ... )
    
    See Also
    --------
    prepare_for_blocking_analysis : Data preparation step
    compute_meridional_gradients : Gradient computation step (now global)
    detect_abs_blocking : Full detection with all criteria
    """
    

    
    # Step 1: Prepare data for blocking analysis

    
    z500_prepared = prepare_for_blocking_analysis(
        z500_data, 
        lat_min=lat_min, 
        lat_max=lat_max
    )

    # Step 2: Compute meridional gradients globally
 
    gradients = compute_meridional_gradients(
        z500_prepared,
        delta_lat=delta_lat
    )
    

    
    # Step 3 & 4: Apply ABS criteria (includes longitudinal averaging internally)

    
    # Infer n_lon from prepared data
    n_lon_total = len(z500_prepared.lon)
    lon_vals = z500_prepared.lon.values
    if n_lon_total > 1:
        lon_spacing = lon_vals[1] - lon_vals[0]
        expected_last = lon_vals[0] + (n_lon_total - 1) * lon_spacing
        has_cyclic = abs(lon_vals[-1] - expected_last) > lon_spacing * 0.5
        n_lon = n_lon_total - 1 if has_cyclic else n_lon_total
    else:
        n_lon = n_lon_total
    
    blocking_results = detect_abs_blocking(
        gradients,
        z500_prepared,
        ghgn_threshold=ghgn_threshold,
        ghgs_threshold=ghgs_threshold,
        ghgs2_threshold=ghgs2_threshold,
        min_area_km2=min_area_km2,
        min_duration_days=min_duration_days,
        n_lon=n_lon
    )
    
    # Step 5: Extract blocking frequency

    blocking_frequency = blocking_results['blocking_frequency']
    
    # Add metadata
    blocking_frequency.name = 'blocking_frequency'
    lat_range_str = f'{float(blocking_frequency.lat.min()):.1f}-{float(blocking_frequency.lat.max()):.1f}°N'
    blocking_frequency.attrs = {
        'long_name': 'Atmospheric blocking frequency',
        'units': 'fraction',
        'description': 'Fraction of time each grid point experiences blocking (Global ABS)',
        'method': 'ABS (Atmospheric Blocking System) - Global computation',
        'thresholds': f'GHGN<{ghgn_threshold}, GHGS>{ghgs_threshold}, GHGS2<{ghgs2_threshold}',
        'min_area_km2': min_area_km2,
        'min_duration_days': min_duration_days,
        'delta_lat': delta_lat,
        'central_lat_range': lat_range_str,
        'note': 'Gradients computed for all valid latitudes where φ-30° and φ+15° exist'
    }
    

    
    return blocking_frequency


