import numpy as np
import xarray as xr
from pathlib import Path
from typing import List, Union, Optional
from scipy import ndimage
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import standardized preprocessing from Process module
from Process import standardize_coordinates


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


def compute_meridional_gradients(z500_data, delta_lat=15.0, central_lat_min=45.0, central_lat_max=67.0):
    """
    Compute the three meridional geopotential-height gradients for ABS blocking detection.
    FIXED VERSION: Works with Gaussian grids (non-uniform latitude spacing).
    
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
    central_lat_min : float, optional
        Minimum central latitude for gradient computation in degrees N (default: 45.0).
    central_lat_max : float, optional
        Maximum central latitude for gradient computation in degrees N (default: 67.0).
    
    Returns
    -------
    gradients : xarray.Dataset
        Dataset containing GHGN, GHGS, GHGS2 (time, lat, lon) in m/degree
    """
    
    # Verify input data has required latitude range
    lat_min_required = central_lat_min - 2 * delta_lat
    lat_max_required = central_lat_max + delta_lat
    
    data_lat_min = float(z500_data.lat.min())
    data_lat_max = float(z500_data.lat.max())
    
    # More lenient check: allow some margin for Gaussian grids
    # We use nearest neighbor with tolerance=5.0, so exact bounds aren't critical
    # Allow up to 5° margin on each side since we use tolerance=5.0 in nearest neighbor selection
    if data_lat_min > lat_min_required + 5.0 or data_lat_max < lat_max_required - 5.0:
        raise ValueError(
            f"Input data latitude range [{data_lat_min:.1f}, {data_lat_max:.1f}] "
            f"is insufficient. Need approximately [{lat_min_required:.1f}, {lat_max_required:.1f}]. "
            f"(Allowing 5° margin for nearest neighbor selection)"
        )
    
    # Select only central latitudes for output
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
    gradients.attrs['central_lat_range'] = f'{central_lat_min}°N to {central_lat_max}°N'
    gradients.attrs['method'] = 'ABS meridional gradients (Gaussian grid compatible)'
    
    return gradients

def apply_longitudinal_smoothing(data, window_degrees=7.5, n_lon=128):
    """
    Apply longitudinal averaging to gradient data as required by ABS method.
    Uses periodic boundary conditions (wrapping).
    
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
    
    # Check for existing cyclic point
    # Usually data comes with 0 and 360 (duplicate)
    # If we pad blindly, we duplicate the duplicate -> bad smoothing
    
    lon_vals = data.lon.values
    n_lon_total = len(lon_vals)
    has_cyclic_point = False
    
    if n_lon_total > 1:
        # Check if last point wraps around (e.g., last ≈ 360° and first ≈ 0°)
        # Or if the range covers 360 degrees (e.g. 0 to 360)
        lon_range = lon_vals[-1] - lon_vals[0]
        
        # Case 1: Last point is same as first (modulo 360) - explicit cyclic point
        # e.g. 0, 10, ..., 350, 360
        if abs(lon_range - 360.0) < 1e-4:
             has_cyclic_point = True
             
        # Case 2: Standard check for missing cyclic point (e.g. 0, ..., 350)
        # expected_last would be 360 if we added one more step
        else:
            lon_spacing_est = lon_vals[1] - lon_vals[0]
            expected_last = lon_vals[0] + (n_lon_total - 1) * lon_spacing_est
            if abs(lon_vals[-1] - expected_last) > lon_spacing_est * 0.5:
                # This branch was the original logic, but it's a bit ambiguous what it catches
                # If spacing is uniform, this shouldn't trigger unless there's a gap
                pass
    
    # Prepare data for smoothing: remove cyclic point if present
    if has_cyclic_point:
        data_unique = data.isel(lon=slice(0, -1))
    else:
        data_unique = data
        
    # Pad data circularly to handle periodic boundary correctly
    pad_width = window_size // 2 + 1  # Add extra padding to be safe
    
    # Use xarray padding with wrap
    try:
        # Pad unique data with wrap (correct periodic boundary)
        data_padded = data_unique.pad(lon=pad_width, mode='wrap')
        
        # Apply rolling mean
        smoothed_padded = data_padded.rolling(lon=window_size, center=True, min_periods=1).mean()
        
        # Slice back to unique size
        smoothed_unique = smoothed_padded.isel(lon=slice(pad_width, -pad_width))
        
        # Ensure coordinate alignment
        smoothed_unique['lon'] = data_unique.lon
        
        # Re-add cyclic point if it was originally present
        if has_cyclic_point:
            # Create the cyclic point by copying the first point (0°) to the end
            # but assign it the original last longitude value (360°)
            cyclic_slice = smoothed_unique.isel(lon=0)
            cyclic_slice['lon'] = data.lon.values[-1]
            
            # Concatenate
            smoothed = xr.concat([smoothed_unique, cyclic_slice], dim='lon')
        else:
            smoothed = smoothed_unique
            
    except Exception as e:
        print(f"Warning: Periodic smoothing failed ({e}), falling back to simple rolling.")
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
        lon_range = lon_vals[-1] - lon_vals[0]
        if abs(lon_range - 360.0) < 1e-4:
            has_cyclic = True
            n_lon_unique = n_lon_total - 1
        else:
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


def identify_spatial_regions_periodic(mask_2d):
    """
    Identify connected regions in a 2D binary mask with PERIODIC BOUNDARY HANDLING.
    Uses 8-connectivity (neighbors + diagonals) and handles longitude wrapping.
    
    Parameters
    ----------
    mask_2d : np.ndarray
        2D binary array (lat, lon)
        
    Returns
    -------
    labeled : np.ndarray
        Labeled array with same shape as input
    num_features : int
        Number of unique features found
    """
    # Check for existing cyclic point (duplicate column)
    rows, cols = mask_2d.shape
    has_cyclic_point = False
    if cols > 1 and np.array_equal(mask_2d[:, 0], mask_2d[:, -1]):
        has_cyclic_point = True

    if has_cyclic_point:
        mask_unique = mask_2d[:, :-1]
    else:
        mask_unique = mask_2d

    # Append start column to end for periodicity check (ndimage needs this connectivity)
    mask_periodic = np.concatenate([mask_unique, mask_unique[:, :1]], axis=1)

    # 8-connectivity structure (includes diagonals)
    structure = np.ones((3, 3), dtype=int)

    # Label connected components
    labeled_periodic, num_features = ndimage.label(mask_periodic, structure=structure)

    # Merge labels that wrap around the boundary
    labels_start = labeled_periodic[:, 0]
    labels_end = labeled_periodic[:, -1]
    
    # Find rows where both ends are part of a blob
    connected_rows = (labels_start > 0) & (labels_end > 0)

    if np.any(connected_rows):
        # Identify pairs of labels that need merging
        pairs = np.column_stack((labels_start[connected_rows], labels_end[connected_rows]))
        
        # Union-Find / Graph merge logic
        adj = {}
        for l1, l2 in pairs:
            if l1 == l2: continue
            if l1 not in adj: adj[l1] = set()
            if l2 not in adj: adj[l2] = set()
            adj[l1].add(l2)
            adj[l2].add(l1)
        
        if adj:
            label_map = {}
            visited = set()
            for label in adj:
                if label in visited: continue
                component = set()
                stack = [label]
                while stack:
                    curr = stack.pop()
                    if curr in visited: continue
                    visited.add(curr)
                    component.add(curr)
                    if curr in adj:
                        stack.extend(adj[curr])
                
                target_label = min(component)
                for l in component:
                    label_map[l] = target_label
            
            # Apply mapping if any merges found
            if label_map:
                max_lbl = labeled_periodic.max()
                mapping = np.arange(max_lbl + 1)
                for old, new in label_map.items():
                    mapping[old] = new
                labeled_periodic = mapping[labeled_periodic]
                
                # Renumber to be consecutive (1..N)
                unique_labels = np.unique(labeled_periodic)
                if unique_labels[0] == 0:
                    unique_labels = unique_labels[1:]
                
                dense_map = np.zeros(max_lbl + 1, dtype=int)
                for i, lbl in enumerate(unique_labels):
                    dense_map[lbl] = i + 1
                labeled_periodic = dense_map[labeled_periodic]
                num_features = len(unique_labels)

    # Return correct shape
    if has_cyclic_point:
        labeled = labeled_periodic
    else:
        labeled = labeled_periodic[:, :-1]
        
    return labeled, num_features


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
        
        # Label connected components (8-connectivity with periodic boundaries)
        labeled, n_features = identify_spatial_regions_periodic(mask_2d)
        
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
        lon_range = lon_values[-1] - lon_values[0]
        if abs(lon_range - 360.0) < 1e-4:
            has_cyclic = True
            n_lon_unique = n_lon_total - 1
        else:
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
        labeled_today, n_regions_today = identify_spatial_regions_periodic(mask_today)
        
        # Match with ongoing events
        matched = [False] * len(current_events)
        used_regions = [False] * n_regions_today
        
        for i, event in enumerate(current_events):
            prev_mask = event['masks'][-1]
            
            # Check overlap with each region today
            for region_id in range(1, n_regions_today + 1):
                # Note: We allow multiple events to match the same region (merger case)
                # This prevents a short-lived southern event from "killing" a persistent 
                # northern event by stealing its continuation.
                # However, an event still greedily picks the first region it overlaps with (split case).
                
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
                           lat_max=85.0,
                           central_lat_min=45.0,
                           central_lat_max=67.0,
                           delta_lat=15.0):
    """
    Perform complete ABS (Atmospheric Blocking System) blocking detection workflow.
    
    This is a high-level wrapper that orchestrates all steps of the ABS method:
    1. Data preparation (add cyclic point, trim latitudes)
    2. Compute meridional gradients (GHGN, GHGS, GHGS2)
    3. Apply longitudinal averaging (Δ/2 = 7.5°)
    4. Apply ABS criteria (A: gradient reversal, B: subtropical filter, 
                           C: spatial extent, D: temporal persistence)
    5. Calculate blocking frequency
    
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
        Maximum latitude for data preparation in degrees N (default: 85.0)
    central_lat_min : float, optional
        Minimum central latitude for gradient computation in degrees N (default: 45.0)
    central_lat_max : float, optional
        Maximum central latitude for gradient computation in degrees N (default: 67.0)
    delta_lat : float, optional
        Latitude spacing for gradient computation in degrees (default: 15.0)
    
    Returns
    -------
    blocking_frequency : xarray.DataArray
        2-D array (lat, lon) showing the fraction of time each grid point 
        experiences blocking (0 = never blocked, 1 = always blocked).
        Only includes central latitudes where gradients are computed.
    event_info : dict
        Dictionary containing:
        - 'event_count': Number of blocking events detected
        - 'persistent_events': List of persistent blocking events
        - 'final_blocking_mask': Final mask with only persistent events
        - 'instantaneous_mask': Instantaneous blocking points (before persistence)
        - 'spatial_filtered_mask': After spatial extent criterion
        - 'smoothed_gradients': Dict with zonally-averaged GHGN, GHGS, GHGS2
        - 'gradients': Original gradient dataset
        - 'z500_prepared': Prepared z500 data
        - 'parameters': Dict of all parameters used
    
    Notes
    -----
    - n_lon is automatically inferred from the input data
    - For PlaSim T42 grid: use central_lat_max=67.0 (grid extends to ~82°N)
    - The function prints progress information during execution
    
    Examples
    --------
    >>> # Detect blocking in DJF season
    >>> z500_djf = process_multiple_files(range(1000, 1010), season='DJF')
    >>> blocking_freq, event_info = abs_blocking_frequency(z500_djf)
    >>> print(f"Max blocking frequency: {float(blocking_freq.max()):.3f}")
    >>> print(f"Number of events: {event_info['event_count']}")
    >>> 
    >>> # With custom thresholds
    >>> blocking_freq, event_info = abs_blocking_frequency(
    ...     z500_djf, 
    ...     ghgn_threshold=-8.0,
    ...     min_area_km2=6e5,
    ...     min_duration_days=7
    ... )
    
    See Also
    --------
    prepare_for_blocking_analysis : Data preparation step
    compute_meridional_gradients : Gradient computation step
    detect_abs_blocking : Full detection with all criteria
    """
    
    # Step 1: Prepare data for blocking analysis
    z500_prepared = prepare_for_blocking_analysis(
        z500_data, 
        lat_min=lat_min, 
        lat_max=lat_max
    )

    # Step 2: Compute meridional gradients
    gradients = compute_meridional_gradients(
        z500_prepared,
        delta_lat=delta_lat,
        central_lat_min=central_lat_min,
        central_lat_max=central_lat_max
    )
    
    # Step 3 & 4: Apply ABS criteria (includes longitudinal averaging internally)
    # Infer n_lon from prepared data
    n_lon_total = len(z500_prepared.lon)
    lon_vals = z500_prepared.lon.values
    if n_lon_total > 1:
        # Check if grid wraps around 360 degrees
        # Standard cyclic point means last point is 360 degrees after first point
        lon_range = lon_vals[-1] - lon_vals[0]
        has_cyclic = abs(lon_range - 360.0) < 2.0  # Allow small tolerance
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
    blocking_frequency.attrs = {
        'long_name': 'Atmospheric blocking frequency',
        'units': 'fraction',
        'description': 'Fraction of time each grid point experiences blocking',
        'method': 'ABS (Atmospheric Blocking System)',
        'thresholds': f'GHGN<{ghgn_threshold}, GHGS>{ghgs_threshold}, GHGS2<{ghgs2_threshold}',
        'min_area_km2': min_area_km2,
        'min_duration_days': min_duration_days,
        'delta_lat': delta_lat,
        'central_lat_range': f'{central_lat_min}-{central_lat_max}°N'
    }
    
    # Prepare event_info dictionary (similar to ANO method)
    event_info = {
        'event_count': blocking_results['event_count'],
        'persistent_events': blocking_results['persistent_events'],
        'final_blocking_mask': blocking_results['final_blocking_mask'],
        'instantaneous_mask': blocking_results['instantaneous_mask'],
        'spatial_filtered_mask': blocking_results['spatial_filtered_mask'],
        'smoothed_gradients': blocking_results['smoothed_gradients'],
        'gradients': gradients,
        'z500_prepared': z500_prepared,
        'parameters': {
            'ghgn_threshold': ghgn_threshold,
            'ghgs_threshold': ghgs_threshold,
            'ghgs2_threshold': ghgs2_threshold,
            'min_area_km2': min_area_km2,
            'min_duration_days': min_duration_days,
            'lat_min': lat_min,
            'lat_max': lat_max,
            'data_range': f'{lat_min}-{lat_max}°N',
            'central_lat_min': central_lat_min,
            'central_lat_max': central_lat_max,
            'central_lat_range': f'{central_lat_min}-{central_lat_max}°N',
            'delta_lat': delta_lat,
            'method': 'ABS (Atmospheric Blocking System)',
            'n_lon': n_lon
        }
    }
    
    return blocking_frequency, event_info
