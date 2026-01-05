
import xarray as xr
import numpy as np
import pandas as pd
import os
from glob import glob


def load_seasonal_z500_fast(year, season, extend=False, data_dir='../Data/', resample_daily=True):
    """
    Fast parallel loading of ERA5 Z500 data using xarray's open_mfdataset.
    Optimized for loading multiple years at once.
    
    Parameters:
    -----------
    year : int or list of int
        The year(s) for the season (e.g., 2024 or [1980, 1981, ...] for multiple years)
    season : str
        Season abbreviation. Options: 'DJF', 'MAM', 'JJA', 'SON'
    extend : bool
        If True, load one month before and one month after the season (default: False)
    data_dir : str
        Path to the data directory (default: '../Data/')
    resample_daily : bool
        If True, resample to daily averages during loading (default: False)
        This is much more efficient than resampling after concatenation
    
    Returns:
    --------
    xr.Dataset
        Combined xarray dataset containing Z500 data, sorted by time
    """
    
    # Convert single year to list for uniform processing
    years = [year] if isinstance(year, int) else list(year)
    
    # Define season-to-months mapping
    season_months = {
        'DJF': ['Dec', 'January', 'February'],
        'MAM': ['March', 'April', 'May'],
        'JJA': ['June', 'July', 'August'],
        'SON': ['September', 'October', 'November']
    }
    
    # Define extended months
    extended_months_map = {
        'DJF': {'before': ('November', 11), 'after': ('March', 3)},
        'MAM': {'before': ('February', 2), 'after': ('June', 6)},
        'JJA': {'before': ('May', 5), 'after': ('September', 9)},
        'SON': {'before': ('August', 8), 'after': ('December', 12)}
    }
    
    if season not in season_months:
        raise ValueError(f"Invalid season '{season}'. Must be one of: DJF, MAM, JJA, SON")
    
    # Build list of all file paths to load
    file_paths = []
    
    print(f"Building file list for {season} {years[0]}-{years[-1]}...")
    
    for yr in years:
        # Extended month BEFORE (if requested)
        if extend:
            ext_month_name, ext_month_num = extended_months_map[season]['before']
            ext_year = yr - 1 if season == 'DJF' else yr
            filename = f'ERA5_Z500_NH_{ext_year}_{ext_month_num:02d}_{ext_month_name}.nc'
            filepath = os.path.join(data_dir, filename)
            if os.path.exists(filepath):
                file_paths.append(filepath)
        
        # Core season months
        for month_name in season_months[season]:
            if month_name == 'Dec':
                # December belongs to previous year in file naming
                filename = f'ERA5_Z500_NH_{season}_{yr}_{month_name}{yr-1}.nc'
            else:
                filename = f'ERA5_Z500_NH_{season}_{yr}_{month_name}.nc'
            
            filepath = os.path.join(data_dir, filename)
            if os.path.exists(filepath):
                file_paths.append(filepath)
            else:
                print(f"Warning: File not found: {filepath}")
        
        # Extended month AFTER (if requested)
        if extend:
            ext_month_name, ext_month_num = extended_months_map[season]['after']
            filename = f'ERA5_Z500_NH_{yr}_{ext_month_num:02d}_{ext_month_name}.nc'
            filepath = os.path.join(data_dir, filename)
            if os.path.exists(filepath):
                file_paths.append(filepath)
    
    if not file_paths:
        raise FileNotFoundError(f"No data files found for {season} {years[0]}-{years[-1]} in {data_dir}")
    
    print(f"Loading {len(file_paths)} files in parallel using dask...")
    if resample_daily:
        print(f"  (with daily resampling during load - much faster!)")
    
    # Suppress dask performance warnings
    import warnings
    warnings.filterwarnings('ignore', message='.*chunks separate the stored chunks.*')
    
    # Define preprocessing function for daily resampling
    def preprocess_resample(ds):
        """Resample each file to daily before concatenation"""
        if resample_daily:
            # Resample to daily averages
            ds = ds.resample(valid_time='D').mean()
        return ds
    
    # Open all files at once with parallel loading
    # Use chunks='auto' for optimal chunking strategy
    combined_ds = xr.open_mfdataset(
        file_paths,
        combine='by_coords',
        parallel=True,
        chunks='auto',  # Let dask determine optimal chunking
        engine='netcdf4',
        preprocess=preprocess_resample if resample_daily else None
    )
    
    # Sort by time
    combined_ds = combined_ds.sortby('valid_time')
    
    # Remove singleton pressure_level dimension if present
    if 'pressure_level' in combined_ds.dims and combined_ds.sizes['pressure_level'] == 1:
        combined_ds = combined_ds.squeeze('pressure_level', drop=True)
    
    print(f"✓ Loaded {season} data: {years[0]}-{years[-1]}")
    print(f"  Time range: {pd.Timestamp(combined_ds['valid_time'].values[0]).strftime('%Y-%m-%d')} to {pd.Timestamp(combined_ds['valid_time'].values[-1]).strftime('%Y-%m-%d')}")
    print(f"  Total time steps: {len(combined_ds['valid_time'])}")
    
    return combined_ds


def load_seasonal_z500(year, season, extend=False, data_dir='../Data/'):
    """
    Load and combine ERA5 Z500 data for a given year and season.
    
    Parameters:
    -----------
    year : int or list of int
        The year(s) for the season (e.g., 2024 or [2023, 2024] for DJF)
        If a list is provided, data from all years will be concatenated in order
    season : str
        Season abbreviation. Options: 'DJF', 'MAM', 'JJA', 'SON'
    extend : bool
        If True, load one month before and one month after the season (default: False)
    data_dir : str
        Path to the data directory (default: '../Data/')
    
    Returns:
    --------
    xr.Dataset
        Combined xarray dataset containing Z500 data for all months in the season,
        sorted by time
    
    Examples:
    ---------
    >>> ds_djf = load_seasonal_z500(2024, 'DJF')
    >>> ds_djf_extended = load_seasonal_z500(2024, 'DJF', extend=True)
    >>> ds_multi_year = load_seasonal_z500([2023, 2024], 'DJF')  # Multiple years
    >>> ds_jja = load_seasonal_z500(2024, 'JJA')
    """
    import os
    
    # Handle multiple years - recursively load each year and concatenate
    if isinstance(year, (list, tuple)):
        print(f"Loading {season} data for {len(year)} years ({year[0]}-{year[-1]})...")
        
        all_datasets = []
        for i, yr in enumerate(year, 1):
            # Suppress output for individual years, just show progress
            import sys
            from io import StringIO
            old_stdout = sys.stdout
            sys.stdout = StringIO()  # Suppress prints
            
            try:
                ds_year = load_seasonal_z500(yr, season, extend=extend, data_dir=data_dir)
                all_datasets.append(ds_year)
            finally:
                sys.stdout = old_stdout
            
            # Show progress every 10 years
            if i % 10 == 0 or i == len(year):
                print(f"  Progress: {i}/{len(year)} years loaded")
        
        # Concatenate all years along time dimension
        print(f"Combining {len(all_datasets)} year(s) of data...")
        combined_ds = xr.concat(all_datasets, dim='valid_time')
        combined_ds = combined_ds.sortby('valid_time')
        
        print(f"✓ Loaded {season} data: {year[0]}-{year[-1]}")
        print(f"  Time range: {pd.Timestamp(combined_ds['valid_time'].values[0]).strftime('%Y-%m-%d')} to {pd.Timestamp(combined_ds['valid_time'].values[-1]).strftime('%Y-%m-%d')}")
        print(f"  Total time steps: {len(combined_ds['valid_time'])}")
        
        return combined_ds
    
    # Single year processing (original logic)
    # Define season-to-months mapping (core season months)
    season_months = {
        'DJF': [('Dec', year-1), ('January', year), ('February', year)],  # Dec is from previous year
        'MAM': [('March', year), ('April', year), ('May', year)],
        'JJA': [('June', year), ('July', year), ('August', year)],
        'SON': [('September', year), ('October', year), ('November', year)]
    }
    
    # Define extended months (one month before and after each season)
    # Format: (month_name, month_number, year_offset)
    extended_months = {
        'DJF': {
            'before': ('November', 11, year-1),  # November before December
            'after': ('March', 3, year)          # March after February
        },
        'MAM': {
            'before': ('February', 2, year),     # February before March
            'after': ('June', 6, year)           # June after May
        },
        'JJA': {
            'before': ('May', 5, year),          # May before June
            'after': ('September', 9, year)      # September after August
        },
        'SON': {
            'before': ('August', 8, year),       # August before September
            'after': ('December', 12, year)      # December after November
        }
    }
    
    if season not in season_months:
        raise ValueError(f"Invalid season '{season}'. Must be one of: DJF, MAM, JJA, SON")
    
    # Get the months for the specified season
    months = season_months[season]
    
    # Load datasets for each month
    datasets = []
    
    # Load extended month BEFORE if extend=True
    if extend:
        ext_month_name, ext_month_num, ext_year = extended_months[season]['before']
        filename = f'ERA5_Z500_NH_{ext_year}_{ext_month_num:02d}_{ext_month_name}.nc'
        filepath = os.path.join(data_dir, filename)
        
        if os.path.exists(filepath):
            print(f"Loading (extended): {filename}")
            ds = xr.open_dataset(filepath)
            datasets.append(ds)
        else:
            print(f"Warning: Extended file not found: {filepath}")
    
    # Load core season months
    for month_name, month_year in months:
        # Construct filename
        # For December, the file name uses the format Dec{year-1}
        if month_name == 'Dec':
            filename = f'ERA5_Z500_NH_{season}_{year}_{month_name}{month_year}.nc'
        else:
            filename = f'ERA5_Z500_NH_{season}_{year}_{month_name}.nc'
        
        filepath = os.path.join(data_dir, filename)
        
        # Check if file exists
        if not os.path.exists(filepath):
            print(f"Warning: File not found: {filepath}")
            continue
        
        # Load the dataset
        print(f"Loading: {filename}")
        ds = xr.open_dataset(filepath)
        datasets.append(ds)
    
    # Load extended month AFTER if extend=True
    if extend:
        ext_month_name, ext_month_num, ext_year = extended_months[season]['after']
        filename = f'ERA5_Z500_NH_{ext_year}_{ext_month_num:02d}_{ext_month_name}.nc'
        filepath = os.path.join(data_dir, filename)
        
        if os.path.exists(filepath):
            #print(f"Loading (extended): {filename}")
            ds = xr.open_dataset(filepath)
            datasets.append(ds)
        else:
            print(f"Warning: Extended file not found: {filepath}")
    
    if not datasets:
        raise FileNotFoundError(f"No data files found for {season} {year} in {data_dir}")
    
    # Concatenate along time dimension and sort by time
    #print(f"\nCombining {len(datasets)} file(s)...")
    combined_ds = xr.concat(datasets, dim='valid_time')
    combined_ds = combined_ds.sortby('valid_time')
    
    # Remove singleton pressure_level dimension (size 1)
    if 'pressure_level' in combined_ds.dims and combined_ds.sizes['pressure_level'] == 1:
        combined_ds = combined_ds.squeeze('pressure_level', drop=True)
    
    extend_str = " (extended)" if extend else ""
    # print(f"✓ Successfully loaded {season} {year} data{extend_str}")
    # print(f"  Time range: {pd.Timestamp(combined_ds['valid_time'].values[0])} to {pd.Timestamp(combined_ds['valid_time'].values[-1])}")
    # print(f"  Total time steps: {len(combined_ds['valid_time'])}")
    
    return combined_ds




def resample_to_daily(ds, time_coord='valid_time'):
    """
    Resample 6-hourly (or any sub-daily) data to daily averages.
    
    Parameters:
    -----------
    ds : xr.Dataset
        Input xarray dataset with sub-daily temporal resolution
    time_coord : str
        Name of the time coordinate (default: 'valid_time')
    
    Returns:
    --------
    xr.Dataset
        Dataset resampled to daily averages
    
    Examples:
    ---------
    >>> ds_6hourly = load_seasonal_z500(2023, 'DJF')
    >>> ds_daily = resample_to_daily(ds_6hourly)
    """
    
    # Debug: Check time resolution
    time_vals = pd.DatetimeIndex(ds[time_coord].values)
    if len(time_vals) > 1:
        time_diff = time_vals[1] - time_vals[0]
        print(f"  Input time resolution: {time_diff}")
    
    # For dask arrays, we need to load time coordinate first and rechunk properly
    # This ensures resample works correctly with lazy data
    original_steps = len(ds[time_coord])
    
    # Check if data is chunked with dask
    try:
        if hasattr(ds, 'chunks') and ds.chunks:
            print(f"  Dask-backed data detected, unifying chunks...")
            # First unify chunks if they're inconsistent (common with open_mfdataset)
            ds = ds.unify_chunks()
            print(f"  Rechunking for resampling...")
            # Rechunk along time to ensure proper resampling
            # Use smaller time chunks to avoid memory issues
            ds = ds.chunk({time_coord: 500})
    except (ValueError, AttributeError):
        # If chunks access fails, try to unify first
        print(f"  Inconsistent chunks detected, unifying...")
        ds = ds.unify_chunks()
        ds = ds.chunk({time_coord: 500})
    
    # Resample to daily frequency and compute immediately
    # The key is to use groupby with date instead of resample for dask compatibility
    print(f"  Resampling {original_steps} time steps to daily...")
    
    # Convert to daily using resample, then compute
    ds_daily = ds.resample({time_coord: 'D'}).mean()
    
    # Force computation if it's dask
    if hasattr(ds_daily, 'compute'):
        print(f"  Computing daily averages (dask)...")
        ds_daily = ds_daily.compute()
    
    daily_steps = len(ds_daily[time_coord])
    
    if daily_steps == original_steps or abs(daily_steps - original_steps) < 10:
        print(f"  ⚠ Warning: No resampling occurred! Data may already be daily.")
        print(f"  ✓ {original_steps} timesteps → {daily_steps} days (÷{original_steps/daily_steps:.1f})")
    else:
        print(f"  ✓ {original_steps} timesteps → {daily_steps} days (÷{original_steps/daily_steps:.1f})")
    
    return ds_daily

def standardize_coordinates(data):
    """
    Standardize coordinate conventions for global atmospheric data.
    
    This function applies three standard preprocessing steps:
    1. Ensures latitude increases northward (sorts if needed)
    2. Ensures longitude runs from 0° to 360° (converts negative values)
    3. Adds a cyclic longitude point to prevent discontinuity across the map boundary
    
    These steps ensure consistent data orientation and prevent interpolation artifacts
    at longitude boundaries for global analyses.
    
    Parameters
    ----------
    data : xarray.DataArray
        Input data with lat and lon coordinates
    
    Returns
    -------
    data_standardized : xarray.DataArray
        Data with standardized coordinates:
        - Latitude increasing northward
        - Longitude in [0, 360] range
        - Cyclic point added at end of longitude dimension
    
    Examples
    --------
    >>> data_std = standardize_coordinates(ds['z'])
    """
    
    # Make a copy to avoid modifying the original
    data_std = data.copy()
    
    # Identify coordinate names (handle different naming conventions)
    lat_names = ['lat', 'latitude', 'y']
    lon_names = ['lon', 'longitude', 'x']
    
    lat_coord = None
    lon_coord = None
    
    for name in lat_names:
        if name in data_std.coords:
            lat_coord = name
            break
    
    for name in lon_names:
        if name in data_std.coords:
            lon_coord = name
            break
    
    if lat_coord is None or lon_coord is None:
        raise ValueError(f"Could not identify lat/lon coordinates. Available coords: {list(data_std.coords.keys())}")
    
    # Step 1: Ensure latitude increases northward
    if data_std[lat_coord].values[0] > data_std[lat_coord].values[-1]:
        data_std = data_std.sortby(lat_coord)
    
    # Step 2: Ensure longitude runs from 0° to 360°
    lon_values = data_std[lon_coord].values
    if np.any(lon_values < 0):
        # Convert negative longitudes to [0, 360] range
        lon_values_new = np.where(lon_values < 0, lon_values + 360, lon_values)
        data_std[lon_coord] = lon_values_new
        
        # Sort by longitude to maintain order after conversion
        data_std = data_std.sortby(lon_coord)
    
    # Step 3: Add cyclic longitude point
    
    # Get the first longitude slice and assign it to position 360 (or 0 + 360)
    # This creates continuity across the map boundary
    first_lon_slice = data_std.isel({lon_coord: 0})
    
    # Create new longitude value (should be 360° if starting from 0°)
    lon_spacing = data_std[lon_coord].values[1] - data_std[lon_coord].values[0]
    new_lon_value = data_std[lon_coord].values[-1] + lon_spacing
    
    # Assign the new longitude coordinate to the slice
    first_lon_slice[lon_coord] = new_lon_value
    
    # Concatenate along longitude dimension
    data_std = xr.concat([data_std, first_lon_slice], dim=lon_coord)
    
    print(f"✓ Coordinates standardized: lat=[{data_std[lat_coord].values[0]:.1f}, {data_std[lat_coord].values[-1]:.1f}], lon=[{data_std[lon_coord].values[0]:.1f}, {data_std[lon_coord].values[-1]:.1f}], shape={data_std.shape}")
    
    return data_std


def rename_coords_to_standard(data):
    """
    Rename coordinates to standard names expected by ABS_PlaSim functions.
    
    Converts:
    - 'latitude' -> 'lat'
    - 'longitude' -> 'lon'
    - 'valid_time' -> 'time'
    
    Parameters:
    -----------
    data : xarray.DataArray or xarray.Dataset
        Input data with any lat/lon/time naming convention
    
    Returns:
    --------
    data_renamed : xarray.DataArray or xarray.Dataset
        Data with coordinates renamed to 'lat', 'lon', and 'time'
    """
    rename_dict = {}
    
    # Check for latitude variations
    lat_names = ['latitude', 'Latitude', 'LAT', 'y']
    for name in lat_names:
        if name in data.coords and name != 'lat':
            rename_dict[name] = 'lat'
            break
    
    # Check for longitude variations
    lon_names = ['longitude', 'Longitude', 'LON', 'x']
    for name in lon_names:
        if name in data.coords and name != 'lon':
            rename_dict[name] = 'lon'
            break
    
    # Check for time variations
    time_names = ['valid_time', 'Valid_time', 'VALID_TIME', 'datetime']
    for name in time_names:
        if name in data.coords and name != 'time':
            rename_dict[name] = 'time'
            break
    
    if rename_dict:
        print(f"✓ Renamed coordinates: {' → '.join([f'{k}→{v}' for k, v in rename_dict.items()])}")
        data = data.rename(rename_dict)
    
    return data



def process_era5_for_blocking(year, season, extend=False, data_dir='../Data/', 
                               resample=True, z_variable='z'):
    """
    Complete preprocessing pipeline for ERA5 Z500 data for ABS blocking analysis.
    
    This is a convenience function that chains together all preprocessing steps:
    1. Load seasonal Z500 data (load_seasonal_z500)
    2. Resample to daily averages (resample_to_daily) - optional
    3. Standardize coordinates (standardize_coordinates)
    4. Rename coordinates to ABS standard names (rename_coords_to_standard)
    
    Parameters:
    -----------
    year : int or list of int
        The year(s) for the season (e.g., 2024 or [2023, 2024] for DJF)
        If a list is provided, data from all years will be concatenated in order
    season : str
        Season abbreviation. Options: 'DJF', 'MAM', 'JJA', 'SON'
    extend : bool
        If True, load one month before and one month after the season (default: False)
    data_dir : str
        Path to the data directory (default: '../Data/')
    resample : bool
        If True, resample to daily averages (default: True)
        Set to False if data is already daily or you want to keep original temporal resolution
    z_variable : str
        Name of the Z500 variable in the dataset (default: 'z')
    
    Returns:
    --------
    xr.DataArray
        Fully preprocessed Z500 data ready for ABS blocking analysis with:
        - Coordinates renamed to 'time', 'lat', 'lon'
        - Daily temporal resolution (if resample=True)
        - Latitude increasing northward
        - Longitude in [0, 360] range with cyclic point
        - Shape: (time, lat, lon)
    
    Examples:
    ---------
    >>> # Single year, daily data
    >>> z500_ready = process_era5_for_blocking(2023, 'DJF')
    >>> 
    >>> # Multiple years with extended months
    >>> z500_ready = process_era5_for_blocking([2020, 2021, 2022], 'DJF', extend=True)
    >>> 
    >>> # Keep 6-hourly resolution
    >>> z500_6hourly = process_era5_for_blocking(2023, 'DJF', resample=False)
    >>> 
    >>> # Now ready to use with ABS blocking analysis
    >>> from Src.ABS_PlaSim import abs_blocking_frequency
    >>> blocking_freq = abs_blocking_frequency(z500_ready)
    """
    
    # Determine if processing single or multiple years
    is_multi_year = isinstance(year, (list, tuple))
    year_str = f"{year[0]}-{year[-1]}" if is_multi_year else str(year)
    
    print("=" * 70)
    print(f"ERA5 PREPROCESSING: {season} {year_str}")
    print("=" * 70)
    
    # Step 1: Load seasonal data
    print("[1/4] Loading seasonal Z500 data...")
    ds = load_seasonal_z500(year, season, extend=extend, data_dir=data_dir)
    
    # Step 2: Resample to daily (optional) - DO THIS BEFORE COORDINATE RENAMING
    if resample:
        print("[2/4] Resampling to daily averages...")
        ds = resample_to_daily(ds, time_coord='valid_time')
    else:
        print("[2/4] Keeping original temporal resolution")
    
    # Extract z500 variable after resampling
    z500_data = ds[z_variable]
    
    # Convert from m²/s² to geopotential meters (divide by g = 9.8 m/s²)
    print("   Converting Z500 from m²/s² to geopotential meters (÷9.8)...")
    z500_data = z500_data / 9.8
    
    # Step 3: Standardize coordinates (lat/lon orientation and cyclic point)
    print("[3/4] Standardizing coordinates...")
    z500_standardized = standardize_coordinates(z500_data)
    
    # Step 4: Rename coordinates to ABS standard names
    print("[4/4] Renaming coordinates...")
    z500_ready = rename_coords_to_standard(z500_standardized)
    
    # Final summary
    time_start = pd.Timestamp(z500_ready.time.values[0]).strftime('%Y-%m-%d')
    time_end = pd.Timestamp(z500_ready.time.values[-1]).strftime('%Y-%m-%d')
    print("=" * 70)
    print(f"✅ READY: {z500_ready.shape} | {time_start} to {time_end}")
    print("=" * 70)
    
    return z500_ready


def process_era5_for_blocking_fast(year, season, extend=False, data_dir='../Data/', 
                                    resample=True, z_variable='z'):
    """
    FAST VERSION: Process ERA5 data for ABS blocking analysis using parallel file loading.
    
    This optimized version uses xarray's open_mfdataset with dask for parallel loading,
    which is 5-10x faster than the sequential version for multiple years.
    
    Same parameters and usage as process_era5_for_blocking().
    """
    
    # Determine if processing single or multiple years
    is_multi_year = isinstance(year, (list, tuple))
    year_str = f"{year[0]}-{year[-1]}" if is_multi_year else str(year)
    
    print("=" * 70)
    print(f"ERA5 PREPROCESSING (FAST): {season} {year_str}")
    print("=" * 70)
    
    # Step 1: Load seasonal data using fast parallel loader
    # Resampling happens DURING load for maximum efficiency
    print("[1/4] Loading seasonal Z500 data (parallel mode)...")
    ds = load_seasonal_z500_fast(year, season, extend=extend, data_dir=data_dir, resample_daily=resample)
    
    # Step 2: Check what we loaded
    if resample:
        print("[2/4] Data resampled to daily during parallel load ✓")
    else:
        print("[2/4] Keeping original temporal resolution")
    
    # Extract z500 variable
    z500_data = ds[z_variable]
    
    # Convert from m²/s² to geopotential meters (divide by g = 9.8 m/s²)
    print("   Converting Z500 from m²/s² to geopotential meters (÷9.8)...")
    z500_data = z500_data / 9.8
    
    # Step 3: Standardize coordinates (lat/lon orientation and cyclic point)
    print("[3/4] Standardizing coordinates...")
    z500_standardized = standardize_coordinates(z500_data)
    
    # Step 4: Rename coordinates to ABS standard names
    print("[4/4] Renaming coordinates...")
    z500_ready = rename_coords_to_standard(z500_standardized)
    
    # Final summary
    time_start = pd.Timestamp(z500_ready.time.values[0]).strftime('%Y-%m-%d')
    time_end = pd.Timestamp(z500_ready.time.values[-1]).strftime('%Y-%m-%d')
    print("=" * 70)
    print(f"✅ READY: {z500_ready.shape} | {time_start} to {time_end}")
    print("=" * 70)
    
    return z500_ready
