import numpy as np
import xarray as xr
from pathlib import Path
from typing import List, Union, Optional
from scipy import ndimage
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import matplotlib.pyplot as plt



def z500_for_season(file_path, season):
    """
    Load a PlaSim NetCDF (same format as 1000_gaussian.nc), and return z500
    for the chosen season ('DJF' or 'JJA') as an xarray.DataArray with dims
    (time, lat, lon). Robust to ancient epochs and missing CF time units.

    Parameters
    ----------
    file_path : str
        Path to the NetCDF file (e.g., '../Data/1000_gaussian.nc').
    season : str
        'DJF' or 'JJA' (case-insensitive).

    Returns
    -------
    z500_season : xarray.DataArray
        Geopotential height at 500 hPa for the selected season.
    """
    season = season.upper()
    if season not in {"DJF", "JJA"}:
        raise ValueError("season must be 'DJF' or 'JJA'")

    # --- Open with cftime-safe decoding, fall back if needed ---
    try:
        from xr.coders import CFDatetimeCoder
        ds = xr.open_dataset(file_path, decode_times=CFDatetimeCoder(use_cftime=True))
    except Exception:
        try:
            ds = xr.open_dataset(file_path, decode_times=True, use_cftime=True)
        except Exception:
            ds = xr.open_dataset(file_path, decode_times=False)

    if "zg" not in ds:
        raise KeyError("Variable 'zg' not found in the dataset.")
    if "plev" not in ds:
        raise KeyError("Coordinate 'plev' not found in the dataset.")

    # --- Select 500 hPa (exact if present, else nearest) ---
    p = np.asarray(ds["plev"].values, dtype=float)
    # Check for both 500 (hPa) and 50000 (Pa)
    if np.isclose(p, 500.0).any():
        zg500 = ds["zg"].sel(plev=500.0)
    elif np.isclose(p, 50000.0).any():
        zg500 = ds["zg"].sel(plev=50000.0)
    else:
        # Try finding anything close to 500 hPa in either unit
        if p.max() > 2000: # Likely in Pa
            target = 50000.0
        else:
            target = 500.0
        nearest_p = p[np.argmin(np.abs(p - target))]
        zg500 = ds["zg"].sel(plev=nearest_p)
        zg500 = zg500.assign_attrs(
            **zg500.attrs,
            note=f"500 hPa/50000 Pa not found; used nearest level {nearest_p}"
        )

    # --- Build a boolean mask for requested season ---
    def month_mask_from_time(time_coord):
        # Preferred: use .dt.month if time is decoded to datetimes (incl. cftime)
        try:
            months = xr.DataArray(time_coord.dt.month.values, dims=("time",))
            return months
        except Exception:
            pass
        # Fallback: numeric time in days within a single 365-day year
        t = np.asarray(time_coord.values, dtype=float)
        if t.ndim != 1:
            raise ValueError("Unexpected time coordinate shape.")
        # Map day-of-year (0..364.999) to Gregorian month (no-leap assumption)
        # Month lengths (no leap): Jan31, Feb28, Mar31, Apr30, May31, Jun30,
        #                           Jul31, Aug31, Sep30, Oct31, Nov30, Dec31
        month_lengths = np.array([31,28,31,30,31,30,31,31,30,31,30,31])
        month_edges = np.concatenate([[0], np.cumsum(month_lengths)])  # 0..365
        # Convert fractional days to day index [0..364]
        doy = np.floor(np.mod(t, 365.0)).astype(int)
        # Find month m where month_edges[m] <= doy < month_edges[m+1]
        months = np.searchsorted(month_edges[1:], doy, side="right") + 1
        return xr.DataArray(months, dims=("time",))

    months = month_mask_from_time(zg500["time"])
    if season == "DJF":
        season_mask = (months == 12) | (months == 1) | (months == 2)
    else:  # JJA
        season_mask = (months == 6) | (months == 7) | (months == 8)

    # --- Subset and return ---
    z500_season = zg500.sel(time=zg500["time"][season_mask])
    # Ensure nice name
    z500_season.name = "zg500"
    return z500_season

def z500_seasonal_to_daily(file_path, season, extend=False):
    """
    Load PlaSim NetCDF, extract z500 for a given season, and convert 
    from 6-hourly to daily averages.
    
    Parameters
    ----------
    file_path : str
        Path to the NetCDF file (e.g., '../Data/1000_gaussian.nc').
    season : str
        'DJF' or 'JJA' (case-insensitive).
    extend : bool, optional
        If True, extend the season by one month before and after.
        DJF becomes NDJFM (Nov-Dec-Jan-Feb-Mar)
        JJA becomes MJJAS (May-Jun-Jul-Aug-Sep)
        Default: False
    
    Returns
    -------
    z500_daily : xarray.DataArray
        Daily-averaged geopotential height at 500 hPa for the selected season.
        Shape: (n_days, lat, lon)
    """
    season = season.upper()
    if season not in {"DJF", "JJA"}:
        raise ValueError("season must be 'DJF' or 'JJA'")
    
    # --- Open with cftime-safe decoding, fall back if needed ---
    try:
        from xarray.coders import CFDatetimeCoder
        ds = xr.open_dataset(file_path, decode_times=CFDatetimeCoder(use_cftime=True))
    except Exception:
        try:
            ds = xr.open_dataset(file_path, decode_times=True, use_cftime=True)
        except Exception:
            ds = xr.open_dataset(file_path, decode_times=False)
    
    if "zg" not in ds:
        raise KeyError("Variable 'zg' not found in the dataset.")
    if "plev" not in ds:
        raise KeyError("Coordinate 'plev' not found in the dataset.")
    
    # --- Select 500 hPa (exact if present, else nearest) ---
    p = np.asarray(ds["plev"].values, dtype=float)
    if np.isclose(p, 500.0).any():
        zg500 = ds["zg"].sel(plev=500.0)
    else:
        nearest_p = p[np.argmin(np.abs(p - 500.0))]
        zg500 = ds["zg"].sel(plev=nearest_p)
        zg500 = zg500.assign_attrs(
            **zg500.attrs,
            note=f"500 hPa not present; used nearest level {nearest_p} hPa"
        )
    
    # --- Build a boolean mask for requested season ---
    def month_mask_from_time(time_coord):
        # Preferred: use .dt.month if time is decoded to datetimes (incl. cftime)
        try:
            months = xr.DataArray(time_coord.dt.month.values, dims=("time",))
            return months
        except Exception:
            pass
        # Fallback: numeric time in days within a single 365-day year
        t = np.asarray(time_coord.values, dtype=float)
        if t.ndim != 1:
            raise ValueError("Unexpected time coordinate shape.")
        # Map day-of-year (0..364.999) to Gregorian month (no-leap assumption)
        month_lengths = np.array([31,28,31,30,31,30,31,31,30,31,30,31])
        month_edges = np.concatenate([[0], np.cumsum(month_lengths)])  # 0..365
        # Convert fractional days to day index [0..364]
        doy = np.floor(np.mod(t, 365.0)).astype(int)
        # Find month m where month_edges[m] <= doy < month_edges[m+1]
        months = np.searchsorted(month_edges[1:], doy, side="right") + 1
        return xr.DataArray(months, dims=("time",))
    
    months = month_mask_from_time(zg500["time"])
    if season == "DJF":
        if extend:
            # NDJFM: November(11), December(12), January(1), February(2), March(3)
            season_mask = (months == 11) | (months == 12) | (months == 1) | (months == 2) | (months == 3)
        else:
            # DJF: December(12), January(1), February(2)
            season_mask = (months == 12) | (months == 1) | (months == 2)
    else:  # JJA
        if extend:
            # MJJAS: May(5), June(6), July(7), August(8), September(9)
            season_mask = (months == 5) | (months == 6) | (months == 7) | (months == 8) | (months == 9)
        else:
            # JJA: June(6), July(7), August(8)
            season_mask = (months == 6) | (months == 7) | (months == 8)
    
    # --- Subset to season ---
    z500_season = zg500.sel(time=zg500["time"][season_mask])
    
    # --- Convert from 6-hourly to daily averages ---
    # The data has 4 timesteps per day (every 6 hours)
    # Group consecutive timesteps into days and average
    n_times = len(z500_season.time)
    n_days = n_times // 4  # Integer division to get complete days
    
    # Trim to complete days if necessary
    n_times_to_use = n_days * 4
    z500_trimmed = z500_season.isel(time=slice(0, n_times_to_use))
    
    # Reshape to (n_days, 4, lat, lon) and average over the 4 timesteps
    # First, get the data values
    data = z500_trimmed.values
    lat = z500_trimmed.lat.values
    lon = z500_trimmed.lon.values
    
    # Reshape: (n_days * 4, lat, lon) -> (n_days, 4, lat, lon)
    data_reshaped = data.reshape(n_days, 4, len(lat), len(lon))
    
    # Average over the second axis (the 4 timesteps per day)
    data_daily = data_reshaped.mean(axis=1)
    
    # Create new time coordinate (one value per day)
    # Use the first timestep of each day as the day's timestamp
    time_daily = z500_trimmed.time.values[::4]
    
    # Determine extended season name
    if extend:
        extended_season = "NDJFM" if season == "DJF" else "MJJAS"
        season_desc = f'{extended_season} (extended {season})'
    else:
        extended_season = season
        season_desc = season
    
    # Create the daily DataArray
    z500_daily = xr.DataArray(
        data_daily,
        coords={
            'time': time_daily,
            'lat': lat,
            'lon': lon
        },
        dims=['time', 'lat', 'lon'],
        name='zg500_daily',
        attrs={
            'description': f'Daily-averaged z500 for {season_desc} season',
            'units': 'm',
            'long_name': 'Geopotential Height at 500 hPa',
            'season': extended_season,
            'season_extended': extend,
            'temporal_resolution': 'daily (averaged from 6-hourly data)'
        }
    )
    
    return z500_daily

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
    
    Notes
    -----
    - The cyclic point is a duplicate of the first longitude value placed at the end
    - This does NOT subset or restrict latitudes - it preserves the full global range
    - For latitude subsetting, use appropriate selection after calling this function
    
    Examples
    --------
    >>> data_raw = xr.DataArray(...)  # Some global data
    >>> data_std = standardize_coordinates(data_raw)
    >>> print(data_std.lon[-1] - data_std.lon[-2])  # Should be ~360/n_lon
    """
    
    # Step 1: Ensure latitude increases northward
    if len(data.lat) > 1 and data.lat[0] > data.lat[-1]:
        data = data.sortby('lat')
    
    # Step 2: Ensure longitude runs from 0 to 360
    lon_vals = data.lon.values
    if np.any(lon_vals < 0):
        # Convert negative longitudes to 0-360 range
        lon_vals_adjusted = np.where(lon_vals < 0, lon_vals + 360, lon_vals)
        data = data.assign_coords(lon=lon_vals_adjusted)
        # Sort by longitude to ensure monotonic increase
        data = data.sortby('lon')
    
    # Step 3: Add cyclic longitude point
    # Get the data at longitude index 0 (which should wrap to 360°)
    cyclic_data = data.isel(lon=0)
    
    # Create new longitude value (360° or next value after the last)
    last_lon = float(data.lon[-1])
    first_lon = float(data.lon[0])
    lon_spacing = float(data.lon[1]) - first_lon
    new_lon_val = last_lon + lon_spacing
    
    # Assign the new longitude coordinate to the cyclic data
    cyclic_data = cyclic_data.assign_coords(lon=new_lon_val)
    
    # Concatenate along longitude dimension
    data_standardized = xr.concat([data, cyclic_data], dim='lon')
    
    # Update metadata
    data_standardized.attrs = data.attrs.copy()
    data_standardized.attrs['lon_cyclic'] = 'Added cyclic point at end'
    data_standardized.attrs['coordinate_convention'] = 'Latitude northward, Longitude 0-360'
    
    return data_standardized


def _process_single_file(file_num, data_dir, file_pattern, season, northern_hemisphere_only, extend):
    """Helper function to process a single file (used for parallel processing)."""
    file_name = file_pattern.format(num=file_num)
    file_path = data_dir / file_name
    
    try:
        # Process this file
        z500_daily = z500_seasonal_to_daily(str(file_path), season, extend=extend)
        
        # Filter to Northern Hemisphere if requested
        if northern_hemisphere_only:
            z500_daily = z500_daily.sel(lat=(z500_daily.lat >= 0))
        
        return (file_num, z500_daily, None)
    except FileNotFoundError:
        return (file_num, None, f"File not found: {file_name}")
    except Exception as e:
        return (file_num, None, f"Error processing {file_name}: {str(e)[:50]}...")


def process_multiple_files(file_numbers: Union[List[int], range], 
                           season: str,
                           data_dir: str = "../data",
                           file_pattern: str = "{num}_gaussian.nc",
                           combine: bool = True,
                           northern_hemisphere_only: bool = True,
                           extend: bool = False,
                           max_workers: int = None):
    """
    Process multiple PlaSim output files in parallel and optionally combine them.
    
    This function is designed for processing many years of PlaSim output
    where each file represents one year (e.g., 1000_gaussian.nc, 1001_gaussian.nc, etc.).
    Uses parallel processing to significantly speed up processing of many files.
    
    Parameters
    ----------
    file_numbers : list of int or range
        File numbers to process (e.g., [1000, 1001, 1002] or range(1000, 1100))
    season : str
        'DJF' or 'JJA' (case-insensitive)
    data_dir : str, optional
        Directory containing the data files (default: "../Data")
    file_pattern : str, optional
        Pattern for file names with {num} placeholder (default: "{num}_gaussian.nc")
    combine : bool, optional
        If True, concatenate all files into a single array. If False, return list of arrays.
    northern_hemisphere_only : bool, optional
        If True, filter data to Northern Hemisphere only (lat >= 0). Default: True
    extend : bool, optional
        If True, extend the season by one month before and after. Default: False
        DJF becomes NDJFM (Nov-Dec-Jan-Feb-Mar)
        JJA becomes MJJAS (May-Jun-Jul-Aug-Sep)
    max_workers : int, optional
        Maximum number of parallel workers. Default: None (uses CPU count)
    
    Returns
    -------
    z500_combined : xarray.DataArray or list of xarray.DataArray
        If combine=True: Single DataArray with all data concatenated along time dimension
        If combine=False: List of DataArrays, one per file
        Data will be filtered to Northern Hemisphere if northern_hemisphere_only=True
    
    Examples
    --------
    # Process 10 years of data (Northern Hemisphere only)
    >>> z500_djf = process_multiple_files(range(1000, 1010), season='DJF')
    
    # Process extended season (NDJFM instead of DJF)
    >>> z500_ndjfm = process_multiple_files(range(1000, 1010), season='DJF', extend=True)
    
    # Process specific years (include both hemispheres)
    >>> z500_jja = process_multiple_files([1000, 1005, 1010], season='JJA', 
    ...                                    northern_hemisphere_only=False)
    
    # Get separate arrays (don't combine)
    >>> z500_list = process_multiple_files(range(1000, 1100), 'DJF', combine=False)
    
    # Use more workers for faster processing
    >>> z500_djf = process_multiple_files(range(1000, 1100), 'DJF', max_workers=16)
    """
    data_dir = Path(data_dir)
    file_numbers_list = list(file_numbers)
    z500_results = {}  # Store results with file_num as key to maintain order
    failed_files = []
    
    hemisphere_str = " (Northern Hemisphere only)" if northern_hemisphere_only else ""
    # print(f"Processing {len(file_numbers_list)} files for {season} season{hemisphere_str}...")
    # print(f"Using parallel processing with max_workers={max_workers or 'auto'}")
    # print("=" * 60)
    
    # Process files in parallel with progress bar
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_filenum = {
            executor.submit(_process_single_file, file_num, data_dir, file_pattern, 
                          season, northern_hemisphere_only, extend): file_num
            for file_num in file_numbers_list
        }
        
        # Process completed tasks with progress bar
        with tqdm(total=len(file_numbers_list), desc="Processing files") as pbar:
            for future in as_completed(future_to_filenum):
                file_num, z500_data, error = future.result()
                
                if error is None:
                    z500_results[file_num] = z500_data
                else:
                    failed_files.append(error)
                
                pbar.update(1)
    
    #print("=" * 60)
    #print(f"Successfully processed: {len(z500_results)}/{len(file_numbers_list)} files")
    
    if failed_files:
        print(f"Failed files: {failed_files[:5]}" + 
              (f" ... and {len(failed_files)-5} more" if len(failed_files) > 5 else ""))
    
    if len(z500_results) == 0:
        raise ValueError("No files were successfully processed!")
    
    # Sort results by file number to maintain order
    z500_arrays = [z500_results[file_num] for file_num in sorted(z500_results.keys())]
    
    # Combine or return list
    if combine:
        print("\nCombining data along time dimension...")
        # Concatenate along time dimension
        z500_combined = xr.concat(z500_arrays, dim='time')
        
        # Determine season name for metadata
        if extend:
            extended_season = "NDJFM" if season.upper() == "DJF" else "MJJAS"
            season_display = f"{extended_season} (extended {season})"
        else:
            extended_season = season
            season_display = season
        
        z500_combined.name = f"z500_daily_{extended_season}_combined"
        
        # Add metadata
        hemisphere_note = " (Northern Hemisphere only)" if northern_hemisphere_only else ""
        z500_combined.attrs['description'] = (
            f"Combined daily z500 for {season_display} season from {len(z500_arrays)} files{hemisphere_note}"
        )
        z500_combined.attrs['source_files'] = f"{len(z500_arrays)} files"
        z500_combined.attrs['file_pattern'] = file_pattern
        z500_combined.attrs['hemisphere'] = 'Northern (lat >= 0)' if northern_hemisphere_only else 'Both'
        z500_combined.attrs['season'] = extended_season
        z500_combined.attrs['season_extended'] = extend
        
        # Apply coordinate standardization (lat orientation, lon 0-360, cyclic point)
        #print("Applying coordinate standardization (lat orientation, lon 0-360, cyclic point)...")
        z500_combined = standardize_coordinates(z500_combined)
        
        # print(f"✅ Combined shape: {z500_combined.shape}")
        # print(f"   Total days: {z500_combined.shape[0]}")
        # print(f"   Latitude points: {z500_combined.shape[1]}")
        # print(f"   Longitude points: {z500_combined.shape[2]} (includes cyclic point)")
        # if northern_hemisphere_only:
        #     print(f"   Latitude range: {float(z500_combined.lat.min()):.2f}° to {float(z500_combined.lat.max()):.2f}°N")
        # print(f"   Longitude range: {float(z500_combined.lon.min()):.2f}° to {float(z500_combined.lon.max()):.2f}°")
        # print(f"   Total years: {z500_combined.shape[0] / 90:.1f} (assuming 90 days/season)")
        
        return z500_combined
    else:
        # print(f"\n✅ Returning list of {len(z500_arrays)} separate arrays")
        # # Apply coordinate standardization to each array in the list
        # print("Applying coordinate standardization to each array...")
        z500_arrays = [standardize_coordinates(arr) for arr in z500_arrays]
        return z500_arrays


def get_climatology_from_multiple_files(file_numbers: Union[List[int], range],
                                       season: str,
                                       data_dir: str = "../Data",
                                       file_pattern: str = "{num}_gaussian.nc",
                                       northern_hemisphere_only: bool = True):
    """
    Calculate climatology (time mean) from multiple PlaSim files.
    
    Parameters
    ----------
    file_numbers : list of int or range
        File numbers to process
    season : str
        'DJF' or 'JJA'
    data_dir : str, optional
        Directory containing the data files
    file_pattern : str, optional
        Pattern for file names with {num} placeholder
    northern_hemisphere_only : bool, optional
        If True, filter data to Northern Hemisphere only (lat >= 0). Default: True
    
    Returns
    -------
    z500_climatology : xarray.DataArray
        Time-mean z500 field (lat, lon) averaged over all files
    z500_std : xarray.DataArray
        Standard deviation across time (lat, lon)
    
    Examples
    --------
    >>> clim, std = get_climatology_from_multiple_files(range(1000, 1100), 'DJF')
    >>> print(f"Climatology shape: {clim.shape}")  # (32, 128) for NH only
    """
    # Get all data combined
    z500_combined = process_multiple_files(
        file_numbers, season, data_dir, file_pattern, combine=True,
        northern_hemisphere_only=northern_hemisphere_only
    )
    
    # print("\nCalculating climatology...")
    z500_climatology = z500_combined.mean(dim='time', keep_attrs=True)
    z500_climatology.name = f"z500_climatology_{season}"
    z500_climatology.attrs['description'] = f"Climatological mean z500 for {season}"
    
    z500_std = z500_combined.std(dim='time', keep_attrs=True)
    z500_std.name = f"z500_std_{season}"
    z500_std.attrs['description'] = f"Standard deviation of z500 for {season}"
    
    # print(f"✅ Climatology shape: {z500_climatology.shape}")
    # print(f"   Mean z500: {float(z500_climatology.mean()):.2f} m")
    # print(f"   Std dev: {float(z500_std.mean()):.2f} m")
    
    return z500_climatology, z500_std

