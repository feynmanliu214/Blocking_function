"""
PlaSim Z500 Data Loader

Load geopotential height at 500 hPa (Z500) from PlaSim simulation output files.
Supports loading data across multiple years with automatic file mapping,
daily averaging of 6-hourly data, and Northern Hemisphere subsetting.

Key Functions
-------------
load_z500_northern_hemisphere : Load Z500 data for a date range
standardize_coordinates : Standardize lat/lon conventions (add cyclic point)
get_available_years : List available simulation years
get_data_time_range : Get overall available date range

Author: Auto-generated
Date: 2026-01-10
"""

import os
from pathlib import Path
from typing import Union, Optional, List
import warnings

import numpy as np
import xarray as xr
import cftime


# Default data directory
DEFAULT_DATA_DIR = Path("/glade/u/home/zhil/project/AI-RES/Blocking/data/PlaSim/sim52/plev_data")

# File naming pattern: {year}_gaussian.nc
FILE_PATTERN = "{year}_gaussian.nc"

# Variable and coordinate names in PlaSim files
ZG_VAR = "zg"  # geopotential height variable
PLEV_COORD = "plev"  # pressure level coordinate
TIME_COORD = "time"
LAT_COORD = "lat"
LON_COORD = "lon"

# Z500 pressure level (500 hPa = 50000 Pa)
# Note: plev metadata says "hPa" but values are actually in Pa
Z500_LEVEL = 50000.0


def parse_date(date_str: str) -> tuple:
    """
    Parse a date string "YYYY-MM-DD" into (year, month, day) integers.
    
    Parameters
    ----------
    date_str : str
        Date in format "YYYY-MM-DD"
    
    Returns
    -------
    tuple
        (year, month, day) as integers
    """
    parts = date_str.strip().split("-")
    if len(parts) != 3:
        raise ValueError(f"Invalid date format: '{date_str}'. Expected 'YYYY-MM-DD'.")
    return int(parts[0]), int(parts[1]), int(parts[2])


def get_years_in_range(start_date: str, end_date: str) -> List[int]:
    """
    Get list of years that need to be loaded for a given date range.
    
    Parameters
    ----------
    start_date : str
        Start date in "YYYY-MM-DD" format
    end_date : str
        End date in "YYYY-MM-DD" format (inclusive)
    
    Returns
    -------
    List[int]
        List of years to load
    """
    start_year, _, _ = parse_date(start_date)
    end_year, _, _ = parse_date(end_date)
    
    if end_year < start_year:
        raise ValueError(f"End date ({end_date}) is before start date ({start_date})")
    
    return list(range(start_year, end_year + 1))


def get_file_path(year: int, data_dir: Union[str, Path] = DEFAULT_DATA_DIR) -> Path:
    """
    Get the file path for a given simulation year.
    
    Parameters
    ----------
    year : int
        Simulation year index
    data_dir : Path or str
        Directory containing the NetCDF files
    
    Returns
    -------
    Path
        Path to the NetCDF file
    """
    data_dir = Path(data_dir)
    filename = FILE_PATTERN.format(year=year)
    return data_dir / filename


def check_file_availability(years: List[int], data_dir: Union[str, Path] = DEFAULT_DATA_DIR) -> dict:
    """
    Check which files exist for the requested years.
    
    Parameters
    ----------
    years : List[int]
        Years to check
    data_dir : Path or str
        Directory containing the NetCDF files
    
    Returns
    -------
    dict
        Dictionary with 'available' and 'missing' lists of years
    """
    data_dir = Path(data_dir)
    available = []
    missing = []
    
    for year in years:
        file_path = get_file_path(year, data_dir)
        if file_path.exists():
            available.append(year)
        else:
            missing.append(year)
    
    return {"available": available, "missing": missing}


def create_cftime_date(year: int, month: int, day: int, 
                       hour: int = 0) -> cftime.DatetimeProlepticGregorian:
    """
    Create a cftime datetime object compatible with PlaSim time coordinates.
    
    Parameters
    ----------
    year, month, day : int
        Date components
    hour : int, optional
        Hour of day (default 0)
    
    Returns
    -------
    cftime.DatetimeProlepticGregorian
        cftime datetime object
    """
    return cftime.DatetimeProlepticGregorian(year, month, day, hour, 0, 0, 0, 
                                              has_year_zero=True)


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


def load_z500_northern_hemisphere(
    start_date: str,
    end_date: str,
    data_dir: Union[str, Path] = DEFAULT_DATA_DIR,
    daily_mean: bool = True,
    min_lat: float = 0.0,
    standardize_coords: bool = True,
) -> xr.DataArray:
    """
    Load Z500 (geopotential height at 500 hPa) for the Northern Hemisphere.
    
    This function:
    - Automatically determines which yearly files to load
    - Extracts the 500 hPa level from geopotential height
    - Subsets to Northern Hemisphere (lat >= min_lat)
    - Computes daily means if data is sub-daily (6-hourly)
    - Standardizes coordinates (by default): ascending lat, 0-360° lon with cyclic point
    - Returns data for the exact date range requested
    
    Parameters
    ----------
    start_date : str
        Start date in "YYYY-MM-DD" format
    end_date : str
        End date in "YYYY-MM-DD" format (inclusive)
    data_dir : Path or str, optional
        Directory containing the PlaSim NetCDF files.
        Default: /glade/u/home/zhil/project/AI-RES/Blocking/data/PlaSim/sim52/plev_data/
    daily_mean : bool, optional
        If True (default), compute daily mean from 6-hourly data.
        If False, return raw 6-hourly data.
    min_lat : float, optional
        Minimum latitude for Northern Hemisphere selection (default 0.0).
        Data with lat >= min_lat will be returned.
    standardize_coords : bool, optional
        If True (default), standardize coordinates:
        - Sort latitude to increase northward (S to N)
        - Convert longitude to 0-360° range
        - Add cyclic point at 360° to prevent plotting artifacts
        If False, return data with original coordinate ordering.
    
    Returns
    -------
    xr.DataArray
        Z500 data with dimensions (time, lat, lon).
        Time coordinate contains daily timestamps (if daily_mean=True).
        Latitude is subset to Northern Hemisphere.
        Coordinates are standardized by default (opt-out with standardize_coords=False).
    
    Raises
    ------
    FileNotFoundError
        If required data files are missing
    ValueError
        If date range is invalid or Z500 level not found
    KeyError
        If expected variable or coordinate not found in files
    
    Examples
    --------
    >>> # Load with default settings (standardized coordinates)
    >>> z500 = load_z500_northern_hemisphere("0010-06-01", "0010-08-31")
    >>> print(z500.dims)
    ('time', 'lat', 'lon')
    >>> print(z500.shape)
    (92, 32, 129)  # 92 days, 32 NH latitudes, 129 longitudes (with cyclic point)
    
    >>> # Load without coordinate standardization (raw data)
    >>> z500_raw = load_z500_northern_hemisphere("0010-06-01", "0010-08-31", 
    ...                                           standardize_coords=False)
    >>> print(z500_raw.shape)
    (92, 32, 128)  # Original 128 longitudes, no cyclic point
    
    >>> # Load multiple years
    >>> z500_multi = load_z500_northern_hemisphere("0050-01-01", "0052-12-31")
    """
    data_dir = Path(data_dir)
    
    # Parse dates and get required years
    start_year, start_month, start_day = parse_date(start_date)
    end_year, end_month, end_day = parse_date(end_date)
    years_needed = get_years_in_range(start_date, end_date)
    
    # Check file availability
    file_check = check_file_availability(years_needed, data_dir)
    if file_check["missing"]:
        missing_files = [str(get_file_path(y, data_dir)) for y in file_check["missing"]]
        raise FileNotFoundError(
            f"Missing data files for years {file_check['missing']}:\n" +
            "\n".join(missing_files)
        )
    
    # Create cftime bounds for time selection
    time_start = create_cftime_date(start_year, start_month, start_day, hour=0)
    # End date is inclusive, so we go to the last timestep of that day (18:00 for 6-hourly)
    time_end = create_cftime_date(end_year, end_month, end_day, hour=23)
    
    # Load data from each year file
    datasets = []
    
    for year in years_needed:
        file_path = get_file_path(year, data_dir)
        
        # Open dataset with cftime decoding
        try:
            ds = xr.open_dataset(
                file_path,
                decode_times=xr.coders.CFDatetimeCoder(use_cftime=True)
            )
        except Exception as e:
            # Fallback for older xarray versions
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ds = xr.open_dataset(file_path, use_cftime=True)
        
        # Check for required variable
        if ZG_VAR not in ds.data_vars:
            ds.close()
            raise KeyError(
                f"Variable '{ZG_VAR}' not found in {file_path}. "
                f"Available variables: {list(ds.data_vars)}"
            )
        
        # Check for required coordinates
        for coord in [PLEV_COORD, LAT_COORD, LON_COORD, TIME_COORD]:
            if coord not in ds.coords:
                ds.close()
                raise KeyError(
                    f"Coordinate '{coord}' not found in {file_path}. "
                    f"Available coordinates: {list(ds.coords)}"
                )
        
        # Check that Z500 level exists
        if Z500_LEVEL not in ds[PLEV_COORD].values:
            ds.close()
            available_levels = ds[PLEV_COORD].values
            raise ValueError(
                f"500 hPa level ({Z500_LEVEL}) not found in {file_path}. "
                f"Available levels: {available_levels}"
            )
        
        # Extract Z500 (select 500 hPa level)
        z500 = ds[ZG_VAR].sel({PLEV_COORD: Z500_LEVEL})
        
        # Subset to Northern Hemisphere
        # Handle both ascending and descending latitude orderings
        lat_values = ds[LAT_COORD].values
        if lat_values[0] > lat_values[-1]:
            # Descending order (N to S) - typical for PlaSim
            nh_mask = lat_values >= min_lat
            z500_nh = z500.isel({LAT_COORD: nh_mask})
        else:
            # Ascending order (S to N)
            z500_nh = z500.sel({LAT_COORD: slice(min_lat, None)})
        
        datasets.append(z500_nh)
        ds.close()
    
    # Concatenate all years
    if len(datasets) == 1:
        z500_combined = datasets[0]
    else:
        z500_combined = xr.concat(datasets, dim=TIME_COORD)
    
    # Select requested time range
    # Use boolean indexing for cftime compatibility
    time_values = z500_combined[TIME_COORD].values
    time_mask = (time_values >= time_start) & (time_values <= time_end)
    z500_subset = z500_combined.isel({TIME_COORD: time_mask})
    
    # Check if we got any data
    if len(z500_subset[TIME_COORD]) == 0:
        raise ValueError(
            f"No data found for date range {start_date} to {end_date}. "
            f"Available time range in loaded files: "
            f"{time_values[0]} to {time_values[-1]}"
        )
    
    # Compute daily mean if requested and data is sub-daily
    if daily_mean:
        # Check if data is sub-daily by looking at time differences
        if len(z500_subset[TIME_COORD]) > 1:
            t0 = z500_subset[TIME_COORD].values[0]
            t1 = z500_subset[TIME_COORD].values[1]
            # Calculate hours between first two timesteps
            # cftime objects support subtraction
            dt_hours = (t1 - t0).total_seconds() / 3600
            
            if dt_hours < 24:
                # Sub-daily data - compute daily mean
                z500_daily = z500_subset.resample({TIME_COORD: "1D"}).mean()
                z500_subset = z500_daily
    
    # Standardize coordinates if requested (default)
    if standardize_coords:
        z500_subset = standardize_coordinates(z500_subset)
    
    # Add metadata
    z500_subset.attrs["long_name"] = "Geopotential Height at 500 hPa"
    z500_subset.attrs["units"] = "gpm"
    z500_subset.attrs["source"] = "PlaSim simulation"
    z500_subset.attrs["hemisphere"] = "Northern"
    z500_subset.attrs["date_range"] = f"{start_date} to {end_date}"
    z500_subset.attrs["coordinates_standardized"] = standardize_coords
    
    # Load data into memory and return
    return z500_subset.compute()


def get_available_years(data_dir: Union[str, Path] = DEFAULT_DATA_DIR) -> List[int]:
    """
    Get list of available simulation years based on existing files.
    
    Parameters
    ----------
    data_dir : Path or str
        Directory containing the NetCDF files
    
    Returns
    -------
    List[int]
        Sorted list of available years
    """
    data_dir = Path(data_dir)
    years = []
    
    for f in data_dir.glob("*_gaussian.nc"):
        try:
            # Extract year from filename
            year_str = f.stem.split("_")[0]
            years.append(int(year_str))
        except (ValueError, IndexError):
            continue
    
    return sorted(years)


def get_data_time_range(data_dir: Union[str, Path] = DEFAULT_DATA_DIR) -> tuple:
    """
    Get the overall time range available in the data files.
    
    Parameters
    ----------
    data_dir : Path or str
        Directory containing the NetCDF files
    
    Returns
    -------
    tuple
        (start_date_str, end_date_str) in "YYYY-MM-DD" format
    """
    years = get_available_years(data_dir)
    if not years:
        raise FileNotFoundError(f"No *_gaussian.nc files found in {data_dir}")
    
    min_year = min(years)
    max_year = max(years)
    
    # Format with leading zeros for consistency
    start_date = f"{min_year:04d}-01-01"
    end_date = f"{max_year:04d}-12-31"
    
    return start_date, end_date


# Main function for convenience
def get_z500(start_date: str, end_date: str, **kwargs) -> xr.DataArray:
    """
    Shorthand for load_z500_northern_hemisphere().
    
    See load_z500_northern_hemisphere() for full documentation.
    """
    return load_z500_northern_hemisphere(start_date, end_date, **kwargs)



