"""
Event Animation Module

This module provides functions for creating animated GIFs of blocking events.
Includes both standard and fast (optimized for large datasets) animation methods.

Fast version features:
- Smart event indexing (O(T) instead of repeated scans)
- Minimal data extraction (loads only needed timesteps/latitudes)
- Grid downsampling for faster plotting
- imageio-based GIF writing
- Optional diagnostic panels (P_block time series, centroid tracking)
"""

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import xarray as xr
from matplotlib.animation import FuncAnimation, PillowWriter
from cartopy.util import add_cyclic_point
import os
import imageio
from io import BytesIO
from multiprocessing import cpu_count


# =============================================================================
# Helper Functions
# =============================================================================

def _ensure_xarray(data, coords=None):
    """
    Ensure data is an xarray DataArray. If it's a numpy array, convert it.
    
    Parameters
    ----------
    data : numpy.ndarray or xarray.DataArray
        Input data
    coords : dict, optional
        Coordinates for conversion if data is numpy array
        
    Returns
    -------
    xarray.DataArray
        Data as xarray DataArray
    """
    if isinstance(data, xr.DataArray):
        return data
    elif isinstance(data, np.ndarray):
        if coords is not None:
            return xr.DataArray(data, coords=coords)
        else:
            # Create default coordinates
            dims = ['time', 'lat', 'lon'] if data.ndim == 3 else ['lat', 'lon']
            return xr.DataArray(data, dims=dims)
    else:
        raise TypeError(f"Expected numpy array or xarray DataArray, got {type(data)}")


def build_event_start_times_index(event_mask, event_ids, n_workers=None):
    """
    Generator that yields (event_id, start_time) tuples as events are found.
    
    Parameters
    ----------
    event_mask : numpy.ndarray
        3D array (time, lat, lon) with event IDs
    event_ids : list
        List of event IDs to find
    n_workers : int, optional
        Not used (kept for API compatibility)
    
    Yields
    ------
    tuple
        (event_id, start_time_index)
    """
    if n_workers is None:
        n_workers = min(cpu_count(), 8)

    n_times = event_mask.shape[0]
    print(f"  Building event start times index ({len(event_ids)} events, {n_workers} workers)...")
    events_found = set()
    events_to_find = set(event_ids)

    for t in range(n_times):
        if not events_to_find:
            break
        unique_ids = np.unique(event_mask[t])
        for eid in unique_ids:
            if eid in events_to_find and eid not in events_found:
                events_found.add(eid)
                events_to_find.discard(eid)
                yield eid, t
        if t % 500 == 0:
            print(f"    Scanned {t}/{n_times} timesteps, found {len(events_found)}/{len(event_ids)} events")


def build_event_start_times(event_mask, event_ids):
    """
    Build a dictionary mapping event IDs to their start time indices.
    
    Parameters
    ----------
    event_mask : numpy.ndarray
        3D array (time, lat, lon) with event IDs
    event_ids : list
        List of event IDs to index
    
    Returns
    -------
    dict
        Mapping of event_id -> start_time_index
    """
    index = {}
    for eid, start in build_event_start_times_index(event_mask, event_ids):
        index[eid] = start
    print(f"  ✓ Index built: {len(index)} events indexed")
    return index


def get_event_time_range(ano_stats, event_id, event_start_times=None):
    """
    Get the time indices for a specific event.
    
    Parameters
    ----------
    ano_stats : dict
        Event statistics dictionary
    event_id : int
        Event ID to find
    event_start_times : dict, optional
        Pre-computed start times for faster lookup
    
    Returns
    -------
    numpy.ndarray
        Array of time indices where the event occurs
    """
    duration = len(ano_stats['event_areas'][event_id])
    if event_start_times and event_id in event_start_times:
        start = event_start_times[event_id]
        return np.arange(start, start + duration)

    # Fallback linear scan (should rarely happen)
    event_mask = ano_stats['event_mask']
    if isinstance(event_mask, xr.DataArray):
        event_mask = event_mask.values
    for t in range(event_mask.shape[0]):
        if event_id in np.unique(event_mask[t]):
            return np.arange(t, t + duration)
    return np.array([])


# =============================================================================
# Format Conversion
# =============================================================================

def convert_abs_events_to_ano_format(abs_event_info, z500_data):
    """
    Convert ABS event information to format compatible with animation functions.
    
    This function creates an event_mask and event_areas from ABS persistent_events
    to allow visualization of ABS events using the same animation function.
    
    Parameters
    ----------
    abs_event_info : dict
        Event info dictionary from abs_blocking_frequency containing:
        - 'persistent_events': List of event dicts
        - 'final_blocking_mask': Boolean mask (time, lat, lon)
        - 'z500_prepared': Prepared z500 data
    z500_data : xarray.DataArray
        Original z500 data (will be used to compute anomalies or as background)
    
    Returns
    -------
    ano_format_stats : dict
        Dictionary in format expected by animation functions:
        - 'event_mask': (time, lat, lon) array with unique event IDs
        - 'z500_anom': Anomaly field or z500 data
        - 'blocked_mask': Binary blocking mask
        - 'event_areas': Dict mapping event_id to list of areas
    """
    from ANO_PlaSim import compute_anomalies
    from ANO_PlaSim import calculate_grid_cell_areas
    
    persistent_events = abs_event_info['persistent_events']
    final_mask = abs_event_info['final_blocking_mask']
    z500_prepared = abs_event_info['z500_prepared']
    
    # Get dimensions
    # final_mask has dimensions of gradients (only central latitudes)
    # z500_prepared has full latitude range
    n_times_final, n_lats_final, n_lons_final = final_mask.shape
    n_times_prep, n_lats_prep, n_lons_prep = z500_prepared.shape
    
    # Use full latitude range from z500_prepared
    lat_full = z500_prepared.lat.values
    lon_full = z500_prepared.lon.values
    lat_central = final_mask.lat.values  # Central latitudes where events occur
    
    # Find indices in full lat array corresponding to central lats
    lat_central_indices = []
    for clat in lat_central:
        idx = np.argmin(np.abs(lat_full - clat))
        lat_central_indices.append(idx)
    
    # Create event_mask with full latitude range
    event_mask = np.zeros((n_times_prep, n_lats_prep, n_lons_prep), dtype=int)
    event_areas = {}
    
    # Calculate grid cell areas for area computation (using central lats)
    grid_areas_central = calculate_grid_cell_areas(lat_central, lon_full)
    
    # Assign event IDs to each persistent event
    for event_idx, event in enumerate(persistent_events, start=1):
        event_id = event_idx
        event_daily_areas = []
        
        for t_offset, mask_2d in enumerate(event['masks']):
            t = event['start_time'] + t_offset
            if t < n_times_prep:
                # mask_2d has shape (n_lats_central, n_lons)
                # Need to place it at correct indices in full latitude array
                for i, lat_idx in enumerate(lat_central_indices):
                    if i < mask_2d.shape[0]:  # Safety check
                        event_mask[t, lat_idx, :] = np.where(
                            mask_2d[i, :] > 0, event_id, event_mask[t, lat_idx, :]
                        )
                
                # Calculate area for this day (using central lat grid areas)
                area = np.sum(grid_areas_central[mask_2d])
                event_daily_areas.append(area)
        
        event_areas[event_id] = event_daily_areas
    
    # Compute anomalies from z500_prepared if possible
    # Otherwise, use z500_prepared as background (will need to adapt visualization)
    try:
        # Try to compute anomalies
        z500_anom = compute_anomalies(z500_prepared)
    except:
        # If that fails, use the prepared data directly
        # (Animation function will need to handle this)
        z500_anom = z500_prepared.copy()
        print("   Note: Using z500 data directly (anomalies could not be computed)")
    
    # Create blocked_mask with full latitude range
    # Expand final_mask to full latitude range
    blocked_mask_full = np.zeros((n_times_prep, n_lats_prep, n_lons_prep), dtype='uint8')
    for t in range(n_times_prep):
        if t < n_times_final:
            for i, lat_idx in enumerate(lat_central_indices):
                if i < final_mask.shape[1]:
                    blocked_mask_full[t, lat_idx, :] = final_mask.values[t, i, :].astype('uint8')
    
    blocked_mask = xr.DataArray(
        blocked_mask_full,
        coords={'time': z500_prepared.time, 'lat': lat_full, 'lon': lon_full},
        dims=['time', 'lat', 'lon'],
        name='blocked_mask'
    )
    
    # Create output dictionary
    # Use z500_prepared coordinates to ensure alignment
    event_mask_da = xr.DataArray(
        event_mask,
        coords={'time': z500_prepared.time, 'lat': lat_full, 'lon': lon_full},
        dims=['time', 'lat', 'lon'],
        name='event_mask'
    )
    
    ano_format_stats = {
        'event_mask': event_mask_da,
        'z500_anom': z500_anom,
        'blocked_mask': blocked_mask,
        'event_areas': event_areas,
        'num_events': len(persistent_events),
        'all_event_ids': list(event_areas.keys()),
        'event_durations': {idx+1: e['duration'] for idx, e in enumerate(persistent_events)}
    }
    
    return ano_format_stats


# =============================================================================
# Standard Animation (using Matplotlib FuncAnimation)
# =============================================================================

def create_event_animation_gif(event_id, ano_stats, save_path=None, 
                                title_prefix='', figsize=(12, 10), 
                                fps=2, dpi=100):
    """
    Create an animated GIF showing the time evolution of a blocking event.
    
    Uses Matplotlib's FuncAnimation - suitable for smaller datasets.
    
    Parameters
    ----------
    event_id : int
        Event ID to visualize
    ano_stats : dict
        Dictionary containing event information from ano_blocking_complete
    save_path : str, optional
        Path to save the GIF file (e.g., 'event_1262.gif')
        If None, uses default naming
    title_prefix : str
        Prefix for the title (e.g., 'JJA PlaSim', 'DJF ERA5')
    figsize : tuple
        Figure size (width, height) in inches
    fps : int
        Frames per second for the animation (default: 2)
    dpi : int
        Resolution of the output GIF (default: 100)
    
    Returns
    -------
    str
        Path to the saved GIF file
    """
    
    # Extract data
    event_mask_raw = ano_stats['event_mask']  # (time, lat, lon) - full domain
    z500_anom_raw = ano_stats['z500_anom']  # Anomaly field (time, lat, lon)
    blocked_mask_raw = ano_stats['blocked_mask']  # Binary mask (time, lat, lon)
    
    # Ensure data is xarray (convert from numpy if needed)
    if isinstance(z500_anom_raw, xr.DataArray):
        z500_anom = z500_anom_raw
        lat = z500_anom.lat.values
        lon = z500_anom.lon.values
        
        # Convert other arrays to xarray with same coordinates
        if not isinstance(event_mask_raw, xr.DataArray):
            event_mask = xr.DataArray(
                event_mask_raw,
                coords={'time': z500_anom.time, 'lat': z500_anom.lat, 'lon': z500_anom.lon},
                dims=['time', 'lat', 'lon']
            )
        else:
            event_mask = event_mask_raw
            
        if not isinstance(blocked_mask_raw, xr.DataArray):
            blocked_mask = xr.DataArray(
                blocked_mask_raw,
                coords={'time': z500_anom.time, 'lat': z500_anom.lat, 'lon': z500_anom.lon},
                dims=['time', 'lat', 'lon']
            )
        else:
            blocked_mask = blocked_mask_raw
    else:
        # All are numpy arrays - need to extract coordinates from somewhere
        # Assume standard dimensions
        raise ValueError(
            "z500_anom must be an xarray DataArray with coordinates. "
            "If using numpy arrays, please convert to xarray first."
        )
    
    # Debug: Check dimensions
    print(f"\n  Debug: z500_anom shape: {z500_anom.shape}")
    print(f"  Debug: event_mask shape: {event_mask.shape}")
    print(f"  Debug: blocked_mask shape: {blocked_mask.shape}")
    print(f"  Debug: lat length: {len(lat)}, lon length: {len(lon)}")
    
    # Get the time steps where this event occurs
    event_times = np.where((event_mask == event_id).any(axis=(1, 2)))[0]
    
    if len(event_times) == 0:
        print(f"Error: Event {event_id} not found in event_mask")
        return None
    
    print(f"\n{'='*70}")
    print(f"Creating Animation for Event {event_id}")
    print(f"{'='*70}")
    print(f"  Duration: {len(event_times)} days")
    print(f"  Start time index: {event_times[0]}")
    print(f"  End time index: {event_times[-1]}")
    print(f"  Time range: {z500_anom.time.values[event_times[0]]} to {z500_anom.time.values[event_times[-1]]}")
    
    # Get event area information
    event_areas = list(ano_stats['event_areas'][event_id])
    print(f"  Mean area: {np.mean(event_areas)/1e6:.2f} × 10⁶ km²")
    print(f"  Area range: [{np.min(event_areas)/1e6:.2f}, {np.max(event_areas)/1e6:.2f}] × 10⁶ km²")
    print(f"  Animation frames: {len(event_times)}")
    print(f"  FPS: {fps} (duration: {len(event_times)/fps:.1f} seconds)")
    
    # Set up the figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())
    ax.set_extent([-180, 180, 30, 90], ccrs.PlateCarree())
    
    # Add static map features
    ax.coastlines(linewidth=0.5, color='black')
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
    ax.gridlines(draw_labels=False, linewidth=0.5, 
                 color='gray', alpha=0.3, linestyle='-')
    
    # Define anomaly levels
    levels_anom = np.arange(-300, 301, 30)
    
    # Initialize plot elements (will be updated in animation)
    # Get first frame data
    t_idx_0 = event_times[0]
    z500_0 = z500_anom.isel(time=t_idx_0)
    blocked_0 = blocked_mask.isel(time=t_idx_0)
    event_mask_0 = (event_mask.isel(time=t_idx_0) == event_id)  # Use isel for consistency
    
    print(f"  Debug: z500_0 shape: {z500_0.shape}")
    print(f"  Debug: event_mask_0 shape: {event_mask_0.shape}")
    print(f"  Debug: event_mask_0 lat: {len(event_mask_0.lat.values)}, lon: {len(event_mask_0.lon.values)}")
    
    # Add cyclic point to data for proper contour wrapping
    # add_cyclic_point returns both the data and the updated coordinate
    z500_0_cyclic, lon_cyclic = add_cyclic_point(z500_0.values, coord=lon)
    event_mask_0_cyclic, _ = add_cyclic_point(event_mask_0.values.astype(float), coord=lon)
    blocked_0_cyclic, _ = add_cyclic_point(blocked_0.values, coord=lon)
    
    # Verify shapes match
    print(f"  Debug: z500_0_cyclic shape: {z500_0_cyclic.shape}, lon_cyclic length: {len(lon_cyclic)}")
    if z500_0_cyclic.shape[1] != len(lon_cyclic):
        print(f"  Warning: Shape mismatch! z500_0_cyclic has {z500_0_cyclic.shape[1]} lons but lon_cyclic has {len(lon_cyclic)}")
        # Fix the mismatch by using the correct coordinate
        lon_cyclic = np.linspace(lon[0], lon[-1] + (lon[1] - lon[0]), z500_0_cyclic.shape[1])
    
    # Create initial contourf
    cf = ax.contourf(lon_cyclic, lat, z500_0_cyclic, 
                     levels=levels_anom, 
                     cmap='RdBu_r', 
                     transform=ccrs.PlateCarree(),
                     alpha=0.6, extend='both')
    
    # Create initial contours (will be updated)
    event_contour = ax.contour(lon_cyclic, lat, event_mask_0_cyclic, 
                               levels=[0.5], 
                               colors='black', 
                               linewidths=3,
                               transform=ccrs.PlateCarree())
    
    blocked_contour = ax.contour(lon_cyclic, lat, blocked_0_cyclic, 
                                 levels=[0.5], 
                                 colors='orange', 
                                 linewidths=1.5,
                                 linestyles='--',
                                 transform=ccrs.PlateCarree(),
                                 alpha=0.5)
    
    # Add colorbar
    cbar = fig.colorbar(cf, ax=ax, orientation='horizontal', 
                       pad=0.05, shrink=0.8, label='Z500 Anomaly (m)')
    
    # Create title text
    time_str_0 = str(z500_anom.time.values[t_idx_0])[:10]
    area_0 = event_areas[0] / 1e6
    title_text = f'Event {event_id}'
    if title_prefix:
        title_text += f' - {title_prefix}'
    title_text += f'\nDay 1/{len(event_times)} | {time_str_0} | Area: {area_0:.2f}×10⁶ km²\n'
    title_text += 'Black contour = Event boundary | Orange dashed = All blocked regions'
    
    title = ax.set_title(title_text, fontsize=12, fontweight='bold', pad=10)
    
    # Animation update function
    def update(frame):
        """Update function for animation"""
        # Get data for this frame
        day_idx = frame
        t_idx = event_times[day_idx]
        
        z500_t = z500_anom.isel(time=t_idx)
        blocked_t = blocked_mask.isel(time=t_idx)
        event_mask_t = (event_mask.isel(time=t_idx) == event_id)
        
        # Clear previous contours
        for coll in ax.collections:
            if coll != cf:  # Don't remove the filled contour
                coll.remove()
        
        # Update filled contour (anomaly field)
        # Note: contourf creates new collections, so we need to clear and redraw
        for coll in list(ax.collections):
            coll.remove()
        
        # Add cyclic point to data for proper contour wrapping
        z500_t_cyclic, _ = add_cyclic_point(z500_t.values, coord=lon)
        event_mask_t_cyclic, _ = add_cyclic_point(event_mask_t.values.astype(float), coord=lon)
        blocked_t_cyclic, _ = add_cyclic_point(blocked_t.values, coord=lon)
        
        # Verify shapes still match (only print on first iteration for debugging)
        if day_idx == 0:
            print(f"  Debug update: z500_t_cyclic shape: {z500_t_cyclic.shape}, lon_cyclic length: {len(lon_cyclic)}")
        
        cf_new = ax.contourf(lon_cyclic, lat, z500_t_cyclic, 
                            levels=levels_anom, 
                            cmap='RdBu_r', 
                            transform=ccrs.PlateCarree(),
                            alpha=0.6, extend='both')
        
        # Add event boundary contour
        if event_mask_t.values.any():
            ax.contour(lon_cyclic, lat, event_mask_t_cyclic, 
                      levels=[0.5], 
                      colors='black', 
                      linewidths=3,
                      transform=ccrs.PlateCarree())
        
        # Add all blocked regions contour
        if blocked_t.values.any():
            ax.contour(lon_cyclic, lat, blocked_t_cyclic, 
                      levels=[0.5], 
                      colors='orange', 
                      linewidths=1.5,
                      linestyles='--',
                      transform=ccrs.PlateCarree(),
                      alpha=0.5)
        
        # Re-add coastlines and features (since collections were cleared)
        ax.coastlines(linewidth=0.5, color='black')
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
        
        # Update title
        time_str = str(z500_anom.time.values[t_idx])[:10]
        area_day = event_areas[day_idx] / 1e6
        title_text = f'Event {event_id}'
        if title_prefix:
            title_text += f' - {title_prefix}'
        title_text += f'\nDay {day_idx+1}/{len(event_times)} | {time_str} | Area: {area_day:.2f}×10⁶ km²\n'
        title_text += 'Black contour = Event boundary | Orange dashed = All blocked regions'
        title.set_text(title_text)
        
        # Print progress
        if (day_idx + 1) % max(1, len(event_times) // 10) == 0:
            print(f"  Progress: {day_idx+1}/{len(event_times)} frames")
        
        return ax.collections + [title]
    
    # Create animation
    print(f"\nGenerating animation...")
    anim = FuncAnimation(fig, update, frames=len(event_times), 
                        interval=1000/fps, blit=False, repeat=True)
    
    # Save as GIF
    if save_path is None:
        save_path = f'/home/zhixingliu/projects/aires/figures/Event_{event_id}_animation.gif'
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    print(f"Saving GIF to: {save_path}")
    print(f"  (This may take a while for long events...)")
    
    writer = PillowWriter(fps=fps)
    anim.save(save_path, writer=writer, dpi=dpi)
    
    plt.close(fig)
    
    # Get file size
    file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
    
    print(f"\n{'='*70}")
    print(f"✅ Animation saved successfully!")
    print(f"  File: {save_path}")
    print(f"  Size: {file_size_mb:.2f} MB")
    print(f"  Frames: {len(event_times)}")
    print(f"  Duration: {len(event_times)/fps:.1f} seconds at {fps} FPS")
    print(f"{'='*70}\n")
    
    return save_path


# =============================================================================
# Fast Animation (using imageio - optimized for large datasets)
# =============================================================================

def create_event_animation_gif_fast(event_id, ano_stats, save_path=None,
                                    title_prefix='', figsize=(12, 10),
                                    fps=1, dpi=100, max_lat_points=90, max_lon_points=360,
                                    event_start_times=None,
                                    # New parameters for P_block and centroid panels
                                    show_diagnostics=False,
                                    region_lon_min=30.0, region_lon_max=100.0,
                                    region_lat_min=55.0, region_lat_max=75.0,
                                    # New parameter to visualize entire simulation
                                    animate_full_simulation=False):
    """
    Create animated GIF for a blocking event or entire simulation.
    
    Optimized for large datasets (ERA5-sized grids):
    - Builds event start index once (O(T)) and reuses it
    - Extracts ONLY the event's timesteps (no 14GB loads)
    - Subsets to 25–90°N and downsamples if needed
    - Uses imageio for quick GIF writing
    
    Parameters
    ----------
    event_id : int
        Event ID to animate (ignored if animate_full_simulation=True)
    ano_stats : dict
        Event statistics from ano_blocking_complete
    save_path : str, optional
        Path to save GIF
    title_prefix : str
        Prefix for title
    figsize : tuple
        Figure size (width, height)
    fps : int
        Frames per second
    dpi : int
        Resolution
    max_lat_points, max_lon_points : int
        Maximum grid points for downsampling
    event_start_times : dict, optional
        Pre-computed event start times
    show_diagnostics : bool
        If True, show P_block time series and centroid evolution panels
    region_lon_min, region_lon_max : float
        Longitude bounds for regional scoring (used when show_diagnostics=True)
    region_lat_min, region_lat_max : float
        Latitude bounds for regional scoring (used when show_diagnostics=True)
    animate_full_simulation : bool
        If True, animate entire simulation instead of single event
    
    Returns
    -------
    str
        Path to the saved GIF file
    """
    # Determine what to animate: single event or full simulation
    if animate_full_simulation:
        print(f"\n  Creating animation for FULL SIMULATION (fast v4)...")
        z500_anom_raw = ano_stats['z500_anom']
        if not isinstance(z500_anom_raw, xr.DataArray):
            raise ValueError("z500_anom must be xarray DataArray")
        
        # Use all timesteps
        duration = len(z500_anom_raw.time)
        event_times = np.arange(duration)
        print(f"  Duration: {duration} days (entire simulation)")
        print(f"  Time indices: {event_times[0]} to {event_times[-1]}")
        
        # For full simulation, no specific event areas
        event_areas = [0] * duration  # Placeholder, will be computed from event_mask
        current_event_id = event_id  # Use for tracking which events appear
    else:
        print(f"\n  Creating animation for Event {event_id} (fast v4)...")
        duration = len(ano_stats['event_areas'][event_id])
        print(f"  Duration: {duration} days")

        event_times = get_event_time_range(ano_stats, event_id, event_start_times)
        if len(event_times) == 0:
            print(f"  Error: Event {event_id} not found")
            return None
        print(f"  Time indices: {event_times[0]} to {event_times[-1]}")
        
        event_areas = list(ano_stats['event_areas'][event_id])
        current_event_id = event_id

    # Extract coordinates
    z500_anom_raw = ano_stats['z500_anom']
    if not isinstance(z500_anom_raw, xr.DataArray):
        raise ValueError("z500_anom must be xarray DataArray")
    lat = z500_anom_raw.lat.values
    lon = z500_anom_raw.lon.values
    time_coords = z500_anom_raw.time.values

    lat_mask = (lat >= 25) & (lat <= 90)
    lat_indices = np.where(lat_mask)[0]
    lat_subset = lat[lat_mask]

    event_time_list = list(map(int, event_times))
    print(f"  Extracting {len(event_time_list)} timesteps × {len(lat_subset)} lats")

    # Extract minimal data (no 14GB loads!)
    z500_event = z500_anom_raw.isel(time=event_time_list, lat=lat_indices).values
    event_mask_raw = ano_stats['event_mask']
    blocked_mask_raw = ano_stats['blocked_mask']

    if isinstance(event_mask_raw, xr.DataArray):
        event_mask_event = event_mask_raw.isel(time=event_time_list, lat=lat_indices).values
    else:
        event_mask_event = np.asarray(event_mask_raw)[event_time_list][:, lat_mask, :]

    if isinstance(blocked_mask_raw, xr.DataArray):
        blocked_mask_event = blocked_mask_raw.isel(time=event_time_list, lat=lat_indices).values
    else:
        blocked_mask_event = np.asarray(blocked_mask_raw)[event_time_list][:, lat_mask, :]
    
    # For full simulation mode, compute event areas dynamically
    if animate_full_simulation:
        print(f"  Computing event areas for full simulation...")
        event_areas = []
        for t in range(duration):
            # Count all blocked pixels at this timestep
            blocked_count = np.sum(blocked_mask_event[t] > 0)
            # Approximate area (this is a rough estimate)
            event_areas.append(blocked_count * 1e5)  # Rough km² per grid cell

    # Downsample if needed
    lat_step = max(1, len(lat_subset) // max_lat_points)
    lon_step = max(1, len(lon) // max_lon_points)
    if lat_step > 1 or lon_step > 1:
        print(f"  Downsampling grid: lat/{lat_step}, lon/{lon_step}")
        lat_ds = lat_subset[::lat_step]
        lon_ds = lon[::lon_step]
        z500_ds = z500_event[:, ::lat_step, ::lon_step]
        event_mask_ds = event_mask_event[:, ::lat_step, ::lon_step]
        blocked_mask_ds = blocked_mask_event[:, ::lat_step, ::lon_step]
    else:
        lat_ds, lon_ds = lat_subset, lon
        z500_ds, event_mask_ds, blocked_mask_ds = z500_event, event_mask_event, blocked_mask_event

    print(f"  Final grid: {len(lat_ds)} × {len(lon_ds)}")
    
    # Check if cyclic point is already present (e.g. 0 and 360 both in data)
    # If lon_ds[-1] is close to lon_ds[0] + 360, we don't need to add it again
    has_cyclic_point = np.isclose(lon_ds[-1], lon_ds[0] + 360)
    
    if has_cyclic_point:
        lon_cyclic = lon_ds
    else:
        lon_cyclic = np.append(lon_ds, lon_ds[0] + 360)
        
    levels_anom = np.arange(-300, 301, 30)
    
    # ---------------------------------------------------------------------
    # Pre-compute P_block and centroid time series if show_diagnostics=True
    # ---------------------------------------------------------------------
    P_block_series = []
    centroid_lon_series = []
    centroid_lat_series = []
    
    if show_diagnostics:
        print(f"  Computing P_block and centroid time series for diagnostics...")
        
        # Create regional mask
        lat_full = z500_anom_raw.lat.values
        lon_full = z500_anom_raw.lon.values
        weights_lat = np.cos(np.deg2rad(lat_full))
        
        # Regional mask W
        lat_in_region = (lat_full >= region_lat_min) & (lat_full <= region_lat_max)
        lon_in_region = (lon_full >= region_lon_min) & (lon_full <= region_lon_max)
        W = np.outer(lat_in_region.astype(float), lon_in_region.astype(float))
        
        # Get full blocked mask for these times
        if isinstance(blocked_mask_raw, xr.DataArray):
            blocked_full = blocked_mask_raw.isel(time=event_time_list).values
        else:
            blocked_full = np.asarray(blocked_mask_raw)[event_time_list]
        
        # Get full z500_anom for these times
        z500_full = z500_anom_raw.isel(time=event_time_list).values
        z_plus = np.maximum(z500_full, 0)  # Z'+ = max(0, Z')
        
        LON_grid, LAT_grid = np.meshgrid(lon_full, lat_full)
        W_weights = np.broadcast_to(weights_lat[:, None], (len(lat_full), len(lon_full)))
        
        for t in range(duration):
            # Regional blocking mask: I_R = blocked * W
            I_R_t = blocked_full[t] * W
            
            # P_block(t) = sum(I_R * Z'+ * cos(lat))
            integrand = I_R_t * z_plus[t] * W_weights
            P_block_t = np.sum(integrand)
            P_block_series.append(P_block_t)
            
            # Centroid (within region)
            mask_region = I_R_t > 0
            if np.any(mask_region):
                w_i = W_weights[mask_region]
                lam_i = LON_grid[mask_region]
                phi_i = LAT_grid[mask_region]
                denom = np.sum(w_i)
                if denom > 0:
                    centroid_lon = np.sum(w_i * lam_i) / denom
                    centroid_lat = np.sum(w_i * phi_i) / denom
                else:
                    centroid_lon, centroid_lat = np.nan, np.nan
            else:
                centroid_lon, centroid_lat = np.nan, np.nan
            
            centroid_lon_series.append(centroid_lon)
            centroid_lat_series.append(centroid_lat)
        
        P_block_series = np.array(P_block_series)
        centroid_lon_series = np.array(centroid_lon_series)
        centroid_lat_series = np.array(centroid_lat_series)
        
        print(f"    P_block range: {P_block_series.min():.0f} to {P_block_series.max():.0f}")
        print(f"    Centroid lon range: {np.nanmin(centroid_lon_series):.1f}° to {np.nanmax(centroid_lon_series):.1f}°")

    frames = []
    print(f"  Generating {len(event_times)} frames...")
    
    # Adjust figsize for diagnostics panels
    if show_diagnostics:
        fig_width = figsize[0] + 5  # Extra width for right panels
        fig_height = figsize[1] + 1  # Slightly taller for better proportions
    else:
        fig_width, fig_height = figsize

    for frame_idx in range(len(event_times)):
        if show_diagnostics:
            # Create figure with GridSpec for multi-panel layout
            fig = plt.figure(figsize=(fig_width, fig_height))
            from matplotlib.gridspec import GridSpec
            # height_ratios [1, 2] makes centroid panel 2x as tall as time series
            # hspace=0.3 adds vertical space between top and bottom rows
            # wspace=0.1 adds horizontal space between map and panels
            gs = GridSpec(2, 2, figure=fig, width_ratios=[1.8, 1], height_ratios=[1, 2],
                         wspace=0.1, hspace=0.3)
            
            # Main map (left, spans both rows)
            ax_map = fig.add_subplot(gs[:, 0], projection=ccrs.NorthPolarStereo())
            
            # Blocking intensity time series (top right)
            ax_pblock = fig.add_subplot(gs[0, 1])
            
            # Centroid evolution map (bottom right) - use same projection as main map
            # central_longitude=0 to match the main map orientation
            ax_centroid = fig.add_subplot(gs[1, 1], projection=ccrs.NorthPolarStereo(central_longitude=0))
        else:
            fig = plt.figure(figsize=figsize)
            ax_map = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())
        
        ax_map.set_extent([-180, 180, 30, 90], ccrs.PlateCarree())

        z500_t = z500_ds[frame_idx]
        blocked_t = blocked_mask_ds[frame_idx]
        
        # In full simulation mode, show ALL events; otherwise show only the specific event
        if animate_full_simulation:
            # Show any event (event_mask > 0) as black contours
            event_t = (event_mask_ds[frame_idx] > 0).astype(float)
        else:
            # Show only the specific event_id
            event_t = (event_mask_ds[frame_idx] == current_event_id).astype(float)

        if has_cyclic_point:
            z500_cyc = z500_t
            event_cyc = event_t
            blocked_cyc = blocked_t
        else:
            z500_cyc = np.concatenate([z500_t, z500_t[:, :1]], axis=1)
            event_cyc = np.concatenate([event_t, event_t[:, :1]], axis=1)
            blocked_cyc = np.concatenate([blocked_t, blocked_t[:, :1]], axis=1)

        cf = ax_map.contourf(lon_cyclic, lat_ds, z500_cyc,
                         levels=levels_anom, cmap='RdBu_r',
                         transform=ccrs.PlateCarree(), alpha=0.6, extend='both')
        if event_cyc.any():
            ax_map.contour(lon_cyclic, lat_ds, event_cyc, levels=[0.5],
                      colors='black', linewidths=3, transform=ccrs.PlateCarree())
        if blocked_cyc.any():
            ax_map.contour(lon_cyclic, lat_ds, blocked_cyc, levels=[0.5],
                      colors='orange', linewidths=1.5, linestyles='--',
                      transform=ccrs.PlateCarree(), alpha=0.5)

        ax_map.coastlines(linewidth=0.5)
        ax_map.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
        
        # Draw region of interest box on main map (when showing diagnostics)
        if show_diagnostics:
            # Use spherical interpolation for proper curved box on polar projection
            n_interp = 50
            # Bottom edge (constant min_lat)
            bottom_lons = np.linspace(region_lon_min, region_lon_max, n_interp)
            bottom_lats = np.full(n_interp, region_lat_min)
            # Right edge (constant max_lon)
            right_lats = np.linspace(region_lat_min, region_lat_max, n_interp)
            right_lons = np.full(n_interp, region_lon_max)
            # Top edge (constant max_lat, reversed)
            top_lons = np.linspace(region_lon_max, region_lon_min, n_interp)
            top_lats = np.full(n_interp, region_lat_max)
            # Left edge (constant min_lon, reversed)
            left_lats = np.linspace(region_lat_max, region_lat_min, n_interp)
            left_lons = np.full(n_interp, region_lon_min)
            
            region_box_lons = np.concatenate([bottom_lons, right_lons, top_lons, left_lons])
            region_box_lats = np.concatenate([bottom_lats, right_lats, top_lats, left_lats])
            ax_map.plot(region_box_lons, region_box_lats, 'b-', linewidth=2, 
                       transform=ccrs.PlateCarree(), alpha=0.8, zorder=5)
        
        # Colorbar for main map
        if show_diagnostics:
            cbar = fig.colorbar(cf, ax=ax_map, orientation='horizontal', pad=0.08, shrink=0.7, aspect=25)
            cbar.set_label('Z500 Anomaly (m)', fontsize=10)
            cbar.ax.tick_params(labelsize=9)
        else:
            fig.colorbar(cf, ax=ax_map, orientation='horizontal', pad=0.05, shrink=0.8, label='Z500 Anomaly (m)')

        t_idx = event_times[frame_idx]
        time_str = str(time_coords[t_idx])[:10]
        area = event_areas[frame_idx] / 1e6
        
        # Generate title based on visualization mode
        if animate_full_simulation:
            # Use provided event_id for title (e.g., 320 for a specific blocking event series)
            if event_id is not None:
                title = f'Event ID {event_id} + Full Simulation'
            else:
                title = f'Full Simulation'
            
            if title_prefix:
                title += f' - {title_prefix}'
            # Second line: Day xx/xx | date | Area
            title += f'\nDay {frame_idx+1}/{duration} | {time_str} | Area: ${area:.2f} \\times 10^6\\ \\mathrm{{km}}^2$'
            if not show_diagnostics:
                title += '\nBlack contour = Blocking events | Orange dashed = All blocked regions'
        else:
            title = f'Event {current_event_id}'
            if title_prefix:
                title += f' - {title_prefix}'
            # Use LaTeX formatting for area (matplotlib supports this)
            title += f'\nDay {frame_idx+1}/{duration} | {time_str} | Area: ${area:.2f} \\times 10^6\\ \\mathrm{{km}}^2$'
            if not show_diagnostics:
                title += '\nBlack contour = Event boundary | Orange dashed = All blocked regions'
        
        ax_map.set_title(title, fontsize=12, fontweight='bold', pad=10)
        
        # -------------------------------------------------------------
        # Draw diagnostic panels if enabled
        # -------------------------------------------------------------
        if show_diagnostics:
            days = np.arange(1, duration + 1)
            
            # === TOP RIGHT: Regional Blocking Intensity time series ===
            ax_pblock.fill_between(days, 0, P_block_series, alpha=0.3, color='steelblue')
            ax_pblock.plot(days, P_block_series, 'o-', color='steelblue', markersize=4, linewidth=1.5)
            
            # Current day marker
            ax_pblock.axvline(x=frame_idx + 1, color='red', linewidth=2, linestyle='--', alpha=0.8)
            ax_pblock.plot(frame_idx + 1, P_block_series[frame_idx], 'o', 
                          color='red', markersize=10, markeredgecolor='darkred', markeredgewidth=2)
            
            ax_pblock.set_xlim(0.5, duration + 0.5)
            ax_pblock.set_ylim(0, P_block_series.max() * 1.15)
            ax_pblock.set_xlabel('Day', fontsize=11)
            ax_pblock.set_ylabel('Area-weighted Positive\nZ500 Anomaly (m)', fontsize=10)
            ax_pblock.set_title('Time Evolution of Regional Blocking Intensity', fontsize=12, fontweight='bold')
            ax_pblock.grid(True, alpha=0.3)
            
            # Annotate current value
            ax_pblock.text(0.98, 0.95, f'Day {frame_idx+1}: {P_block_series[frame_idx]:.0f} m',
                          transform=ax_pblock.transAxes, ha='right', va='top',
                          fontsize=10, fontweight='bold', color='red',
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # === BOTTOM RIGHT: Centroid evolution map (zoomed polar view) ===
            # Use polar stereographic projection zoomed to show ~1/4 of hemisphere
            # centered on the region of interest (same look as main map but zoomed)
            lon_padding = 30  # degrees padding around region
            lat_min_view = max(region_lat_min - 15, 40)  # Don't go below 40°N for polar view
            
            # Set extent to show the region with padding (polar view zoomed in)
            ax_centroid.set_extent([region_lon_min - lon_padding, region_lon_max + lon_padding, 
                                   lat_min_view, 90], ccrs.PlateCarree())
            ax_centroid.coastlines(linewidth=0.8)
            ax_centroid.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.5)
            ax_centroid.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
            ax_centroid.gridlines(draw_labels=False, linewidth=0.3, alpha=0.5, linestyle='--')
            
            # Draw regional box with proper spherical representation (interpolated points)
            n_interp = 30  # points along each edge
            # Bottom edge (constant min_lat)
            bottom_lons = np.linspace(region_lon_min, region_lon_max, n_interp)
            bottom_lats = np.full(n_interp, region_lat_min)
            # Right edge (constant max_lon)
            right_lats = np.linspace(region_lat_min, region_lat_max, n_interp)
            right_lons = np.full(n_interp, region_lon_max)
            # Top edge (constant max_lat, reversed)
            top_lons = np.linspace(region_lon_max, region_lon_min, n_interp)
            top_lats = np.full(n_interp, region_lat_max)
            # Left edge (constant min_lon, reversed)
            left_lats = np.linspace(region_lat_max, region_lat_min, n_interp)
            left_lons = np.full(n_interp, region_lon_min)
            
            box_lons = np.concatenate([bottom_lons, right_lons, top_lons, left_lons])
            box_lats = np.concatenate([bottom_lats, right_lats, top_lats, left_lats])
            ax_centroid.plot(box_lons, box_lats, 'b-', linewidth=2.5, transform=ccrs.PlateCarree(), alpha=0.8)
            
            # Plot centroid trail (past positions)
            valid_past = ~np.isnan(centroid_lon_series[:frame_idx+1])
            if np.any(valid_past):
                past_lons = centroid_lon_series[:frame_idx+1][valid_past]
                past_lats = centroid_lat_series[:frame_idx+1][valid_past]
                
                # Trail with fading colors
                n_past = len(past_lons)
                if n_past > 1:
                    # Draw connecting line
                    ax_centroid.plot(past_lons, past_lats, '-', color='purple', 
                                    linewidth=2, alpha=0.6, transform=ccrs.PlateCarree())
                
                # Draw past points with fading alpha
                for i, (plon, plat) in enumerate(zip(past_lons[:-1], past_lats[:-1])):
                    alpha = 0.3 + 0.5 * (i / max(n_past - 1, 1))
                    ax_centroid.plot(plon, plat, 'o', color='purple', markersize=6,
                                    alpha=alpha, transform=ccrs.PlateCarree())
                
                # Current centroid (bright red)
                if not np.isnan(centroid_lon_series[frame_idx]):
                    ax_centroid.plot(centroid_lon_series[frame_idx], centroid_lat_series[frame_idx],
                                    'o', color='red', markersize=14, markeredgecolor='darkred',
                                    markeredgewidth=2, transform=ccrs.PlateCarree(), zorder=10)
            
            # Title with current centroid coordinates
            if not np.isnan(centroid_lon_series[frame_idx]):
                ax_centroid.set_title(f'Centroid Evolution\n(λ={centroid_lon_series[frame_idx]:.1f}°E, φ={centroid_lat_series[frame_idx]:.1f}°N)', 
                                     fontsize=11, fontweight='bold')
            else:
                ax_centroid.set_title('Centroid Evolution\n(No blocking in region)', 
                                     fontsize=11, fontweight='bold')

        buf = BytesIO()
        # Use subplots_adjust for consistent layout across all frames (fixes first frame sizing issue)
        if show_diagnostics:
            # Fixed margins for consistent frame sizing across all frames
            fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.08)
            fig.savefig(buf, format='png', dpi=dpi, facecolor='white')
        else:
            # For single map, use consistent layout as well
            fig.subplots_adjust(left=0.05, right=0.95, top=0.88, bottom=0.12)
            fig.savefig(buf, format='png', dpi=dpi, facecolor='white')
        buf.seek(0)
        frames.append(imageio.imread(buf))
        buf.close()
        plt.close(fig)

        if (frame_idx + 1) % max(1, duration // 3) == 0:
            print(f"    Frame {frame_idx+1}/{duration}")

    if save_path is None:
        save_path = f'/home/zhixingliu/projects/aires/figures/Event_{event_id}.gif'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    imageio.mimsave(save_path, frames, duration=1.0 / fps, loop=0)
    size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"  ✅ Saved {save_path} ({size_mb:.1f} MB)\n")
    return save_path


# =============================================================================
# Driver: create GIFs for requested durations
# =============================================================================

def create_duration_specific_gifs_fast(event_info, z500_data, method, season,
                                       target_durations,
                                       output_base_dir='/home/zhixingliu/projects/aires/figures',
                                       fps=1, dpi=100, max_lat_points=90, max_lon_points=360):
    """
    Create GIFs for events with specific durations.
    
    Parameters
    ----------
    event_info : dict
        Event information from blocking analysis
    z500_data : xarray.DataArray
        Z500 data
    method : str
        Blocking detection method ('ABS' or 'ANO')
    season : str
        Season name (e.g., 'JJA', 'DJF')
    target_durations : list
        List of target durations in days
    output_base_dir : str
        Base directory for output files
    fps : int
        Frames per second
    dpi : int
        Resolution
    max_lat_points, max_lon_points : int
        Maximum grid points for downsampling
    
    Returns
    -------
    dict
        Mapping of duration -> GIF path
    """
    print("\n" + "🚀" * 30)
    print("USING FAST VERSION v4: create_duration_specific_gifs_fast")
    print("  ✓ Smart event indexing once")
    print("  ✓ Minimal data extraction per event")
    print("  ✓ Downsampled plotting grid")
    print("🚀" * 30 + "\n")

    method = method.upper()
    output_folder = os.path.join(output_base_dir, f'{method} {season}')
    os.makedirs(output_folder, exist_ok=True)
    print(f"📁 Output folder: {output_folder}")

    if 'event_count' in event_info:
        if event_info.get('event_count', 0) == 0:
            return {dur: None for dur in target_durations}
        print("🔄 Converting ABS format → ANO format...")
        ano_stats = convert_abs_events_to_ano_format(event_info, z500_data)
    elif 'num_events' in event_info:
        if event_info.get('num_events', 0) == 0:
            return {dur: None for dur in target_durations}
        ano_stats = event_info
    else:
        raise ValueError("Unknown event_info format")

    event_durations = ano_stats.get('event_durations', {})
    all_event_ids = ano_stats.get('all_event_ids', list(event_durations.keys()))

    print(f"🎬 Preparing GIFs for durations {target_durations}")
    events_to_index = [eid for eid in all_event_ids if event_durations.get(eid) in target_durations]

    event_mask = ano_stats['event_mask']
    if isinstance(event_mask, xr.DataArray):
        event_mask_np = event_mask.values
    else:
        event_mask_np = np.asarray(event_mask)

    print(f"📊 Building start-time index for {len(events_to_index)} events...")
    event_start_times = build_event_start_times(event_mask_np, events_to_index)

    gif_paths = {}
    for target_dur in target_durations:
        matching = [eid for eid in all_event_ids if event_durations.get(eid) == target_dur]
        if not matching:
            print(f"⚠️  No events with duration = {target_dur} days")
            gif_paths[target_dur] = None
            continue

        event_id = matching[0]  # pick first (or change to random if desired)
        print("\n" + "=" * 60)
        print(f"Duration {target_dur} days → Using Event {event_id} ({len(matching)} available)")
        print("=" * 60)

        filename = f'{method}_{season}_{target_dur}days_v4.gif'
        save_path = os.path.join(output_folder, filename)
        try:
            gif_path = create_event_animation_gif_fast(
                event_id=event_id,
                ano_stats=ano_stats,
                save_path=save_path,
                title_prefix=f'{method} {season}',
                fps=fps,
                dpi=dpi,
                max_lat_points=max_lat_points,
                max_lon_points=max_lon_points,
                event_start_times=event_start_times
            )
            gif_paths[target_dur] = gif_path
        except Exception as exc:
            print(f"❌ Error while creating GIF for duration {target_dur}: {exc}")
            import traceback
            traceback.print_exc()
            gif_paths[target_dur] = None

    success = sum(1 for path in gif_paths.values() if path)
    print(f"\n✅ GIF creation complete: {success}/{len(target_durations)} succeeded")
    return gif_paths


