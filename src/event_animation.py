import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import xarray as xr
from matplotlib.animation import FuncAnimation, PillowWriter
from cartopy.util import add_cyclic_point
import os


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


def create_event_animation_gif(event_id, ano_stats, save_path=None, 
                                title_prefix='', figsize=(12, 10), 
                                fps=2, dpi=100):
    """
    Create an animated GIF showing the time evolution of a blocking event.
    
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


def convert_abs_events_to_ano_format(abs_event_info, z500_data):
    """
    Convert ABS event information to format compatible with create_event_animation_gif.
    
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
        Dictionary in format expected by create_event_animation_gif:
        - 'event_mask': (time, lat, lon) array with unique event IDs
        - 'z500_anom': Anomaly field or z500 data
        - 'blocked_mask': Binary blocking mask
        - 'event_areas': Dict mapping event_id to list of areas
    """
    import xarray as xr
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


# Example usage:
if __name__ == "__main__":
    # Assuming you have ano_stats_jja loaded
    # For Event 1262
    # gif_path = create_event_animation_gif(
    #     1262, 
    #     ano_stats_jja,
    #     save_path='/home/zhixingliu/projects/aires/figures/Event_1262_animation.gif',
    #     title_prefix='JJA PlaSim',
    #     fps=2,  # 2 frames per second
    #     dpi=100
    # )
    
    # For Event 3901
    # gif_path = create_event_animation_gif(
    #     3901,
    #     ano_stats_jja,
    #     save_path='/home/zhixingliu/projects/aires/figures/Event_3901_animation.gif',
    #     title_prefix='JJA PlaSim',
    #     fps=2,
    #     dpi=100
    # )
    pass

