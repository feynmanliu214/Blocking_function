"""
Event Animation - Fast Version v4

Designed for ERA5-sized grids:
1. Builds event start index once (O(T)) and reuses it
2. Extracts ONLY the event's timesteps (no 14GB loads)
3. Subsets to 25–90°N and downsamples if needed
4. Uses imageio for quick GIF writing
"""

import matplotlib
# matplotlib.use('Agg') # Removed to allow notebook plotting

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import xarray as xr
import os
import imageio
from io import BytesIO
from multiprocessing import cpu_count


# -----------------------------------------------------------------------------
# Helper: build start-time index once
# -----------------------------------------------------------------------------
def build_event_start_times_index(event_mask, event_ids, n_workers=None):
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
    index = {}
    for eid, start in build_event_start_times_index(event_mask, event_ids):
        index[eid] = start
    print(f"  ✓ Index built: {len(index)} events indexed")
    return index


# -----------------------------------------------------------------------------
# Helper: get time range for event
# -----------------------------------------------------------------------------
def get_event_time_range(ano_stats, event_id, event_start_times=None):
    duration = len(ano_stats['event_areas'][event_id])
    if event_start_times and event_id in event_start_times:
        start = event_start_times[event_id]
        return np.arange(start, start + duration)

    # fallback linear scan (should rarely happen)
    event_mask = ano_stats['event_mask']
    if isinstance(event_mask, xr.DataArray):
        event_mask = event_mask.values
    for t in range(event_mask.shape[0]):
        if event_id in np.unique(event_mask[t]):
            return np.arange(t, t + duration)
    return np.array([])


# -----------------------------------------------------------------------------
# Core animation function
# -----------------------------------------------------------------------------
def create_event_animation_gif_fast(event_id, ano_stats, save_path=None,
                                    title_prefix='', figsize=(12, 10),
                                    fps=1, dpi=100, max_lat_points=90, max_lon_points=360,
                                    event_start_times=None,
                                    # New parameters for P_block and centroid panels
                                    show_diagnostics=False,
                                    region_lon_min=30.0, region_lon_max=100.0,
                                    region_lat_min=55.0, region_lat_max=75.0):
    """
    Create animated GIF for a blocking event.
    
    Parameters
    ----------
    event_id : int
        Event ID to animate
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
    """
    print(f"\n  Creating animation for Event {event_id} (fast v4)...")
    duration = len(ano_stats['event_areas'][event_id])
    print(f"  Duration: {duration} days")

    event_times = get_event_time_range(ano_stats, event_id, event_start_times)
    if len(event_times) == 0:
        print(f"  Error: Event {event_id} not found")
        return None
    print(f"  Time indices: {event_times[0]} to {event_times[-1]}")

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
    print(f"  Extracting ONLY {len(event_time_list)} timesteps × {len(lat_subset)} lats")

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

    event_areas = list(ano_stats['event_areas'][event_id])
    
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
        event_t = (event_mask_ds[frame_idx] == event_id).astype(float)
        blocked_t = blocked_mask_ds[frame_idx]

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
        title = f'Event {event_id}'
        if title_prefix:
            title += f' - {title_prefix}'
        title += f'\nDay {frame_idx+1}/{duration} | {time_str} | Area: {area:.2f}×10⁶ km²'
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
        # For diagnostics: rely on GridSpec hspace/wspace for spacing (no tight_layout to avoid jitter)
        # For single map: bbox_inches='tight' is fine
        if show_diagnostics:
            fig.savefig(buf, format='png', dpi=dpi, facecolor='white')
        else:
            fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', facecolor='white')
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


# -----------------------------------------------------------------------------
# Driver: create GIFs for requested durations
# -----------------------------------------------------------------------------
def create_duration_specific_gifs_fast(event_info, z500_data, method, season,
                                       target_durations,
                                       output_base_dir='/home/zhixingliu/projects/aires/figures',
                                       fps=1, dpi=100, max_lat_points=90, max_lon_points=360):
    print("\n" + "🚀" * 30)
    print("USING FAST VERSION v4: create_duration_specific_gifs_fast")
    print("  ✓ Smart event indexing once")
    print("  ✓ Minimal data extraction per event")
    print("  ✓ Downsampled plotting grid")
    print("🚀" * 30 + "\n")

    from event_animation import convert_abs_events_to_ano_format

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
