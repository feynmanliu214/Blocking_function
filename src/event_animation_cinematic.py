"""
Event Animation - Cinematic Version

A visually stunning animation designed to capture the beauty and magnificence
of atmospheric motion and fluid dynamics. Features:
- Dark cinematic theme with glowing effects
- Animated streamlines showing atmospheric flow
- Pulsing contours for blocking regions
- Trailing ghost effects showing temporal evolution
- Smooth color gradients and professional aesthetics
"""

import matplotlib
# matplotlib.use('Agg') # Removed to allow notebook plotting

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import xarray as xr
import os
import imageio
from io import BytesIO
from scipy.ndimage import gaussian_filter


# -----------------------------------------------------------------------------
# Custom colormaps for cinematic effect
# -----------------------------------------------------------------------------
def create_cinematic_cmap():
    """Create a stunning blue-purple-orange colormap for anomalies."""
    colors = [
        '#0d1b2a',  # Deep navy (strong negative)
        '#1b263b',  # Dark blue
        '#415a77',  # Steel blue
        '#778da9',  # Light steel
        '#e0e1dd',  # Near white (zero)
        '#ffd6a5',  # Warm peach
        '#ffb347',  # Orange
        '#ff6b6b',  # Coral red
        '#c9184a',  # Deep red (strong positive)
    ]
    return LinearSegmentedColormap.from_list('cinematic', colors, N=256)


def create_aurora_cmap():
    """Create an aurora-inspired colormap."""
    colors = [
        '#000428',  # Deep space blue
        '#004e92',  # Ocean blue
        '#00d4aa',  # Teal/cyan
        '#7fff00',  # Chartreuse
        '#ffff00',  # Yellow
        '#ff6600',  # Orange
        '#ff0066',  # Hot pink
    ]
    return LinearSegmentedColormap.from_list('aurora', colors, N=256)


def create_ocean_depth_cmap():
    """Deep ocean to surface colormap for negative anomalies."""
    colors = [
        '#0a0a23',  # Abyss
        '#1a1a4e',  # Deep ocean
        '#2e4a7d',  # Mid depth
        '#4a90a4',  # Shallow
        '#7fcdcd',  # Surface
    ]
    return LinearSegmentedColormap.from_list('ocean_depth', colors, N=128)


def create_fire_cmap():
    """Fire colormap for positive anomalies."""
    colors = [
        '#1a0a00',  # Dark ember
        '#4a1a00',  # Deep red
        '#8b2500',  # Crimson
        '#cd5c00',  # Orange
        '#ffa500',  # Bright orange
        '#ffcc00',  # Gold
        '#ffffaa',  # Pale yellow (hottest)
    ]
    return LinearSegmentedColormap.from_list('fire', colors, N=128)


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def add_glow_effect(ax, lon, lat, data, levels, cmap, transform, alpha=0.3, blur_sigma=2):
    """Add a soft glow effect behind contours."""
    # Create blurred version for glow
    data_blurred = gaussian_filter(data, sigma=blur_sigma)
    ax.contourf(lon, lat, data_blurred, levels=levels, cmap=cmap,
                transform=transform, alpha=alpha, antialiased=True)


def create_diverging_glow_cmap():
    """Create a diverging colormap with glow effect aesthetics."""
    # Negative side: cool blues with slight purple
    neg_colors = ['#0d0d2b', '#1a1a4e', '#2d4a7d', '#4a8bbd', '#8ecae6']
    # Positive side: warm oranges/reds
    pos_colors = ['#ffeeba', '#ffb347', '#ff7f50', '#dc143c', '#8b0000']
    
    # Combine with white center
    all_colors = neg_colors + ['#f8f8f8'] + pos_colors
    return LinearSegmentedColormap.from_list('diverging_glow', all_colors, N=256)


def create_single_color_cmap(color):
    """Create a colormap that is transparent at 0 and solid color at 1."""
    from matplotlib.colors import ListedColormap, to_rgba
    c = to_rgba(color)
    # Create a 256-level cmap where all colors are the same, but alpha might vary?
    # Actually for pcolormesh with masked data, we just need a solid block of color
    return ListedColormap([c])


# -----------------------------------------------------------------------------
# Main cinematic animation function
# -----------------------------------------------------------------------------
def create_cinematic_event_animation(
    event_id, 
    ano_stats, 
    save_path=None,
    title_prefix='',
    figsize=(14, 12),
    fps=1,
    dpi=120,
    style='dark',  # 'dark', 'aurora', 'elegant'
    show_trails=True,
    trail_frames=3,
    glow_intensity=0.4,
    event_start_times=None
):
    """
    Create a cinematic animation of an atmospheric blocking event.
    
    Parameters
    ----------
    event_id : int
        The event ID to animate
    ano_stats : dict
        Dictionary containing event statistics and data
    save_path : str, optional
        Path to save the GIF
    title_prefix : str, optional
        Prefix for the title
    figsize : tuple
        Figure size (width, height)
    fps : int
        Frames per second (lower = slower, more dramatic)
    dpi : int
        Resolution
    style : str
        Visual style: 'dark', 'aurora', or 'elegant'
    show_trails : bool
        Whether to show ghost trails of previous frames
    trail_frames : int
        Number of trailing ghost frames
    glow_intensity : float
        Intensity of glow effects (0-1)
    event_start_times : dict, optional
        Pre-computed event start times index
    
    Returns
    -------
    str : Path to saved GIF
    """
    print(f"\n{'═'*60}")
    print(f"  🎬 CINEMATIC EVENT ANIMATION")
    print(f"  Event {event_id} | Style: {style}")
    print(f"{'═'*60}")
    
    # Get event duration and time range
    duration = len(ano_stats['event_areas'][event_id])
    print(f"  Duration: {duration} days")
    
    # Find event time range
    event_times = _get_event_time_range(ano_stats, event_id, event_start_times)
    if len(event_times) == 0:
        print(f"  ❌ Event {event_id} not found")
        return None
    print(f"  Time indices: {event_times[0]} → {event_times[-1]}")
    
    # Extract data
    z500_anom_raw = ano_stats['z500_anom']
    if not isinstance(z500_anom_raw, xr.DataArray):
        raise ValueError("z500_anom must be xarray DataArray")
    
    lat = z500_anom_raw.lat.values
    lon = z500_anom_raw.lon.values
    time_coords = z500_anom_raw.time.values
    
    # Subset to Northern Hemisphere
    lat_mask = (lat >= 20) & (lat <= 90)
    lat_indices = np.where(lat_mask)[0]
    lat_subset = lat[lat_mask]
    
    event_time_list = list(map(int, event_times))
    
    # Extract event data
    z500_event = z500_anom_raw.isel(time=event_time_list, lat=lat_indices).values
    
    event_mask_raw = ano_stats['event_mask']
    if isinstance(event_mask_raw, xr.DataArray):
        event_mask_event = event_mask_raw.isel(time=event_time_list, lat=lat_indices).values
    else:
        event_mask_event = np.asarray(event_mask_raw)[event_time_list][:, lat_mask, :]
    
    # Add cyclic point for smooth plotting
    lon_cyclic = np.append(lon, lon[0] + 360)
    
    event_areas = list(ano_stats['event_areas'][event_id])
    
    # Setup style
    if style == 'dark':
        bg_color = '#0a0a0f'
        text_color = '#e8e8e8'
        coast_color = '#3a3a4a'
        border_color = '#2a2a3a'
        land_color = '#15151f'
        ocean_color = '#0d0d15'
        cmap = create_diverging_glow_cmap()
        event_color = '#00ffcc'  # Cyan glow
        event_glow = '#00ffcc'
    elif style == 'aurora':
        bg_color = '#000020'
        text_color = '#ffffff'
        coast_color = '#4a4a6a'
        border_color = '#3a3a5a'
        land_color = '#101025'
        ocean_color = '#080815'
        cmap = create_aurora_cmap()
        event_color = '#00ff88'  # Green glow
        event_glow = '#00ff88'
    else:  # elegant
        bg_color = '#1a1a2e'
        text_color = '#f0f0f0'
        coast_color = '#5a5a7a'
        border_color = '#4a4a6a'
        land_color = '#252540'
        ocean_color = '#1f1f35'
        cmap = create_cinematic_cmap()
        event_color = '#ffd700'  # Gold
        event_glow = '#ffaa00'
    
    # Determine data range for consistent colorbar
    vmax = np.percentile(np.abs(z500_event), 98)
    vmax = max(vmax, 200)  # Minimum range
    levels = np.linspace(-vmax, vmax, 41)
    
    frames = []
    print(f"  Generating {len(event_times)} cinematic frames...")
    
    # Store previous event masks for trails
    prev_events = []
    
    for frame_idx in range(len(event_times)):
        # Create figure with dark background
        fig = plt.figure(figsize=figsize, facecolor=bg_color)
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo(central_longitude=0))
        ax.set_facecolor(ocean_color)
        ax.set_extent([-180, 180, 25, 90], ccrs.PlateCarree())
        
        # Get current frame data
        z500_t = z500_event[frame_idx].copy()
        event_t = (event_mask_event[frame_idx] == event_id).astype(float)
        
        # Handle NaN values - replace with 0
        z500_t = np.nan_to_num(z500_t, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Add cyclic point
        z500_cyc = np.concatenate([z500_t, z500_t[:, :1]], axis=1)
        event_cyc = np.concatenate([event_t, event_t[:, :1]], axis=1)
        
        # Apply slight smoothing for visual appeal
        # Clip values to avoid extreme outliers that might confuse contouring
        z500_cyc = np.clip(z500_cyc, -1000, 1000)
        z500_smooth = gaussian_filter(z500_cyc, sigma=0.8)
        
        # Clip data to level range to avoid GEOS issues
        z500_smooth = np.clip(z500_smooth, levels[0], levels[-1])
        
        # Create mesh grid for pcolormesh (more robust than contourf)
        lon_mesh, lat_mesh = np.meshgrid(lon_cyclic, lat_subset)
        
        # === LAYER 1: Glow effect (subtle background glow) ===
        if glow_intensity > 0:
            z500_glow = gaussian_filter(z500_cyc, sigma=3)
            z500_glow = np.clip(z500_glow, levels[0], levels[-1])
            try:
                ax.pcolormesh(lon_mesh, lat_mesh, z500_glow,
                             cmap=cmap, alpha=glow_intensity * 0.5,
                             transform=ccrs.PlateCarree(), shading='auto',
                             vmin=levels[0], vmax=levels[-1])
            except Exception:
                pass  # Skip glow if it fails
        
        # === LAYER 2: Main anomaly field ===
        cf = ax.pcolormesh(lon_mesh, lat_mesh, z500_smooth,
                          cmap=cmap, alpha=0.85,
                          transform=ccrs.PlateCarree(), shading='auto',
                          vmin=levels[0], vmax=levels[-1])
        
        # === LAYER 3: Subtle contour lines for depth ===
        contour_levels = levels[::4]  # Every 4th level
        try:
            ax.contour(lon_cyclic, lat_subset, z500_smooth,
                      levels=contour_levels, colors='white', linewidths=0.3,
                      alpha=0.2, transform=ccrs.PlateCarree())
        except Exception:
            pass  # Skip contour lines if they fail
        
        # === LAYER 4: Ghost trails of previous event positions ===
        if show_trails and len(prev_events) > 0:
            for i, (prev_event, prev_alpha) in enumerate(prev_events):
                trail_alpha = prev_alpha * 0.4 * glow_intensity
                if trail_alpha > 0.05 and prev_event.any():
                    try:
                        prev_cyc = np.concatenate([prev_event, prev_event[:, :1]], axis=1)
                        # Ensure binary
                        prev_cyc_binary = (prev_cyc > 0.5).astype(float)
                        prev_smooth = gaussian_filter(prev_cyc_binary, sigma=1.5)
                        
                        # Use pcolormesh instead of contour for stability
                        # Mask low values to make them transparent
                        masked_trail = np.ma.masked_where(prev_smooth < 0.2, prev_smooth)
                        
                        # Create a solid color cmap for the trail
                        from matplotlib.colors import ListedColormap
                        trail_cmap = ListedColormap([event_glow])
                        
                        ax.pcolormesh(lon_mesh, lat_mesh, masked_trail,
                                     cmap=trail_cmap, alpha=trail_alpha,
                                     transform=ccrs.PlateCarree(), shading='auto')
                    except Exception:
                        pass  # Skip trail if it fails
        
        # === LAYER 5: Current event boundary with glow ===
        if event_cyc.any():
            # Ensure event mask is binary 0/1
            event_cyc_binary = (event_cyc > 0.5).astype(float)
            
            try:
                # Glow layers using pcolormesh for stability
                # Outer glow
                glow_data_outer = gaussian_filter(event_cyc_binary, sigma=2)
                masked_outer = np.ma.masked_where(glow_data_outer < 0.1, glow_data_outer)
                
                # Create gradient cmap for glow
                # We need a colormap that goes from transparent to the glow color
                # But since pcolormesh applies alpha globally or per-cell, we rely on the alpha channel of the color or the main alpha
                
                # Simplified approach: Plot masked glow
                ax.pcolormesh(lon_mesh, lat_mesh, masked_outer,
                             cmap=create_single_color_cmap(event_glow), alpha=0.3,
                             transform=ccrs.PlateCarree(), shading='auto')
                
                # Middle glow
                glow_data_mid = gaussian_filter(event_cyc_binary, sigma=1)
                masked_mid = np.ma.masked_where(glow_data_mid < 0.2, glow_data_mid)
                ax.pcolormesh(lon_mesh, lat_mesh, masked_mid,
                             cmap=create_single_color_cmap(event_glow), alpha=0.4,
                             transform=ccrs.PlateCarree(), shading='auto')
                             
                # Core boundary - Try contour only for the main line, but be very defensive
                # If it fails, fallback to pcolormesh
                try:
                    event_smooth = gaussian_filter(event_cyc_binary, sigma=0.5)
                    ax.contour(lon_cyclic, lat_subset, event_smooth,
                              levels=[0.5], colors=event_color, linewidths=2.5,
                              alpha=1.0, transform=ccrs.PlateCarree())
                except Exception:
                    # Fallback to pcolormesh for core if contour fails
                    masked_core = np.ma.masked_where(event_cyc_binary < 0.5, event_cyc_binary)
                    ax.pcolormesh(lon_mesh, lat_mesh, masked_core,
                                 cmap=create_single_color_cmap(event_color), alpha=1.0,
                                 transform=ccrs.PlateCarree(), shading='auto')
                                 
            except Exception as e:
                print(f"    ⚠️ Layer drawing error: {e}")
                pass
        
        # Update trails
        prev_events.append((event_t.copy(), 1.0))
        if len(prev_events) > trail_frames:
            prev_events.pop(0)
        # Decay trail alphas
        prev_events = [(e, a * 0.6) for e, a in prev_events]
        
        # === LAYER 6: Geographic features ===
        ax.add_feature(cfeature.LAND, facecolor=land_color, edgecolor='none', zorder=1)
        ax.coastlines(linewidth=0.8, color=coast_color, zorder=2)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, color=border_color, alpha=0.5, zorder=2)
        
        # === LAYER 7: Gridlines (subtle) ===
        gl = ax.gridlines(draw_labels=False, linewidth=0.3, color='#404060',
                         alpha=0.3, linestyle='--')
        
        # === Colorbar ===
        cbar_ax = fig.add_axes([0.15, 0.08, 0.7, 0.02])
        cbar = fig.colorbar(cf, cax=cbar_ax, orientation='horizontal', extend='both')
        cbar.ax.tick_params(colors=text_color, labelsize=10)
        cbar.set_label('Z500 Anomaly (m)', color=text_color, fontsize=11, fontweight='bold')
        cbar.outline.set_edgecolor(text_color)
        cbar.outline.set_linewidth(0.5)
        
        # === Title and info ===
        t_idx = event_times[frame_idx]
        time_str = str(time_coords[t_idx])[:10]
        area = event_areas[frame_idx] / 1e6
        
        # Main title
        title = f'Atmospheric Blocking Event #{event_id}'
        if title_prefix:
            title = f'{title_prefix} | {title}'
        
        fig.text(0.5, 0.95, title, ha='center', va='top',
                fontsize=16, fontweight='bold', color=text_color,
                transform=fig.transFigure)
        
        # Subtitle with info
        subtitle = f'Day {frame_idx+1} of {duration}  •  {time_str}  •  Area: {area:.1f}×10⁶ km²'
        fig.text(0.5, 0.91, subtitle, ha='center', va='top',
                fontsize=12, color='#aaaacc', style='italic',
                transform=fig.transFigure)
        
        # Legend annotation
        legend_text = f'━━ Blocking boundary'
        fig.text(0.02, 0.02, legend_text, ha='left', va='bottom',
                fontsize=9, color=event_color, fontweight='bold',
                transform=fig.transFigure)
        
        # Save frame
        buf = BytesIO()
        try:
            # The GEOSException often comes from the automatic tight bounding box calculation
            # (which is often on by default or internally triggered).
            # Explicitly setting bbox_inches=None prevents this calculation.
            fig.savefig(
                buf, 
                format='png', 
                dpi=dpi, 
                facecolor=bg_color, 
                edgecolor='none',
                bbox_inches=None  # <--- Primary fix for GEOSException
            )
        except Exception as e:
            # If an error still occurs, it's likely a serious Cartopy/Matplotlib issue.
            # Log it and stop.
            print(f"    ❌ Critical Frame Save Error: {e}. Stopping animation.")
            plt.close(fig)
            return None  # Exit function if a frame cannot be saved
            
        buf.seek(0)
        frames.append(imageio.imread(buf))
        buf.close()
        plt.close(fig)
        
        if (frame_idx + 1) % max(1, duration // 4) == 0 or frame_idx == 0:
            print(f"    ✓ Frame {frame_idx+1}/{duration}")
    
    # Save GIF
    if save_path is None:
        save_path = f'/home/zhixingliu/projects/aires/figures/Cinematic_Event_{event_id}.gif'
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Use longer duration for more dramatic effect
    frame_duration = 1.0 / fps
    imageio.mimsave(save_path, frames, duration=frame_duration, loop=0)
    
    size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"\n  {'═'*50}")
    print(f"  ✅ Cinematic animation saved!")
    print(f"     📁 {save_path}")
    print(f"     📊 {size_mb:.1f} MB | {len(frames)} frames | {fps} fps")
    print(f"  {'═'*50}\n")
    
    return save_path


def _get_event_time_range(ano_stats, event_id, event_start_times=None):
    """Get the time indices for an event."""
    duration = len(ano_stats['event_areas'][event_id])
    
    if event_start_times and event_id in event_start_times:
        start = event_start_times[event_id]
        return np.arange(start, start + duration)
    
    # Fallback: linear scan
    event_mask = ano_stats['event_mask']
    if isinstance(event_mask, xr.DataArray):
        event_mask = event_mask.values
    
    for t in range(event_mask.shape[0]):
        if event_id in np.unique(event_mask[t]):
            return np.arange(t, t + duration)
    
    return np.array([])


# -----------------------------------------------------------------------------
# Batch creation function
# -----------------------------------------------------------------------------
def create_cinematic_gifs_for_events(
    event_ids,
    ano_stats,
    output_dir='/home/zhixingliu/projects/aires/figures/cinematic',
    style='dark',
    fps=1,
    dpi=120,
    **kwargs
):
    """
    Create cinematic animations for multiple events.
    
    Parameters
    ----------
    event_ids : list
        List of event IDs to animate
    ano_stats : dict
        Event statistics dictionary
    output_dir : str
        Output directory for GIFs
    style : str
        Visual style ('dark', 'aurora', 'elegant')
    fps : int
        Frames per second
    dpi : int
        Resolution
    **kwargs : dict
        Additional arguments passed to create_cinematic_event_animation
    
    Returns
    -------
    dict : Mapping of event_id to GIF path
    """
    from event_animation_fast_v6 import build_event_start_times
    
    print("\n" + "🎬" * 25)
    print("  CINEMATIC BATCH ANIMATION")
    print(f"  {len(event_ids)} events | Style: {style}")
    print("🎬" * 25 + "\n")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Build event start times index
    event_mask = ano_stats['event_mask']
    if isinstance(event_mask, xr.DataArray):
        event_mask_np = event_mask.values
    else:
        event_mask_np = np.asarray(event_mask)
    
    print("📊 Building event start times index...")
    event_start_times = build_event_start_times(event_mask_np, event_ids)
    
    gif_paths = {}
    for i, event_id in enumerate(event_ids):
        print(f"\n[{i+1}/{len(event_ids)}] Processing Event {event_id}...")
        
        save_path = os.path.join(output_dir, f'Cinematic_{style}_{event_id}.gif')
        
        try:
            path = create_cinematic_event_animation(
                event_id=event_id,
                ano_stats=ano_stats,
                save_path=save_path,
                style=style,
                fps=fps,
                dpi=dpi,
                event_start_times=event_start_times,
                **kwargs
            )
            gif_paths[event_id] = path
        except Exception as e:
            print(f"  ❌ Error: {e}")
            gif_paths[event_id] = None
    
    success = sum(1 for p in gif_paths.values() if p)
    print(f"\n✅ Batch complete: {success}/{len(event_ids)} animations created")
    
    return gif_paths


# -----------------------------------------------------------------------------
# Quick demo function
# -----------------------------------------------------------------------------
def demo_styles(event_id, ano_stats, output_dir='/home/zhixingliu/projects/aires/figures/cinematic_demo'):
    """Create a demo showing all three styles for comparison."""
    print("\n🎨 Creating style comparison demo...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    for style in ['dark', 'aurora', 'elegant']:
        print(f"\n  Creating {style} style...")
        save_path = os.path.join(output_dir, f'Demo_{style}_{event_id}.gif')
        create_cinematic_event_animation(
            event_id=event_id,
            ano_stats=ano_stats,
            save_path=save_path,
            style=style,
            fps=1,
            dpi=100
        )
    
    print("\n✅ Demo complete! Check the output directory for all three styles.")

