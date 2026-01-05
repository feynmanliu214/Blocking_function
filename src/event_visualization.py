import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

def plot_event_evolution(event_id, ano_stats, figsize=(20, 12), n_snapshots=8, 
                         save_path=None, title_prefix=''):
    """
    Plot the time evolution of a blocking event.
    
    Parameters
    ----------
    event_id : int
        Event ID to visualize
    ano_stats : dict
        Dictionary containing event information from ano_blocking_complete
    figsize : tuple
        Figure size (width, height)
    n_snapshots : int
        Number of snapshots to show (default: 8)
    save_path : str, optional
        Path to save the figure
    title_prefix : str
        Prefix for the title (e.g., 'JJA PlaSim', 'DJF ERA5')
    """
    
    # Extract data
    event_mask = ano_stats['event_mask']  # (time, lat, lon) - full domain
    z500_anom = ano_stats['z500_anom']  # Anomaly field (time, lat, lon)
    blocked_mask = ano_stats['blocked_mask']  # Binary mask (time, lat, lon)
    
    # Get coordinates - all should have same lat/lon now
    lat = z500_anom.lat.values
    lon = z500_anom.lon.values
    
    # Get the time steps where this event occurs
    event_times = np.where((event_mask == event_id).any(axis=(1, 2)))[0]
    
    if len(event_times) == 0:
        print(f"Error: Event {event_id} not found in event_mask")
        return
    
    print(f"Event {event_id} Information:")
    print(f"  Duration: {len(event_times)} days")
    print(f"  Start time index: {event_times[0]}")
    print(f"  End time index: {event_times[-1]}")
    print(f"  Time range: {z500_anom.time.values[event_times[0]]} to {z500_anom.time.values[event_times[-1]]}")
    
    # Get event area information
    event_areas = list(ano_stats['event_areas'][event_id])
    print(f"  Mean area: {np.mean(event_areas)/1e6:.2f} × 10⁶ km²")
    print(f"  Area range: [{np.min(event_areas)/1e6:.2f}, {np.max(event_areas)/1e6:.2f}] × 10⁶ km²")
    
    # Select snapshots evenly spaced through the event
    n_snapshots = min(n_snapshots, len(event_times))
    snapshot_indices = np.linspace(0, len(event_times)-1, n_snapshots, dtype=int)
    snapshot_times = event_times[snapshot_indices]
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    
    for subplot_idx, (snap_pos, t_idx) in enumerate(zip(snapshot_indices, snapshot_times), start=1):
        # Data for this time step
        z500_t = z500_anom.isel(time=t_idx)
        blocked_t = blocked_mask.isel(time=t_idx)
        event_mask_t = (event_mask[t_idx] == event_id)  # (lat, lon) - full domain
        
        # Create subplot with polar stereographic projection
        ax = fig.add_subplot(2, 4, subplot_idx, projection=ccrs.NorthPolarStereo())
        ax.set_extent([-180, 180, 30, 90], ccrs.PlateCarree())
        
        # Plot anomaly field (background)
        levels_anom = np.arange(-300, 301, 30)
        cf = ax.contourf(lon, lat, z500_t.values, 
                         levels=levels_anom, 
                         cmap='RdBu_r', 
                         transform=ccrs.PlateCarree(),
                         alpha=0.6, extend='both')
        
        # Overlay the specific event boundary (full domain)
        if event_mask_t.any():
            ax.contour(lon, lat, event_mask_t.astype(float), 
                      levels=[0.5], 
                      colors='black', 
                      linewidths=3,
                      transform=ccrs.PlateCarree())
        
        # Overlay all blocked regions (lighter contour) using full domain mask
        if blocked_t.values.any():
            ax.contour(lon, lat, blocked_t.values, 
                      levels=[0.5], 
                      colors='orange', 
                      linewidths=1.5,
                      linestyles='--',
                      transform=ccrs.PlateCarree(),
                      alpha=0.5)
        
        # Add coastlines and features
        ax.coastlines(linewidth=0.5, color='black')
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
        
        # Add gridlines
        ax.gridlines(draw_labels=False, linewidth=0.5, 
                     color='gray', alpha=0.3, linestyle='-')
        
        # Title with day number and date
        day_num = snap_pos + 1
        time_str = str(z500_anom.time.values[t_idx])[:10]
        area_day = event_areas[snap_pos] / 1e6
        ax.set_title(f'Day {day_num}/{len(event_times)}\n{time_str}\nArea: {area_day:.2f}×10⁶ km²', 
                    fontsize=11, fontweight='bold')
    
    # Adjust spacing to prevent overlap
    plt.subplots_adjust(left=0.05, right=0.90, top=0.88, bottom=0.08, hspace=0.35, wspace=0.15)
    
    # Add colorbar for anomaly
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(cf, cax=cbar_ax, label='Z500 Anomaly (m)')
    cbar.ax.tick_params(labelsize=10)
    
    # Main title
    title = f'Event {event_id} Time Evolution'
    if title_prefix:
        title += f' - {title_prefix}'
    title += f'\nDuration: {len(event_times)} days | Mean Area: {np.mean(event_areas)/1e6:.2f}×10⁶ km²\n'
    title += 'Black contour = Event boundary | Orange dashed = All blocked regions'
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.96)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✅ Figure saved to: {save_path}")
    
    plt.show()
    
    return fig


# Example usage:
# For Event 1262
# fig = plot_event_evolution(1262, ano_stats_jja, 
#                            save_path='/home/zhixingliu/projects/aires/figures/Event_1262_evolution_JJA.png',
#                            title_prefix='JJA PlaSim')

# For Event 3901
# fig = plot_event_evolution(3901, ano_stats_jja,
#                            save_path='/home/zhixingliu/projects/aires/figures/Event_3901_evolution_JJA.png', 
#                            title_prefix='JJA PlaSim')

