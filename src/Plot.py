import numpy as np
import xarray as xr
from pathlib import Path
from typing import List, Union, Optional
from scipy import ndimage
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import matplotlib.pyplot as plt




def plot_blocking_frequency(blocking_freq: xr.DataArray,
                            plot_type: str = 'auto',
                            figsize: tuple = (16, 8),
                            cmap: str = 'YlOrRd',
                            title: Optional[str] = None,
                            data_source: Optional[str] = None,
                            as_percentage: bool = True,
                            vmax: Optional[float] = 20.0,
                            capped: bool = True,
                            white_threshold: Optional[float] = 0.5,
                            show_stats: bool = True,
                            save_path: Optional[str] = None,
                            dpi: int = 300,
                            min_lat: Optional[float] = None,
                            min_lon: Optional[float] = None,
                            max_lat: Optional[float] = None,
                            max_lon: Optional[float] = None):
    """
    Plot atmospheric blocking frequency on a map.
    
    Creates a publication-quality visualization of blocking frequency data.
    Supports both polar stereographic projection (with cartopy) and simple 
    contour plots (without cartopy).
    
    Parameters
    ----------
    blocking_freq : xarray.DataArray
        2D blocking frequency data with dimensions (lat, lon).
        Values should be between 0 and 1 (fraction of time blocked).
    plot_type : str, optional
        Type of plot to create:
        - 'auto': Try polar projection first, fall back to simple if cartopy unavailable
        - 'polar': Polar stereographic projection (requires cartopy)
        - 'simple': Simple lat/lon contour plot (no cartopy needed)
        Default: 'auto'
    figsize : tuple, optional
        Figure size in inches (width, height). Default: (16, 8)
    cmap : str, optional
        Matplotlib colormap name. Default: 'YlOrRd'
    title : str, optional
        Plot title. If None, uses default title. Default: None
    data_source : str, optional
        Data source to include in the title (e.g., 'ERA5', 'PlaSim').
        If provided and title is None, will be added to default title.
        If title is provided, data_source is ignored. Default: None
    as_percentage : bool, optional
        If True, display values as percentages (multiply by 100). Default: True
    vmax : float, optional
        Maximum value for colorbar when capped=True. If as_percentage=True, this is in percent (default: 20%).
        If as_percentage=False, this is in fraction (default: 0.20).
        Ignored when capped=False. Default: 20.0
    capped : bool, optional
        If True, use vmax as the colorbar maximum (default: 20.0 when as_percentage=True).
        If False, use the data maximum for the colorbar. Default: True
    white_threshold : float, optional
        Threshold below which values are displayed as white (no color). 
        If as_percentage=True, this is in percent (e.g., 0.5 means 0.5%).
        If as_percentage=False, this is in fraction (e.g., 0.005 means 0.5%).
        Set to None or 0 to disable white threshold. Default: 0.5
    show_stats : bool, optional
        If True, show statistics (max, mean) on the plot. Default: True
    save_path : str, optional
        If provided, save figure to this path. Default: None
    dpi : int, optional
        DPI for saved figure. Default: 300
    min_lat : float, optional
        Minimum latitude for bounding box. If all four box parameters are provided,
        draws a rectangular box on the plot. Default: None
    min_lon : float, optional
        Minimum longitude for bounding box. Default: None
    max_lat : float, optional
        Maximum latitude for bounding box. Default: None
    max_lon : float, optional
        Maximum longitude for bounding box. Default: None
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object
    ax : matplotlib.axes.Axes
        The axes object
    
    Examples
    --------
    >>> # Compute blocking frequency
    >>> z500_djf = process_multiple_files(range(1000, 1100), season='DJF')
    >>> blocking_freq = abs_blocking_frequency(z500_djf)
    >>> 
    >>> # Create polar projection plot (auto)
    >>> fig, ax = plot_blocking_frequency(blocking_freq)
    >>> 
    >>> # Create simple plot
    >>> fig, ax = plot_blocking_frequency(blocking_freq, plot_type='simple')
    >>> 
    >>> # Add data source to title
    >>> fig, ax = plot_blocking_frequency(blocking_freq, data_source='ERA5')
    >>> # Title will be: "Atmospheric Blocking Frequency - ERA5"
    >>> 
    >>> # Customize and save
    >>> fig, ax = plot_blocking_frequency(
    ...     blocking_freq,
    ...     title='Winter Blocking Frequency (1000-1099)',
    ...     cmap='RdYlBu_r',
    ...     save_path='blocking_freq_DJF.png'
    ... )
    
    Notes
    -----
    - Cartopy must be installed for polar projection plots
    - Simple plots work with matplotlib only
    - Values are typically in the range 0-0.05 (0-5%)
    """
    # Add cyclic point if not present to avoid white line at 0 longitude
    # Check if the data wraps around (360 degrees)
    lons = blocking_freq.lon.values
    if len(lons) > 1:
        lon_range = lons[-1] - lons[0]
        lon_spacing = lons[1] - lons[0]
        expected_range = 360.0 - lon_spacing
        
        # If range is close to 360-spacing (e.g. 0 to 357.5), we need to add cyclic point
        if abs(lon_range - expected_range) < lon_spacing * 0.5:
            # Create cyclic point (copy of first point)
            cyclic_data = blocking_freq.isel(lon=0)
            new_lon = lons[-1] + lon_spacing
            cyclic_data = cyclic_data.assign_coords(lon=new_lon)
            blocking_freq = xr.concat([blocking_freq, cyclic_data], dim='lon')
    # Check plot type and cartopy availability
    if plot_type == 'auto':
        try:
            import cartopy.crs as ccrs
            plot_type = 'polar'
        except ImportError:
            plot_type = 'simple'
            print("⚠️  Cartopy not available, using simple plot")
    
    # Determine vmax based on capped parameter
    if not capped:
        # Use data maximum
        data_max = float(blocking_freq.max().values)
        if as_percentage:
            vmax_plot = data_max * 100
        else:
            vmax_plot = data_max
    else:
        # Use provided vmax (default 20.0)
        vmax_plot = vmax
    
    # Construct title with data source if provided
    if title is None and data_source is not None:
        title = f'Atmospheric Blocking Frequency - {data_source}'
    elif title is None:
        # Will use default title in helper functions
        pass
    
    # Package bounding box parameters
    bbox = None
    if all(x is not None for x in [min_lat, min_lon, max_lat, max_lon]):
        bbox = (min_lat, min_lon, max_lat, max_lon)
    
    if plot_type == 'polar':
        try:
            import cartopy.crs as ccrs
            return _plot_blocking_frequency_polar(
                blocking_freq, figsize, cmap, title, as_percentage, 
                vmax_plot, white_threshold, show_stats, save_path, dpi, bbox
            )
        except ImportError:
            raise ImportError(
                "Cartopy is required for polar plots. "
                "Install with: conda install -c conda-forge cartopy\n"
                "Or use plot_type='simple'"
            )
    elif plot_type == 'simple':
        return _plot_blocking_frequency_simple(
            blocking_freq, figsize, cmap, title, as_percentage,
            vmax_plot, white_threshold, show_stats, save_path, dpi, bbox
        )
    else:
        raise ValueError(f"Invalid plot_type: {plot_type}. Use 'auto', 'polar', or 'simple'")


def _plot_blocking_frequency_polar(blocking_freq, figsize, cmap, title, 
                                   as_percentage, vmax, white_threshold, show_stats, save_path, dpi, bbox=None):
    """Internal function for polar stereographic plot with cartopy."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.colors as mcolors
    
    # Create figure with cartopy projection
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo(central_longitude=0))
    
    # Set extent to show Northern Hemisphere mid-to-high latitudes
    ax.set_extent([-180, 180, 30, 90], crs=ccrs.PlateCarree())
    
    # Get coordinates
    lons = blocking_freq.lon.values
    lats = blocking_freq.lat.values
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    # Get data values
    data = blocking_freq.values.copy()
    if as_percentage:
        data = data * 100
        units = '%'
        # white_threshold is expected in percentage when as_percentage=True
        threshold = white_threshold if white_threshold is not None else 0.0
    else:
        units = 'fraction'
        # white_threshold is expected as percentage, convert to fraction
        if white_threshold is not None:
            threshold = white_threshold / 100.0
        else:
            threshold = 0.0
    
    # Determine colorbar maximum
    # vmax is already converted to the correct units (percentage or fraction) in main function
    vmax_plot = vmax
    
    # Create custom colormap with white for values below threshold (if threshold > 0)
    cmap_obj = plt.get_cmap(cmap)
    if threshold > 0:
        cmap_obj = cmap_obj.copy()
        cmap_obj.set_under('white')  # Set color for values below vmin to white
        vmin_plot = threshold
    else:
        vmin_plot = 0
    
    # Create filled contour plot
    levels = np.linspace(vmin_plot, vmax_plot, 21)
    cf = ax.contourf(lon_grid, lat_grid, data,
                     levels=levels,
                     cmap=cmap_obj,
                     transform=ccrs.PlateCarree(),
                     extend='both' if threshold > 0 else 'max',
                     vmin=vmin_plot, vmax=vmax_plot)
    
    # Add contour lines for clarity
    cs = ax.contour(lon_grid, lat_grid, data,
                    levels=levels[::2],  # Every other level
                    colors='black',
                    linewidths=0.5,
                    alpha=0.3,
                    transform=ccrs.PlateCarree())
    
    # Add text annotation about white regions (only if threshold > 0)
    if threshold > 0:
        note_text = f'White regions: 0–{threshold:.1f}{units}'
        ax.text(0.98, 0.02, note_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='gray'))
    
    # Add geographic features
    ax.coastlines(linewidth=0.8, color='black')
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, alpha=0.5)
    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray',
                      alpha=0.5, linestyle='--')
    
    # Add colorbar
    if threshold > 0:
        # For colorbar with white region, we need to show range [0, vmax]
        # but the actual contour levels are [threshold, vmax]
        import matplotlib as mpl
        
        # Create a new axis for custom colorbar
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("bottom", size="5%", pad=0.5, axes_class=plt.Axes)
        
        # Create colormap that spans [0, vmax] with white for [0, threshold]
        n_white = int(256 * threshold / vmax_plot)
        n_colors = 256 - n_white
        
        # White colors
        white_part = np.ones((n_white, 4))
        # Original colormap colors
        orig_cmap = plt.get_cmap(cmap)
        color_part = orig_cmap(np.linspace(0, 1, n_colors))
        
        # Combine
        from matplotlib.colors import ListedColormap
        full_colors = np.vstack([white_part, color_part])
        display_cmap = ListedColormap(full_colors)
        
        # Create colorbar with full range
        norm = mpl.colors.Normalize(vmin=0, vmax=vmax_plot)
        cbar = mpl.colorbar.ColorbarBase(cax, cmap=display_cmap, norm=norm,
                                         orientation='horizontal')
        cbar.set_label(f'Blocking Frequency ({units})', fontsize=10, fontweight='bold')
        cax.tick_params(labelsize=8)
    else:
        cbar = plt.colorbar(cf, ax=ax, orientation='horizontal',
                           pad=0.05, shrink=0.4, aspect=20, extend='max')
        cbar.set_label(f'Blocking Frequency ({units})', fontsize=10, fontweight='bold')
        cbar.ax.tick_params(labelsize=8)
    
    # Add title using suptitle to avoid layout issues with custom colorbar
    if title is None:
        title = 'Atmospheric Blocking Frequency'
    if threshold > 0:
        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    else:
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    # Add statistics text box
    if show_stats:
        stats_text = f'Max: {data.max():.2f}{units}\nMean: {data.mean():.2f}{units}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=props)
    
    # Draw bounding box if provided (with proper spherical representation)
    if bbox is not None:
        min_lat, min_lon, max_lat, max_lon = bbox
        # Interpolate points along each edge to create smooth curves on the projection
        n_points = 50  # Number of points along each edge for smooth curves
        
        # Bottom edge: constant min_lat, varying longitude
        bottom_lons = np.linspace(min_lon, max_lon, n_points)
        bottom_lats = np.full(n_points, min_lat)
        
        # Right edge: constant max_lon, varying latitude
        right_lats = np.linspace(min_lat, max_lat, n_points)
        right_lons = np.full(n_points, max_lon)
        
        # Top edge: constant max_lat, varying longitude (reversed)
        top_lons = np.linspace(max_lon, min_lon, n_points)
        top_lats = np.full(n_points, max_lat)
        
        # Left edge: constant min_lon, varying latitude (reversed)
        left_lats = np.linspace(max_lat, min_lat, n_points)
        left_lons = np.full(n_points, min_lon)
        
        # Concatenate all edges to form closed polygon
        box_lons = np.concatenate([bottom_lons, right_lons, top_lons, left_lons])
        box_lats = np.concatenate([bottom_lats, right_lats, top_lats, left_lats])
        
        ax.plot(box_lons, box_lats, color='blue', linewidth=2, linestyle='-',
                transform=ccrs.PlateCarree(), zorder=10)
    
    # Adjust layout
    if threshold > 0:
        plt.subplots_adjust(top=0.92, bottom=0.15)  # Leave space for title (higher) and colorbar
    else:
        plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"✅ Figure saved to: {save_path}")
    
    plt.show()
    return fig, ax


def _plot_blocking_frequency_simple(blocking_freq, figsize, cmap, title,
                                    as_percentage, vmax, white_threshold, show_stats, save_path, dpi, bbox=None):
    """Internal function for simple contour plot without cartopy."""
    import matplotlib.colors as mcolors
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get coordinates
    lons = blocking_freq.lon.values
    lats = blocking_freq.lat.values
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    # Get data values
    data = blocking_freq.values.copy()
    if as_percentage:
        data = data * 100
        units = '%'
        # white_threshold is expected in percentage when as_percentage=True
        threshold = white_threshold if white_threshold is not None else 0.0
    else:
        units = 'fraction'
        # white_threshold is expected as percentage, convert to fraction
        if white_threshold is not None:
            threshold = white_threshold / 100.0
        else:
            threshold = 0.0
    
    # Determine colorbar maximum
    # vmax is already converted to the correct units (percentage or fraction) in main function
    vmax_plot = vmax
    
    # Create custom colormap with white for values below threshold (if threshold > 0)
    cmap_obj = plt.get_cmap(cmap)
    if threshold > 0:
        cmap_obj = cmap_obj.copy()
        cmap_obj.set_under('white')  # Set color for values below vmin to white
        vmin_plot = threshold
    else:
        vmin_plot = 0
    
    # Create filled contour plot
    levels = np.linspace(vmin_plot, vmax_plot, 21)
    cf = ax.contourf(lon_grid, lat_grid, data,
                     levels=levels, cmap=cmap_obj, extend='both' if threshold > 0 else 'max',
                     vmin=vmin_plot, vmax=vmax_plot)
    
    # Add contour lines
    cs = ax.contour(lon_grid, lat_grid, data,
                    levels=levels[::2], colors='black',
                    linewidths=0.5, alpha=0.4)
    
    # Add text annotation about white regions (only if threshold > 0)
    if threshold > 0:
        note_text = f'White regions: 0–{threshold:.1f}{units}'
        ax.text(0.98, 0.02, note_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='gray'))
    
    # Labels and formatting
    ax.set_xlabel('Longitude (°E)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Latitude (°N)', fontsize=12, fontweight='bold')
    
    # Add title
    if title is None:
        title = 'Atmospheric Blocking Frequency'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    
    # Add colorbar
    if threshold > 0:
        # For colorbar with white region, we need to show range [0, vmax]
        # but the actual contour levels are [threshold, vmax]
        import matplotlib as mpl
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        
        # Create a new axis for custom colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        
        # Create colormap that spans [0, vmax] with white for [0, threshold]
        n_white = int(256 * threshold / vmax_plot)
        n_colors = 256 - n_white
        
        # White colors
        white_part = np.ones((n_white, 4))
        # Original colormap colors
        orig_cmap = plt.get_cmap(cmap)
        color_part = orig_cmap(np.linspace(0, 1, n_colors))
        
        # Combine
        from matplotlib.colors import ListedColormap
        full_colors = np.vstack([white_part, color_part])
        display_cmap = ListedColormap(full_colors)
        
        # Create colorbar with full range
        norm = mpl.colors.Normalize(vmin=0, vmax=vmax_plot)
        cbar = mpl.colorbar.ColorbarBase(cax, cmap=display_cmap, norm=norm,
                                         orientation='vertical')
        cbar.set_label(f'Blocking Frequency ({units})', fontsize=10, fontweight='bold')
        cax.tick_params(labelsize=8)
    else:
        cbar = plt.colorbar(cf, ax=ax, orientation='vertical', pad=0.02, shrink=0.5)
        cbar.set_label(f'Blocking Frequency ({units})', fontsize=10, fontweight='bold')
        cbar.ax.tick_params(labelsize=8)
    
    # Grid
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
    
    # Set aspect ratio for better map-like appearance
    ax.set_aspect('auto')
    
    # Add statistics
    if show_stats:
        stats_text = f'Max: {data.max():.2f}{units}  |  Mean: {data.mean():.2f}{units}'
        ax.text(0.5, 1.02, stats_text, transform=ax.transAxes,
                fontsize=11, ha='center', bbox=dict(boxstyle='round',
                facecolor='wheat', alpha=0.8))
    
    # Draw bounding box if provided
    if bbox is not None:
        min_lat, min_lon, max_lat, max_lon = bbox
        # Create box coordinates (need to close the box by returning to start)
        box_lons = [min_lon, max_lon, max_lon, min_lon, min_lon]
        box_lats = [min_lat, min_lat, max_lat, max_lat, min_lat]
        ax.plot(box_lons, box_lats, color='blue', linewidth=2, linestyle='-', zorder=10)
    
    # Adjust layout
    if threshold > 0:
        plt.subplots_adjust(right=0.88)  # Leave space for custom colorbar on right
    else:
        plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"✅ Figure saved to: {save_path}")
    
    plt.show()
    return fig, ax









