"""
Event Statistics and Visualization Module

This module provides functions for analyzing and visualizing blocking event statistics,
including duration distributions, frequency analysis, and other event characteristics.

Author: Generated for AIRES project
Date: 2025-11-14
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple


def plot_event_duration_histogram(
    event_info: Dict,
    season: str,
    method: Optional[str] = None,
    save_path: Optional[str] = None,
    color: str = 'steelblue',
    figsize: Tuple[int, int] = (10, 6),
    dpi: int = 150,
    show_plot: bool = True
) -> Optional[Dict[str, float]]:
    """
    Create a histogram of blocking event durations with statistics.
    
    Supports both ABS and ANO event_info formats:
    - ABS format: {'event_count': int, 'persistent_events': [{'duration': int, ...}, ...]}
    - ANO format: {'num_events': int, 'event_durations': {event_id: duration, ...}, ...}
    
    Parameters
    ----------
    event_info : dict
        Event information dictionary. Can be either:
        
        ABS format:
        {
            'event_count': int,
            'persistent_events': [
                {'duration': int, 'start_time': int, 'end_time': int, ...},
                ...
            ]
        }
        
        ANO format:
        {
            'num_events': int,
            'event_durations': {event_id: duration, ...},
            'all_event_ids': [event_id, ...],
            ...
        }
    season : str
        Season identifier (e.g., 'DJF', 'JJA', 'MAM', 'SON') for labeling.
    method : str, optional
        Blocking detection method ('ABS' or 'ANO'). If None, will be auto-detected
        from event_info format. If provided, will be included in the title.
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    color : str, default='steelblue'
        Color for the histogram bars. Common choices:
        - 'steelblue' for winter (DJF)
        - 'coral' for summer (JJA)
        - 'forestgreen' for spring (MAM)
        - 'goldenrod' for autumn (SON)
    figsize : tuple of int, default=(10, 6)
        Figure size in inches (width, height).
    dpi : int, default=150
        Resolution for saved figure.
    show_plot : bool, default=True
        Whether to display the plot using plt.show().
    
    Returns
    -------
    dict or None
        Dictionary containing duration statistics:
        {
            'total_events': int,
            'mean': float,
            'median': float,
            'std': float,
            'min': int,
            'max': int,
            'durations': list of int
        }
        Returns None if no events are found.
    
    Examples
    --------
    >>> # For DJF season
    >>> stats = plot_event_duration_histogram(
    ...     event_info=abs_event_info,
    ...     season='DJF',
    ...     save_path='figures/duration_hist_djf.png',
    ...     color='steelblue'
    ... )
    
    >>> # For JJA season
    >>> stats = plot_event_duration_histogram(
    ...     event_info=abs_event_info_jja,
    ...     season='JJA',
    ...     save_path='figures/duration_hist_jja.png',
    ...     color='coral'
    ... )
    """
    # Auto-detect method if not provided
    if method is None:
        if 'event_count' in event_info and 'persistent_events' in event_info:
            method = 'ABS'
        elif 'num_events' in event_info and 'event_durations' in event_info:
            method = 'ANO'
        else:
            method = 'Blocking'  # Fallback if format not recognized
    
    # Detect format and extract durations
    # Check for ABS format (has 'event_count' and 'persistent_events')
    # Check for ANO format (has 'num_events' and 'event_durations')
    
    if 'event_count' in event_info and 'persistent_events' in event_info:
        # ABS format
        event_count = event_info.get('event_count', 0)
        if event_count == 0:
            print(f"No events detected for {season}. Cannot create histogram.")
            return None
        persistent_events = event_info['persistent_events']
        durations = [event['duration'] for event in persistent_events]
    elif 'num_events' in event_info and 'event_durations' in event_info:
        # ANO format
        num_events = event_info.get('num_events', 0)
        if num_events == 0:
            print(f"No events detected for {season}. Cannot create histogram.")
            return None
        event_durations = event_info['event_durations']
        durations = list(event_durations.values())
    else:
        # Try to determine format by checking available keys
        available_keys = list(event_info.keys())
        raise ValueError(
            f"event_info must be in ABS format (with 'event_count' and 'persistent_events') "
            f"or ANO format (with 'num_events' and 'event_durations'). "
            f"Available keys: {available_keys}"
        )
    
    # Calculate statistics
    mean_duration = np.mean(durations)
    median_duration = np.median(durations)
    std_duration = np.std(durations)
    min_duration = min(durations)
    max_duration = max(durations)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create histogram with 1-day bins
    bins = np.arange(min_duration, max_duration + 2, 1)
    counts, edges, patches = ax.hist(
        durations, 
        bins=bins, 
        edgecolor='black', 
        alpha=0.7, 
        color=color
    )
    
    # Add mean and median lines
    ax.axvline(
        mean_duration, 
        color='red', 
        linestyle='--', 
        linewidth=2, 
        label=f'Mean: {mean_duration:.1f} days'
    )
    ax.axvline(
        median_duration, 
        color='orange', 
        linestyle='--', 
        linewidth=2, 
        label=f'Median: {median_duration:.1f} days'
    )
    
    # Labels and title
    ax.set_xlabel('Event Duration (days)', fontsize=12)
    ax.set_ylabel('Number of Events', fontsize=12)
    ax.set_title(
        f'{method} Blocking Event Duration Distribution ({season})\n'
        f'Total Events: {len(durations)}',
        fontsize=14,
        fontweight='bold'
    )
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    # Add statistics text box
    stats_text = (
        f'Min: {min_duration} days\n'
        f'Max: {max_duration} days\n'
        f'Std: {std_duration:.1f} days'
    )
    ax.text(
        0.98, 0.97, 
        stats_text, 
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Histogram saved to: {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    # Print statistics to console
    print(f"\n{season} Event Duration Statistics:")
    print(f"  Total events: {len(durations)}")
    print(f"  Mean duration: {mean_duration:.2f} days")
    print(f"  Median duration: {median_duration:.2f} days")
    print(f"  Std deviation: {std_duration:.2f} days")
    print(f"  Min duration: {min_duration} days")
    print(f"  Max duration: {max_duration} days")
    
    # Return statistics dictionary
    return {
        'total_events': len(durations),
        'mean': float(mean_duration),
        'median': float(median_duration),
        'std': float(std_duration),
        'min': int(min_duration),
        'max': int(max_duration),
        'durations': durations
    }


def plot_event_area_histogram(
    event_info: Dict,
    season: str,
    method: Optional[str] = None,
    save_path: Optional[str] = None,
    color: str = 'steelblue',
    figsize: Tuple[int, int] = (10, 6),
    dpi: int = 150,
    show_plot: bool = True,
    area_metric: str = 'mean'
) -> Optional[Dict[str, float]]:
    """
    Create a histogram of blocking event areas with statistics.
    
    Supports both ABS and ANO event_info formats:
    - ABS format: {'event_count': int, 'persistent_events': [{'duration': int, 'masks': [...], ...}, ...]}
    - ANO format: {'num_events': int, 'event_areas': {event_id: [area1, area2, ...], ...}, ...}
    
    Parameters
    ----------
    event_info : dict
        Event information dictionary containing event areas.
    season : str
        Season identifier (e.g., 'DJF', 'JJA', 'MAM', 'SON') for labeling.
    method : str, optional
        Blocking detection method ('ABS' or 'ANO'). If None, will be auto-detected.
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    color : str, default='steelblue'
        Color for the histogram bars.
    figsize : tuple of int, default=(10, 6)
        Figure size in inches (width, height).
    dpi : int, default=150
        Resolution for saved figure.
    show_plot : bool, default=True
        Whether to display the plot using plt.show().
    area_metric : str, default='mean'
        Which area metric to use per event: 'mean', 'max', or 'min'.
    
    Returns
    -------
    dict or None
        Dictionary containing area statistics:
        {
            'total_events': int,
            'mean': float,
            'median': float,
            'std': float,
            'min': float,
            'max': float,
            'areas': list of float (in 10⁶ km²)
        }
        Returns None if no events are found.
    
    Examples
    --------
    >>> stats = plot_event_area_histogram(
    ...     event_info=ano_stats,
    ...     season='DJF',
    ...     method='ANO',
    ...     save_path='figures/area_hist_djf.png',
    ...     color='steelblue'
    ... )
    """
    # Auto-detect method if not provided
    if method is None:
        if 'event_count' in event_info and 'persistent_events' in event_info:
            method = 'ABS'
        elif 'num_events' in event_info and 'event_areas' in event_info:
            method = 'ANO'
        else:
            method = 'Blocking'
    
    # Extract areas based on format
    areas_per_event = []  # Will store one area value per event (in 10⁶ km²)
    
    if 'event_count' in event_info and 'persistent_events' in event_info:
        # ABS format - need to compute areas from masks if available
        event_count = event_info.get('event_count', 0)
        if event_count == 0:
            print(f"No events detected for {season}. Cannot create histogram.")
            return None
        
        # Check if event_areas exists (from convert_abs_events_to_ano_format)
        if 'event_areas' in event_info:
            event_areas = event_info['event_areas']
            for event_id, daily_areas in event_areas.items():
                if area_metric == 'mean':
                    area_val = np.mean(daily_areas) / 1e6
                elif area_metric == 'max':
                    area_val = np.max(daily_areas) / 1e6
                else:
                    area_val = np.min(daily_areas) / 1e6
                areas_per_event.append(area_val)
        else:
            # Compute areas from masks using z500_prepared coordinates
            if 'z500_prepared' not in event_info:
                print(f"ABS format requires 'z500_prepared' or 'event_areas' for area histogram.")
                return None
            
            z500_prepared = event_info['z500_prepared']
            lat = z500_prepared.lat.values
            lon = z500_prepared.lon.values
            
            # Get lat from final_blocking_mask (has central lats only)
            if 'final_blocking_mask' in event_info:
                lat_central = event_info['final_blocking_mask'].lat.values
            else:
                lat_central = lat
            
            # Calculate grid cell areas (in km²)
            R_earth = 6371.0  # km
            n_lon = len(lon)
            delta_lon_rad = 2.0 * np.pi / n_lon
            
            # For each persistent event, compute areas from masks
            persistent_events = event_info['persistent_events']
            for event in persistent_events:
                event_daily_areas = []
                for mask_2d in event['masks']:
                    # Calculate area for this day
                    total_area = 0.0
                    for i, lat_val in enumerate(lat_central):
                        if i >= mask_2d.shape[0]:
                            continue
                        n_cells = mask_2d[i, :].sum()
                        if n_cells > 0:
                            lat_rad = np.deg2rad(lat_val)
                            # Approximate cell area
                            cell_area = (R_earth ** 2) * delta_lon_rad * np.cos(lat_rad) * (np.pi / 180) * 2.8  # ~2.8° lat spacing
                            total_area += cell_area * n_cells
                    event_daily_areas.append(total_area)
                
                if event_daily_areas:
                    if area_metric == 'mean':
                        area_val = np.mean(event_daily_areas) / 1e6
                    elif area_metric == 'max':
                        area_val = np.max(event_daily_areas) / 1e6
                    else:
                        area_val = np.min(event_daily_areas) / 1e6
                    areas_per_event.append(area_val)
            
    elif 'num_events' in event_info and 'event_areas' in event_info:
        # ANO format
        num_events = event_info.get('num_events', 0)
        if num_events == 0:
            print(f"No events detected for {season}. Cannot create histogram.")
            return None
        event_areas = event_info['event_areas']
        for event_id, daily_areas in event_areas.items():
            if area_metric == 'mean':
                area_val = np.mean(daily_areas) / 1e6
            elif area_metric == 'max':
                area_val = np.max(daily_areas) / 1e6
            else:
                area_val = np.min(daily_areas) / 1e6
            areas_per_event.append(area_val)
    else:
        available_keys = list(event_info.keys())
        raise ValueError(
            f"event_info must contain 'event_areas'. Available keys: {available_keys}"
        )
    
    if len(areas_per_event) == 0:
        print(f"No event areas found for {season}.")
        return None
    
    # Calculate statistics
    areas = np.array(areas_per_event)
    mean_area = np.mean(areas)
    median_area = np.median(areas)
    std_area = np.std(areas)
    min_area = np.min(areas)
    max_area = np.max(areas)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create histogram with appropriate bins
    bin_width = 0.5  # 0.5 × 10⁶ km² bins
    bins = np.arange(0, max_area + bin_width * 2, bin_width)
    counts, edges, patches = ax.hist(
        areas, 
        bins=bins, 
        edgecolor='black', 
        alpha=0.7, 
        color=color
    )
    
    # Add mean and median lines
    ax.axvline(
        mean_area, 
        color='red', 
        linestyle='--', 
        linewidth=2, 
        label=f'Mean: {mean_area:.2f} ×10⁶ km²'
    )
    ax.axvline(
        median_area, 
        color='orange', 
        linestyle='--', 
        linewidth=2, 
        label=f'Median: {median_area:.2f} ×10⁶ km²'
    )
    
    # Labels and title
    area_label = {'mean': 'Mean', 'max': 'Maximum', 'min': 'Minimum'}[area_metric]
    ax.set_xlabel(f'{area_label} Event Area (×10⁶ km²)', fontsize=12)
    ax.set_ylabel('Number of Events', fontsize=12)
    ax.set_title(
        f'{method} Blocking Event Area Distribution ({season})\n'
        f'Total Events: {len(areas)} | Using {area_label.lower()} area per event',
        fontsize=14,
        fontweight='bold'
    )
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    # Add statistics text box
    stats_text = (
        f'Min: {min_area:.2f} ×10⁶ km²\n'
        f'Max: {max_area:.2f} ×10⁶ km²\n'
        f'Std: {std_area:.2f} ×10⁶ km²'
    )
    ax.text(
        0.98, 0.97, 
        stats_text, 
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Histogram saved to: {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    # Print statistics to console
    print(f"\n{season} Event Area Statistics ({area_label}):")
    print(f"  Total events: {len(areas)}")
    print(f"  Mean area: {mean_area:.2f} ×10⁶ km²")
    print(f"  Median area: {median_area:.2f} ×10⁶ km²")
    print(f"  Std deviation: {std_area:.2f} ×10⁶ km²")
    print(f"  Min area: {min_area:.2f} ×10⁶ km²")
    print(f"  Max area: {max_area:.2f} ×10⁶ km²")
    
    # Return statistics dictionary
    return {
        'total_events': len(areas),
        'mean': float(mean_area),
        'median': float(median_area),
        'std': float(std_area),
        'min': float(min_area),
        'max': float(max_area),
        'areas': list(areas)
    }


def compare_seasonal_durations(
    event_info_dict: Dict[str, Dict],
    method: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 6),
    dpi: int = 150,
    show_plot: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Create side-by-side histograms comparing event durations across seasons.
    
    Parameters
    ----------
    event_info_dict : dict
        Dictionary mapping season names to event_info dictionaries.
        Example: {'DJF': abs_event_info, 'JJA': abs_event_info_jja}
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    figsize : tuple of int, default=(14, 6)
        Figure size in inches (width, height).
    dpi : int, default=150
        Resolution for saved figure.
    show_plot : bool, default=True
        Whether to display the plot using plt.show().
    
    Returns
    -------
    dict
        Dictionary mapping season names to their statistics dictionaries.
    
    Examples
    --------
    >>> stats = compare_seasonal_durations(
    ...     event_info_dict={'DJF': abs_event_info, 'JJA': abs_event_info_jja},
    ...     save_path='figures/duration_comparison.png'
    ... )
    """
    season_colors = {
        'DJF': 'steelblue',
        'MAM': 'forestgreen',
        'JJA': 'coral',
        'SON': 'goldenrod'
    }
    
    n_seasons = len(event_info_dict)
    fig, axes = plt.subplots(1, n_seasons, figsize=figsize)
    
    if n_seasons == 1:
        axes = [axes]
    
    all_stats = {}
    
    for idx, (season, event_info) in enumerate(event_info_dict.items()):
        ax = axes[idx]
        
        # Auto-detect method for this season if not provided
        season_method = method
        if season_method is None:
            if 'event_count' in event_info and 'persistent_events' in event_info:
                season_method = 'ABS'
            elif 'num_events' in event_info and 'event_durations' in event_info:
                season_method = 'ANO'
            else:
                season_method = None  # Will be omitted from title
        
        # Detect format and check for events
        if 'event_count' in event_info:
            # ABS format
            if event_info['event_count'] == 0:
                ax.text(0.5, 0.5, f'No events\nfor {season}',
                       ha='center', va='center', fontsize=14)
                title_prefix = f'{season_method} ' if season_method else ''
                ax.set_title(f'{title_prefix}{season}', fontsize=14, fontweight='bold')
                continue
            persistent_events = event_info['persistent_events']
            durations = [event['duration'] for event in persistent_events]
        elif 'num_events' in event_info:
            # ANO format
            if event_info['num_events'] == 0:
                ax.text(0.5, 0.5, f'No events\nfor {season}',
                       ha='center', va='center', fontsize=14)
                title_prefix = f'{season_method} ' if season_method else ''
                ax.set_title(f'{title_prefix}{season}', fontsize=14, fontweight='bold')
                continue
            event_durations = event_info['event_durations']
            durations = list(event_durations.values())
        else:
            ax.text(0.5, 0.5, f'Invalid format\nfor {season}',
                   ha='center', va='center', fontsize=14)
            title_prefix = f'{season_method} ' if season_method else ''
            ax.set_title(f'{title_prefix}{season}', fontsize=14, fontweight='bold')
            continue
        
        # Calculate statistics
        mean_duration = np.mean(durations)
        median_duration = np.median(durations)
        std_duration = np.std(durations)
        min_duration = min(durations)
        max_duration = max(durations)
        
        # Create histogram
        color = season_colors.get(season, 'steelblue')
        bins = np.arange(min_duration, max_duration + 2, 1)
        ax.hist(durations, bins=bins, edgecolor='black', alpha=0.7, color=color)
        
        # Add mean and median lines
        ax.axvline(mean_duration, color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {mean_duration:.1f}')
        ax.axvline(median_duration, color='orange', linestyle='--', linewidth=2,
                  label=f'Median: {median_duration:.1f}')
        
        # Labels
        ax.set_xlabel('Duration (days)', fontsize=11)
        if idx == 0:
            ax.set_ylabel('Number of Events', fontsize=11)
        title_prefix = f'{season_method} ' if season_method else ''
        ax.set_title(f'{title_prefix}{season}\n(n={len(durations)})', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        
        # Store statistics
        all_stats[season] = {
            'total_events': len(durations),
            'mean': float(mean_duration),
            'median': float(median_duration),
            'std': float(std_duration),
            'min': int(min_duration),
            'max': int(max_duration),
            'durations': durations
        }
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return all_stats

