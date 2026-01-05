"""
Duration-Intensity and Duration-Area distribution plots for blocking events.

This module provides functions to compute and visualize the relationship between
blocking event duration and either intensity (mean Z500 anomaly) or area.

Functions:
    - compute_duration_intensity: Extract duration and amplitude-based intensity
    - compute_duration_area: Extract duration and mean blocked area
    - plot_duration_intensity_enhanced: 2D histogram with KDE contours + marginals for D-I
    - plot_duration_area_enhanced: 2D histogram with KDE contours + marginals for D-A
    - plot_duration_intensity_simple: KDE contours + 90th percentile lines only for D-I
    - plot_duration_area_simple: KDE contours + 90th percentile lines only for D-A
    - plot_duration_intensity: Simple contour plot for overlaying datasets
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm
from scipy.stats import gaussian_kde, spearmanr


def compute_duration_intensity(ano_stats):
    """
    Extract duration and amplitude-based intensity arrays from ano_stats.
    
    Intensity is computed as the area-weighted mean Z500 anomaly inside
    the blocked region, averaged over the event duration:
    
        I_e^(amp) = (1/D_e) * sum_{t=t0}^{tf} [ ∫ I_R(x,t) Z'(x,t) dA / ∫ I_R(x,t) dA ]
    
    This gives the average Z500 anomaly (in meters) inside the block.
    
    Parameters
    ----------
    ano_stats : dict
        Event statistics dictionary containing:
        - 'z500_anom': xarray DataArray of Z500 anomalies
        - 'event_mask': numpy array with event IDs
        - 'all_event_ids': list of event IDs
        - 'event_durations': dict mapping event ID to duration
    
    Returns
    -------
    durations : np.ndarray
        Array of event durations (days)
    intensities : np.ndarray
        Array of mean Z500 anomalies inside blocked region (meters)
    """
    z500_anom = ano_stats['z500_anom']
    lat, lon = z500_anom.lat.values, z500_anom.lon.values
    
    # Grid cell areas (km²)
    dlat, dlon = np.abs(np.diff(lat).mean()), np.abs(np.diff(lon).mean())
    lat_rad = np.deg2rad(lat)
    lat_edges = np.concatenate([[lat_rad[0] - np.deg2rad(dlat)/2], 
                                 (lat_rad[:-1] + lat_rad[1:])/2, 
                                 [lat_rad[-1] + np.deg2rad(dlat)/2]])
    areas = 6371**2 * np.deg2rad(dlon) * np.abs(np.sin(lat_edges[1:]) - np.sin(lat_edges[:-1]))
    grid_areas = np.tile(areas[:, None], (1, len(lon)))
    
    print(f"Grid cell areas computed ({ano_stats.get('season', 'unknown')}): shape = {grid_areas.shape}")
    
    event_mask, z_vals = ano_stats['event_mask'], z500_anom.values
    
    # Squeeze z_vals to ensure it's (time, lat, lon) and matches event_mask
    if z_vals.ndim == 4:
        print(f"  Note: Squeezing z_vals from {z_vals.shape} to remove extra dimension (likely plev)")
        z_vals = z_vals.squeeze()
    
    durations, intensities = [], []
    
    for eid in ano_stats['all_event_ids']:
        t_idx = np.where(np.any(event_mask == eid, axis=(1,2)))[0]
        if len(t_idx) == 0: continue
        
        # Compute area-weighted mean Z500 anomaly for each timestep
        I_t = []
        for t in t_idx:
            mask_t = (event_mask[t] == eid)
            area_weighted_z = np.sum(mask_t * z_vals[t] * grid_areas)
            total_area = np.sum(mask_t * grid_areas)
            I_t.append(area_weighted_z / total_area)  # Mean Z500' inside block [m]
        
        durations.append(ano_stats['event_durations'][eid])
        intensities.append(np.mean(I_t))  # Average over event duration [m]
    
    durations, intensities = np.array(durations), np.array(intensities)
    print(f"\nComputed amplitude intensity for {len(durations)} events")
    print(f"Duration range: {durations.min()} to {durations.max()} days")
    print(f"Intensity range: {intensities.min():.1f} to {intensities.max():.1f} m")
    
    return durations, intensities


def compute_duration_area(ano_stats):
    """
    Extract duration and mean area arrays from ano_stats.
    
    Area is computed as the mean blocked area over the event duration:
    
        A_e = (1/D_e) * sum_{t=t0}^{tf} ∫ I_R(x,t) dA
    
    Parameters
    ----------
    ano_stats : dict
        Event statistics dictionary containing:
        - 'all_event_ids': list of event IDs
        - 'event_durations': dict mapping event ID to duration
        - 'event_areas': dict mapping event ID to list of daily areas (km²)
    
    Returns
    -------
    durations : np.ndarray
        Array of event durations (days)
    areas : np.ndarray
        Array of mean blocked areas (×10⁶ km²)
    """
    durations, areas = [], []
    
    for eid in ano_stats['all_event_ids']:
        duration = ano_stats['event_durations'][eid]
        event_areas = ano_stats['event_areas'][eid]  # Already in km²
        mean_area = np.mean(event_areas) / 1e6  # Convert to 10⁶ km²
        
        durations.append(duration)
        areas.append(mean_area)
    
    durations, areas = np.array(durations), np.array(areas)
    print(f"Computed area for {len(durations)} events")
    print(f"Duration range: {durations.min()} to {durations.max()} days")
    print(f"Area range: {areas.min():.2f} to {areas.max():.2f} × 10⁶ km²")
    
    return durations, areas


def plot_duration_intensity_enhanced(durations, intensities, label='PlaSim', season='DJF',
                                     pct=97, cmap='YlOrRd', figsize=(12, 10)):
    """
    Enhanced duration-intensity plot with 2D histogram, KDE contours, and marginals.
    
    Parameters
    ----------
    durations : np.ndarray
        Array of event durations (days)
    intensities : np.ndarray
        Array of mean Z500 anomalies (meters)
    label : str
        Dataset label for title (default: 'PlaSim')
    season : str
        Season label for title (default: 'DJF')
    pct : int
        Percentile for upper display limit (default: 97)
    cmap : str
        Colormap for 2D histogram (default: 'YlOrRd')
    figsize : tuple
        Figure size (default: (12, 10))
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    ax_main : matplotlib.axes.Axes
    """
    # Create figure with GridSpec for marginal histograms
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(4, 4, figure=fig, hspace=0.05, wspace=0.05)
    
    # Main 2D histogram/contour plot
    ax_main = fig.add_subplot(gs[1:4, 0:3])
    # Top marginal histogram (duration)
    ax_hist_top = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
    # Right marginal histogram (intensity)
    ax_hist_right = fig.add_subplot(gs[1:4, 3], sharey=ax_main)
    
    # --- Debug info ---
    print(f"\n📊 Plotting Debug ({season}, {label}):")
    print(f"  N events: {len(durations)}")
    print(f"  Durations: min={durations.min():.1f}, max={durations.max():.1f}, mean={durations.mean():.1f}")
    print(f"  Intensities: min={intensities.min():.1f}, max={intensities.max():.1f}, mean={intensities.mean():.1f}")
    print(f"  Unique intensities: {len(np.unique(intensities))}")
    
    # --- Compute statistics ---
    n_events = len(durations)
    
    # 90th percentile thresholds
    d_90 = np.percentile(durations, 90)
    i_90 = np.percentile(intensities, 90)
    
    # Upper limit for display (97th percentile)
    d_pct, i_pct = np.percentile(durations, pct), np.percentile(intensities, pct)
    
    # --- 2D Histogram ---
    # Use integer bins for duration (since durations are integers)
    d_bins = np.arange(durations.min() - 0.5, durations.max() + 1.5, 1)
    
    # Use explicit linear bins for intensity to ensure smooth distribution
    i_min, i_max = intensities.min(), intensities.max()
    if i_max == i_min:
        i_bins = np.linspace(i_min - 10, i_min + 10, 30)
    else:
        i_bins = np.linspace(i_min, i_max, 30)
    
    # Create 2D histogram
    h, xedges, yedges = np.histogram2d(durations, intensities, bins=[d_bins, i_bins])
    
    # Plot 2D histogram using pcolormesh with fixed alignment
    # h.T because pcolormesh expects (ny, nx)
    # shading='flat' requires edges to have size +1
    h_masked = np.ma.masked_where(h.T <= 0, h.T)
    
    # Robust normalization for LogNorm
    if h.max() > 1:
        norm = LogNorm(vmin=1, vmax=h.max())
    else:
        norm = None
        
    pcm = ax_main.pcolormesh(xedges, yedges, h_masked, cmap=cmap, shading='flat', 
                             norm=norm, alpha=0.8)
    
    # --- KDE contours overlay ---
    # Normalize data for more robust KDE bandwidth selection
    d_std, i_std = durations.std(), intensities.std()
    if d_std == 0: d_std = 1.0
    if i_std == 0: i_std = 1.0
    
    # Only compute KDE if we have enough points and variation
    if n_events > 5 and i_std > 1e-6 and d_std > 1e-6:
        dur_norm = (durations - durations.mean()) / d_std
        int_norm = (intensities - intensities.mean()) / i_std
        
        xy_norm = np.vstack([dur_norm, int_norm])
        try:
            kde = gaussian_kde(xy_norm, bw_method='scott')
            
            # Grid for contour plot (must also be normalized)
            d_grid = np.linspace(durations.min() - 1, d_pct * 1.1, 100)
            i_grid = np.linspace(intensities.min() * 0.9, i_pct * 1.1, 100)
            D, I = np.meshgrid(d_grid, i_grid)
            
            D_norm = (D - durations.mean()) / d_std
            I_norm = (I - intensities.mean()) / i_std
            
            Z = kde(np.vstack([D_norm.ravel(), I_norm.ravel()])).reshape(D.shape)
            Z = Z / Z.max()  # Normalize
            
            # KDE contour lines
            levels_line = [0.1, 0.3, 0.5, 0.7, 0.9]
            cs = ax_main.contour(D, I, Z, levels=levels_line, colors='black', linewidths=1.2, alpha=0.7)
            ax_main.clabel(cs, inline=True, fontsize=8, fmt=lambda x: f'{x:.1f}')
        except Exception as e:
            print(f"  Warning: KDE computation failed: {e}")
    else:
        print("  Skipping KDE due to low variance or too few events.")
    
    # --- 90th percentile reference lines ---
    ax_main.axvline(d_90, color='#1E88E5', linestyle='--', linewidth=2.5, alpha=0.9,
                    label=rf'$D_{{0.9}}$ = {d_90:.0f} days')
    ax_main.axhline(i_90, color='#43A047', linestyle='--', linewidth=2.5, alpha=0.9,
                    label=rf'$I_{{0.9}}$ = {i_90:.0f} m')
    
    # Labels
    ax_main.set_xlabel('Duration $D$ (days)', fontsize=12, fontweight='bold')
    ax_main.set_ylabel(r'Mean Z500 Anomaly $I$ (m)', fontsize=12, fontweight='bold')
    ax_main.grid(True, alpha=0.3, linestyle='-')
    ax_main.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    # Set limits
    ax_main.set_xlim(durations.min() - 1, d_pct * 1.05)
    ax_main.set_ylim(intensities.min() * 0.9, i_pct * 1.05)
    
    # --- Top marginal histogram (Duration) ---
    bins_d = np.arange(durations.min() - 0.5, durations.max() + 1.5, 1)
    ax_hist_top.hist(durations, bins=bins_d, color='steelblue', edgecolor='white', alpha=0.7, density=True)
    
    # KDE overlay for duration
    d_kde = gaussian_kde(durations, bw_method='scott')
    d_range = np.linspace(durations.min() - 1, d_pct * 1.05, 200)
    ax_hist_top.plot(d_range, d_kde(d_range), 'darkblue', linewidth=2, label='KDE')
    ax_hist_top.axvline(d_90, color='#1E88E5', linestyle='--', linewidth=2, alpha=0.9)
    ax_hist_top.set_ylabel('Density', fontsize=10)
    ax_hist_top.tick_params(labelbottom=False)
    ax_hist_top.set_xlim(ax_main.get_xlim())
    
    # --- Right marginal histogram (Intensity) ---
    bins_i = 30
    ax_hist_right.hist(intensities, bins=bins_i, orientation='horizontal', 
                       color='coral', edgecolor='white', alpha=0.7, density=True)
    
    # KDE overlay for intensity
    i_kde = gaussian_kde(intensities, bw_method='scott')
    i_range = np.linspace(intensities.min() * 0.9, i_pct * 1.05, 200)
    ax_hist_right.plot(i_kde(i_range), i_range, 'darkred', linewidth=2, label='KDE')
    ax_hist_right.axhline(i_90, color='#43A047', linestyle='--', linewidth=2, alpha=0.9)
    ax_hist_right.set_xlabel('Density', fontsize=10)
    ax_hist_right.tick_params(labelleft=False)
    ax_hist_right.set_ylim(ax_main.get_ylim())
    
    # --- Minimal annotation (just N) ---
    ax_main.text(0.02, 0.98, f'$N$ = {n_events}', transform=ax_main.transAxes, fontsize=11,
                 verticalalignment='top', horizontalalignment='left', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    # --- Title ---
    fig.suptitle(f'Duration–Intensity Distribution ({season}, {label})\n'
                 r'$I_e^{\rm (amp)}$ = Mean Z500 Anomaly inside Blocked Region', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # Colorbar for 2D histogram
    cbar_ax = fig.add_axes([0.78, 0.12, 0.02, 0.25])
    cbar = fig.colorbar(pcm, cax=cbar_ax)
    cbar.set_label('Event Count', fontsize=9)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig, ax_main


def plot_duration_area_enhanced(durations, areas, label='PlaSim', season='DJF',
                                pct=97, cmap='YlGnBu', figsize=(12, 10)):
    """
    Duration-Area distribution plot with 2D histogram, KDE contours, and marginals.
    
    Parameters
    ----------
    durations : np.ndarray
        Array of event durations (days)
    areas : np.ndarray
        Array of mean blocked areas (×10⁶ km²)
    label : str
        Dataset label for title (default: 'PlaSim')
    season : str
        Season label for title (default: 'DJF')
    pct : int
        Percentile for upper display limit (default: 97)
    cmap : str
        Colormap for 2D histogram (default: 'YlGnBu')
    figsize : tuple
        Figure size (default: (12, 10))
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    ax_main : matplotlib.axes.Axes
    """
    # Create figure with GridSpec for marginal histograms
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(4, 4, figure=fig, hspace=0.05, wspace=0.05)
    
    # Main 2D histogram/contour plot
    ax_main = fig.add_subplot(gs[1:4, 0:3])
    # Top marginal histogram (duration)
    ax_hist_top = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
    # Right marginal histogram (area)
    ax_hist_right = fig.add_subplot(gs[1:4, 3], sharey=ax_main)
    
    # --- Compute statistics ---
    n_events = len(durations)
    
    # 90th percentile thresholds
    d_90 = np.percentile(durations, 90)
    a_90 = np.percentile(areas, 90)
    
    # Upper limit for display (97th percentile)
    d_pct, a_pct = np.percentile(durations, pct), np.percentile(areas, pct)
    
    # --- 2D Histogram ---
    # Use integer bins for duration (since durations are integers)
    d_bins = np.arange(durations.min() - 0.5, durations.max() + 1.5, 1)
    
    # Use explicit linear bins for area to ensure smooth distribution
    a_min, a_max = areas.min(), areas.max()
    if a_max == a_min:
        a_bins = np.linspace(a_min - 0.5, a_min + 0.5, 30)
    else:
        a_bins = np.linspace(a_min, a_max, 30)
    
    # Create 2D histogram
    h, xedges, yedges = np.histogram2d(durations, areas, bins=[d_bins, a_bins])
    
    # Plot 2D histogram as pcolormesh
    X, Y = np.meshgrid(xedges, yedges)
    h_masked = np.ma.masked_where(h.T == 0, h.T)  # Mask zero counts
    pcm = ax_main.pcolormesh(X, Y, h_masked, cmap=cmap, shading='flat', 
                             norm=LogNorm(vmin=1, vmax=h.max()))
    
    # --- KDE contours overlay ---
    # Normalize data for more robust KDE bandwidth selection
    d_std, a_std = durations.std(), areas.std()
    if d_std == 0: d_std = 1.0
    if a_std == 0: a_std = 1.0
    
    dur_norm = (durations - durations.mean()) / d_std
    area_norm = (areas - areas.mean()) / a_std
    
    xy_norm = np.vstack([dur_norm, area_norm])
    kde = gaussian_kde(xy_norm, bw_method='scott')
    
    # Grid for contour plot (must also be normalized)
    d_grid = np.linspace(durations.min() - 1, d_pct * 1.1, 100)
    a_grid = np.linspace(areas.min() * 0.9, a_pct * 1.1, 100)
    D, A = np.meshgrid(d_grid, a_grid)
    
    D_norm = (D - durations.mean()) / d_std
    A_norm = (A - areas.mean()) / a_std
    
    Z = kde(np.vstack([D_norm.ravel(), A_norm.ravel()])).reshape(D.shape)
    Z = Z / Z.max()  # Normalize
    
    # KDE contour lines
    levels_line = [0.1, 0.3, 0.5, 0.7, 0.9]
    cs = ax_main.contour(D, A, Z, levels=levels_line, colors='black', linewidths=1.2, alpha=0.7)
    ax_main.clabel(cs, inline=True, fontsize=8, fmt=lambda x: f'{x:.1f}')
    
    # --- 90th percentile reference lines ---
    ax_main.axvline(d_90, color='#1E88E5', linestyle='--', linewidth=2.5, alpha=0.9,
                    label=rf'$D_{{0.9}}$ = {d_90:.0f} days')
    ax_main.axhline(a_90, color='#8E24AA', linestyle='--', linewidth=2.5, alpha=0.9,
                    label=rf'$A_{{0.9}}$ = {a_90:.1f} ×10⁶ km²')
    
    # Labels
    ax_main.set_xlabel('Duration $D$ (days)', fontsize=12, fontweight='bold')
    ax_main.set_ylabel(r'Mean Blocked Area $A$ ($\times 10^6$ km²)', fontsize=12, fontweight='bold')
    ax_main.grid(True, alpha=0.3, linestyle='-')
    ax_main.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    # Set limits
    ax_main.set_xlim(durations.min() - 1, d_pct * 1.05)
    ax_main.set_ylim(areas.min() * 0.9, a_pct * 1.05)
    
    # --- Top marginal histogram (Duration) ---
    bins_d = np.arange(durations.min() - 0.5, durations.max() + 1.5, 1)
    ax_hist_top.hist(durations, bins=bins_d, color='steelblue', edgecolor='white', alpha=0.7, density=True)
    
    # KDE overlay for duration
    d_kde = gaussian_kde(durations, bw_method='scott')
    d_range = np.linspace(durations.min() - 1, d_pct * 1.05, 200)
    ax_hist_top.plot(d_range, d_kde(d_range), 'darkblue', linewidth=2, label='KDE')
    ax_hist_top.axvline(d_90, color='#1E88E5', linestyle='--', linewidth=2, alpha=0.9)
    ax_hist_top.set_ylabel('Density', fontsize=10)
    ax_hist_top.tick_params(labelbottom=False)
    ax_hist_top.set_xlim(ax_main.get_xlim())
    
    # --- Right marginal histogram (Area) ---
    bins_a = 30
    ax_hist_right.hist(areas, bins=bins_a, orientation='horizontal', 
                       color='mediumseagreen', edgecolor='white', alpha=0.7, density=True)
    
    # KDE overlay for area
    a_kde = gaussian_kde(areas, bw_method='scott')
    a_range = np.linspace(areas.min() * 0.9, a_pct * 1.05, 200)
    ax_hist_right.plot(a_kde(a_range), a_range, 'darkgreen', linewidth=2, label='KDE')
    ax_hist_right.axhline(a_90, color='#8E24AA', linestyle='--', linewidth=2, alpha=0.9)
    ax_hist_right.set_xlabel('Density', fontsize=10)
    ax_hist_right.tick_params(labelleft=False)
    ax_hist_right.set_ylim(ax_main.get_ylim())
    
    # --- Minimal annotation (just N) ---
    ax_main.text(0.02, 0.98, f'$N$ = {n_events}', transform=ax_main.transAxes, fontsize=11,
                 verticalalignment='top', horizontalalignment='left', fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    # --- Title ---
    fig.suptitle(f'Duration–Area Distribution ({season}, {label})\n'
                 r'$A_e$ = Mean Blocked Area over Event Duration', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # Colorbar for 2D histogram
    cbar_ax = fig.add_axes([0.78, 0.12, 0.02, 0.25])
    cbar = fig.colorbar(pcm, cax=cbar_ax)
    cbar.set_label('Event Count', fontsize=9)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig, ax_main


def plot_duration_intensity(durations, intensities, label='PlaSim', ax=None, 
                            pct=97, cmap='YlOrRd', levels_contour=[0.2, 0.5, 0.8]):
    """
    Simple KDE contour plot for overlaying multiple datasets.
    
    Parameters
    ----------
    durations : np.ndarray
        Array of event durations (days)
    intensities : np.ndarray
        Array of intensities
    label : str
        Dataset label for contour labels
    ax : matplotlib.axes.Axes, optional
        Axes to plot on (creates new figure if None)
    pct : int
        Percentile for upper display limit (default: 97)
    cmap : str
        Colormap for contour lines
    levels_contour : list
        Contour levels to draw
    
    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 6))
    
    d_pct, i_pct = np.percentile(durations, pct), np.percentile(intensities, pct)
    xy = np.vstack([durations, intensities])
    kde = gaussian_kde(xy, bw_method='scott')
    
    d_grid = np.linspace(durations.min() - 0.5, d_pct * 1.05, 80)
    i_grid = np.linspace(intensities.min() * 0.95, i_pct * 1.05, 80)
    D, I = np.meshgrid(d_grid, i_grid)
    Z = kde(np.vstack([D.ravel(), I.ravel()])).reshape(D.shape)
    Z = Z / Z.max()
    
    cs = ax.contour(D, I, Z, levels=levels_contour, cmap=cmap, linewidths=2)
    ax.clabel(cs, inline=True, fontsize=8, fmt=lambda x: f'{label} {x:.1f}')
    
    return ax


def plot_duration_intensity_simple(durations, intensities, label='PlaSim', season='DJF',
                                   pct=97, figsize=(10, 7), levels=[0.1, 0.3, 0.5, 0.7, 0.9],
                                   contour_cmap='Reds', fill_contours=True):
    """
    Simplified duration-intensity plot with KDE contours and 90th percentile lines only.
    No histograms or colored grid points.
    
    Parameters
    ----------
    durations : np.ndarray
        Array of event durations (days)
    intensities : np.ndarray
        Array of mean Z500 anomalies (meters)
    label : str
        Dataset label for title (default: 'PlaSim')
    season : str
        Season label for title (default: 'DJF')
    pct : int
        Percentile for upper display limit (default: 97)
    figsize : tuple
        Figure size (default: (10, 7))
    levels : list
        KDE contour levels (default: [0.1, 0.3, 0.5, 0.7, 0.9])
    contour_cmap : str
        Colormap for filled contours (default: 'Reds')
    fill_contours : bool
        Whether to fill contours (default: True)
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    n_events = len(durations)
    
    # 90th percentile thresholds
    d_90 = np.percentile(durations, 90)
    i_90 = np.percentile(intensities, 90)
    
    # Upper limit for display
    d_pct, i_pct = np.percentile(durations, pct), np.percentile(intensities, pct)
    
    # --- KDE computation ---
    # Normalize data for more robust KDE bandwidth selection
    d_std, i_std = durations.std(), intensities.std()
    if d_std == 0: d_std = 1.0
    if i_std == 0: i_std = 1.0
    
    dur_norm = (durations - durations.mean()) / d_std
    int_norm = (intensities - intensities.mean()) / i_std
    
    xy_norm = np.vstack([dur_norm, int_norm])
    kde = gaussian_kde(xy_norm, bw_method='scott')
    
    # Grid for contour plot (must also be normalized)
    d_grid = np.linspace(durations.min() - 1, d_pct * 1.1, 100)
    i_grid = np.linspace(intensities.min() * 0.9, i_pct * 1.1, 100)
    D, I = np.meshgrid(d_grid, i_grid)
    
    D_norm = (D - durations.mean()) / d_std
    I_norm = (I - intensities.mean()) / i_std
    
    Z = kde(np.vstack([D_norm.ravel(), I_norm.ravel()])).reshape(D.shape)
    Z = Z / Z.max()  # Normalize
    
    # --- Filled contours (optional) ---
    if fill_contours:
        cf = ax.contourf(D, I, Z, levels=np.linspace(0.05, 0.95, 10), 
                         cmap=contour_cmap, alpha=0.5, extend='max')
    
    # --- KDE contour lines ---
    cs = ax.contour(D, I, Z, levels=levels, colors='darkred', linewidths=1.5, alpha=0.8)
    ax.clabel(cs, inline=True, fontsize=9, fmt=lambda x: f'{x:.1f}')
    
    # --- 90th percentile reference lines ---
    ax.axvline(d_90, color='#1E88E5', linestyle='--', linewidth=2.5, alpha=0.9,
               label=rf'$D_{{0.9}}$ = {d_90:.0f} days')
    ax.axhline(i_90, color='#43A047', linestyle='--', linewidth=2.5, alpha=0.9,
               label=rf'$I_{{0.9}}$ = {i_90:.0f} m')
    
    # Labels
    ax.set_xlabel('Duration $D$ (days)', fontsize=12, fontweight='bold')
    ax.set_ylabel(r'Mean Z500 Anomaly $I$ (m)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='-')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    
    # Set limits
    ax.set_xlim(durations.min() - 1, d_pct * 1.05)
    ax.set_ylim(intensities.min() * 0.9, i_pct * 1.05)
    
    # Minimal annotation
    ax.text(0.02, 0.98, f'$N$ = {n_events}', transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='left', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    # Title
    ax.set_title(f'Duration–Intensity Distribution ({season}, {label})\n'
                 r'$I_e^{\rm (amp)}$ = Mean Z500 Anomaly inside Blocked Region',
                 fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    return fig, ax


def plot_duration_area_simple(durations, areas, label='PlaSim', season='DJF',
                              pct=97, figsize=(10, 7), levels=[0.1, 0.3, 0.5, 0.7, 0.9],
                              contour_cmap='Greens', fill_contours=True):
    """
    Simplified duration-area plot with KDE contours and 90th percentile lines only.
    No histograms or colored grid points.
    
    Parameters
    ----------
    durations : np.ndarray
        Array of event durations (days)
    areas : np.ndarray
        Array of mean blocked areas (×10⁶ km²)
    label : str
        Dataset label for title (default: 'PlaSim')
    season : str
        Season label for title (default: 'DJF')
    pct : int
        Percentile for upper display limit (default: 97)
    figsize : tuple
        Figure size (default: (10, 7))
    levels : list
        KDE contour levels (default: [0.1, 0.3, 0.5, 0.7, 0.9])
    contour_cmap : str
        Colormap for filled contours (default: 'Greens')
    fill_contours : bool
        Whether to fill contours (default: True)
    
    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    n_events = len(durations)
    
    # 90th percentile thresholds
    d_90 = np.percentile(durations, 90)
    a_90 = np.percentile(areas, 90)
    
    # Upper limit for display
    d_pct, a_pct = np.percentile(durations, pct), np.percentile(areas, pct)
    
    # --- KDE computation ---
    # Normalize data for more robust KDE bandwidth selection
    d_std, a_std = durations.std(), areas.std()
    if d_std == 0: d_std = 1.0
    if a_std == 0: a_std = 1.0
    
    dur_norm = (durations - durations.mean()) / d_std
    area_norm = (areas - areas.mean()) / a_std
    
    xy_norm = np.vstack([dur_norm, area_norm])
    kde = gaussian_kde(xy_norm, bw_method='scott')
    
    # Grid for contour plot (must also be normalized)
    d_grid = np.linspace(durations.min() - 1, d_pct * 1.1, 100)
    a_grid = np.linspace(areas.min() * 0.9, a_pct * 1.1, 100)
    D, A = np.meshgrid(d_grid, a_grid)
    
    D_norm = (D - durations.mean()) / d_std
    A_norm = (A - areas.mean()) / a_std
    
    Z = kde(np.vstack([D_norm.ravel(), A_norm.ravel()])).reshape(D.shape)
    Z = Z / Z.max()  # Normalize
    
    # --- Filled contours (optional) ---
    if fill_contours:
        cf = ax.contourf(D, A, Z, levels=np.linspace(0.05, 0.95, 10), 
                         cmap=contour_cmap, alpha=0.5, extend='max')
    
    # --- KDE contour lines ---
    cs = ax.contour(D, A, Z, levels=levels, colors='darkgreen', linewidths=1.5, alpha=0.8)
    ax.clabel(cs, inline=True, fontsize=9, fmt=lambda x: f'{x:.1f}')
    
    # --- 90th percentile reference lines ---
    ax.axvline(d_90, color='#1E88E5', linestyle='--', linewidth=2.5, alpha=0.9,
               label=rf'$D_{{0.9}}$ = {d_90:.0f} days')
    ax.axhline(a_90, color='#8E24AA', linestyle='--', linewidth=2.5, alpha=0.9,
               label=rf'$A_{{0.9}}$ = {a_90:.1f} ×10⁶ km²')
    
    # Labels
    ax.set_xlabel('Duration $D$ (days)', fontsize=12, fontweight='bold')
    ax.set_ylabel(r'Mean Blocked Area $A$ ($\times 10^6$ km²)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='-')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    
    # Set limits
    ax.set_xlim(durations.min() - 1, d_pct * 1.05)
    ax.set_ylim(areas.min() * 0.9, a_pct * 1.05)
    
    # Minimal annotation
    ax.text(0.02, 0.98, f'$N$ = {n_events}', transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='left', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    # Title
    ax.set_title(f'Duration–Area Distribution ({season}, {label})\n'
                 r'$A_e$ = Mean Blocked Area over Event Duration',
                 fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    return fig, ax


def _compute_credible_levels(Z, percentages):
    """
    Compute density thresholds that enclose given percentages of probability mass.
    
    Parameters
    ----------
    Z : np.ndarray
        2D array of KDE density values
    percentages : list
        List of percentages (0-100) to compute thresholds for.
        E.g., [50, 75, 90, 95] means contours enclosing 50%, 75%, 90%, 95% of data.
    
    Returns
    -------
    levels : list
        Density thresholds corresponding to each percentage
    """
    # Flatten and sort densities in descending order
    Z_flat = Z.ravel()
    sorted_idx = np.argsort(Z_flat)[::-1]
    sorted_Z = Z_flat[sorted_idx]
    
    # Compute cumulative sum (normalized to 1)
    cumsum = np.cumsum(sorted_Z)
    cumsum = cumsum / cumsum[-1]
    
    # Find density threshold for each percentage
    levels = []
    for pct in percentages:
        target = pct / 100.0
        # Find first index where cumulative sum exceeds target
        idx = np.searchsorted(cumsum, target)
        if idx >= len(sorted_Z):
            idx = len(sorted_Z) - 1
        levels.append(sorted_Z[idx])
    
    return levels


def plot_duration_intensity_comparison(durations1, intensities1, durations2, intensities2,
                                       label1='PlaSim', label2='Emulator', season='DJF',
                                       pct=97, figsize=(10, 7), 
                                       credible_levels=[25, 50, 75, 90]):
    """
    Overlay two duration-intensity distributions for comparison.
    
    Parameters
    ----------
    durations1, intensities1 : np.ndarray
        Duration and intensity arrays for first dataset
    durations2, intensities2 : np.ndarray
        Duration and intensity arrays for second dataset
    label1, label2 : str
        Labels for the two datasets
    season : str
        Season label for title
    pct : int
        Percentile for upper display limit
    figsize : tuple
        Figure size
    credible_levels : list
        Percentages of data points to enclose (default: [50, 75, 90, 95]).
        E.g., 90 means the contour encloses 90% of the data points.
    
    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    from matplotlib.lines import Line2D
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Combine data for axis limits
    all_d = np.concatenate([durations1, durations2])
    all_i = np.concatenate([intensities1, intensities2])
    d_pct, i_pct = np.percentile(all_d, pct), np.percentile(all_i, pct)
    
    # Colors for the two datasets
    colors1 = '#1E88E5'  # Blue for PlaSim
    colors2 = '#E53935'  # Red for Emulator
    
    # --- Dataset 1 (PlaSim) ---
    d1_std, i1_std = durations1.std(), intensities1.std()
    if d1_std == 0: d1_std = 1.0
    if i1_std == 0: i1_std = 1.0
    
    dur1_norm = (durations1 - durations1.mean()) / d1_std
    int1_norm = (intensities1 - intensities1.mean()) / i1_std
    
    xy1_norm = np.vstack([dur1_norm, int1_norm])
    kde1 = gaussian_kde(xy1_norm, bw_method='scott')
    
    d_grid = np.linspace(min(durations1.min(), durations2.min()) - 1, d_pct * 1.1, 100)
    i_grid = np.linspace(min(intensities1.min(), intensities2.min()) * 0.9, i_pct * 1.1, 100)
    D, I = np.meshgrid(d_grid, i_grid)
    
    D1_norm = (D - durations1.mean()) / d1_std
    I1_norm = (I - intensities1.mean()) / i1_std
    Z1 = kde1(np.vstack([D1_norm.ravel(), I1_norm.ravel()])).reshape(D.shape)
    
    # Compute credible level thresholds for dataset 1
    levels1 = _compute_credible_levels(Z1, credible_levels)
    
    cs1 = ax.contour(D, I, Z1, levels=sorted(levels1), colors=colors1, linewidths=2, alpha=0.8)
    # Label contours with percentage of data enclosed
    level_to_pct1 = {lv: pct for lv, pct in zip(levels1, credible_levels)}
    ax.clabel(cs1, inline=True, fontsize=8, 
              fmt=lambda x: f'{level_to_pct1.get(x, int(x*100))}%' if x in level_to_pct1 else '')
    
    # --- Dataset 2 (Emulator) ---
    d2_std, i2_std = durations2.std(), intensities2.std()
    if d2_std == 0: d2_std = 1.0
    if i2_std == 0: i2_std = 1.0
    
    dur2_norm = (durations2 - durations2.mean()) / d2_std
    int2_norm = (intensities2 - intensities2.mean()) / i2_std
    
    xy2_norm = np.vstack([dur2_norm, int2_norm])
    kde2 = gaussian_kde(xy2_norm, bw_method='scott')
    
    D2_norm = (D - durations2.mean()) / d2_std
    I2_norm = (I - intensities2.mean()) / i2_std
    Z2 = kde2(np.vstack([D2_norm.ravel(), I2_norm.ravel()])).reshape(D.shape)
    
    # Compute credible level thresholds for dataset 2
    levels2 = _compute_credible_levels(Z2, credible_levels)
    
    cs2 = ax.contour(D, I, Z2, levels=sorted(levels2), colors=colors2, linewidths=2, alpha=0.8, linestyles='--')
    # Label contours with percentage of data enclosed
    level_to_pct2 = {lv: pct for lv, pct in zip(levels2, credible_levels)}
    ax.clabel(cs2, inline=True, fontsize=8,
              fmt=lambda x: f'{level_to_pct2.get(x, int(x*100))}%' if x in level_to_pct2 else '')
    
    # 90th percentile values (for legend only, no lines)
    d1_90, i1_90 = np.percentile(durations1, 90), np.percentile(intensities1, 90)
    d2_90, i2_90 = np.percentile(durations2, 90), np.percentile(intensities2, 90)
    
    # Create legend with proxy artists (no threshold lines drawn)
    legend_elements = [
        Line2D([0], [0], color=colors1, linewidth=2, linestyle='-',
               label=rf'{label1}: $D_{{0.9}}$={d1_90:.0f}d, $I_{{0.9}}$={i1_90:.0f}m'),
        Line2D([0], [0], color=colors2, linewidth=2, linestyle='--',
               label=rf'{label2}: $D_{{0.9}}$={d2_90:.0f}d, $I_{{0.9}}$={i2_90:.0f}m')
    ]
    
    # Labels
    ax.set_xlabel('Duration $D$ (days)', fontsize=12, fontweight='bold')
    ax.set_ylabel(r'Mean Z500 Anomaly $I$ (m)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='-')
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)
    
    # Limits
    ax.set_xlim(min(durations1.min(), durations2.min()) - 1, d_pct * 1.05)
    ax.set_ylim(min(intensities1.min(), intensities2.min()) * 0.9, i_pct * 1.05)
    
    # Annotation
    ax.text(0.02, 0.98, f'{label1}: N={len(durations1)}\n{label2}: N={len(durations2)}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    ax.set_title(f'Duration–Intensity Comparison ({season})\n'
                 f'{label1} (solid) vs {label2} (dashed)',
                 fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    return fig, ax


def plot_duration_area_comparison(durations1, areas1, durations2, areas2,
                                  label1='PlaSim', label2='Emulator', season='DJF',
                                  pct=97, figsize=(10, 7),
                                  credible_levels=[25, 50, 75, 90]):
    """
    Overlay two duration-area distributions for comparison.
    
    Parameters
    ----------
    durations1, areas1 : np.ndarray
        Duration and area arrays for first dataset
    durations2, areas2 : np.ndarray
        Duration and area arrays for second dataset
    label1, label2 : str
        Labels for the two datasets
    season : str
        Season label for title
    pct : int
        Percentile for upper display limit
    figsize : tuple
        Figure size
    credible_levels : list
        Percentages of data points to enclose (default: [50, 75, 90, 95]).
        E.g., 90 means the contour encloses 90% of the data points.
    
    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    from matplotlib.lines import Line2D
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Combine data for axis limits
    all_d = np.concatenate([durations1, durations2])
    all_a = np.concatenate([areas1, areas2])
    d_pct, a_pct = np.percentile(all_d, pct), np.percentile(all_a, pct)
    
    # Colors for the two datasets
    colors1 = '#1E88E5'  # Blue for PlaSim
    colors2 = '#43A047'  # Green for Emulator
    
    # --- Dataset 1 (PlaSim) ---
    d1_std, a1_std = durations1.std(), areas1.std()
    if d1_std == 0: d1_std = 1.0
    if a1_std == 0: a1_std = 1.0
    
    dur1_norm = (durations1 - durations1.mean()) / d1_std
    area1_norm = (areas1 - areas1.mean()) / a1_std
    
    xy1_norm = np.vstack([dur1_norm, area1_norm])
    kde1 = gaussian_kde(xy1_norm, bw_method='scott')
    
    d_grid = np.linspace(min(durations1.min(), durations2.min()) - 1, d_pct * 1.1, 100)
    a_grid = np.linspace(min(areas1.min(), areas2.min()) * 0.9, a_pct * 1.1, 100)
    D, A = np.meshgrid(d_grid, a_grid)
    
    D1_norm = (D - durations1.mean()) / d1_std
    A1_norm = (A - areas1.mean()) / a1_std
    Z1 = kde1(np.vstack([D1_norm.ravel(), A1_norm.ravel()])).reshape(D.shape)
    
    # Compute credible level thresholds for dataset 1
    levels1 = _compute_credible_levels(Z1, credible_levels)
    
    cs1 = ax.contour(D, A, Z1, levels=sorted(levels1), colors=colors1, linewidths=2, alpha=0.8)
    # Label contours with percentage of data enclosed
    level_to_pct1 = {lv: pct for lv, pct in zip(levels1, credible_levels)}
    ax.clabel(cs1, inline=True, fontsize=8,
              fmt=lambda x: f'{level_to_pct1.get(x, int(x*100))}%' if x in level_to_pct1 else '')
    
    # --- Dataset 2 (Emulator) ---
    d2_std, a2_std = durations2.std(), areas2.std()
    if d2_std == 0: d2_std = 1.0
    if a2_std == 0: a2_std = 1.0
    
    dur2_norm = (durations2 - durations2.mean()) / d2_std
    area2_norm = (areas2 - areas2.mean()) / a2_std
    
    xy2_norm = np.vstack([dur2_norm, area2_norm])
    kde2 = gaussian_kde(xy2_norm, bw_method='scott')
    
    D2_norm = (D - durations2.mean()) / d2_std
    A2_norm = (A - areas2.mean()) / a2_std
    Z2 = kde2(np.vstack([D2_norm.ravel(), A2_norm.ravel()])).reshape(D.shape)
    
    # Compute credible level thresholds for dataset 2
    levels2 = _compute_credible_levels(Z2, credible_levels)
    
    cs2 = ax.contour(D, A, Z2, levels=sorted(levels2), colors=colors2, linewidths=2, alpha=0.8, linestyles='--')
    # Label contours with percentage of data enclosed
    level_to_pct2 = {lv: pct for lv, pct in zip(levels2, credible_levels)}
    ax.clabel(cs2, inline=True, fontsize=8,
              fmt=lambda x: f'{level_to_pct2.get(x, int(x*100))}%' if x in level_to_pct2 else '')
    
    # 90th percentile values (for legend only, no lines)
    d1_90, a1_90 = np.percentile(durations1, 90), np.percentile(areas1, 90)
    d2_90, a2_90 = np.percentile(durations2, 90), np.percentile(areas2, 90)
    
    # Create legend with proxy artists (no threshold lines drawn)
    legend_elements = [
        Line2D([0], [0], color=colors1, linewidth=2, linestyle='-',
               label=rf'{label1}: $D_{{0.9}}$={d1_90:.0f}d, $A_{{0.9}}$={a1_90:.1f}'),
        Line2D([0], [0], color=colors2, linewidth=2, linestyle='--',
               label=rf'{label2}: $D_{{0.9}}$={d2_90:.0f}d, $A_{{0.9}}$={a2_90:.1f}')
    ]
    
    # Labels
    ax.set_xlabel('Duration $D$ (days)', fontsize=12, fontweight='bold')
    ax.set_ylabel(r'Mean Blocked Area $A$ ($\times 10^6$ km²)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='-')
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)
    
    # Limits
    ax.set_xlim(min(durations1.min(), durations2.min()) - 1, d_pct * 1.05)
    ax.set_ylim(min(areas1.min(), areas2.min()) * 0.9, a_pct * 1.05)
    
    # Annotation
    ax.text(0.02, 0.98, f'{label1}: N={len(durations1)}\n{label2}: N={len(durations2)}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    ax.set_title(f'Duration–Area Comparison ({season})\n'
                 f'{label1} (solid) vs {label2} (dashed)',
                 fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    return fig, ax


def plot_survival_comparison(durations1, durations2, label1='PlaSim', label2='Emulator',
                             season='DJF', figsize=(10, 6), xmax=50):
    """
    Overlay survival curves for two datasets.
    
    Parameters
    ----------
    durations1 : np.ndarray
        Event durations for first dataset
    durations2 : np.ndarray
        Event durations for second dataset
    label1, label2 : str
        Labels for the two datasets
    season : str
        Season label for title
    figsize : tuple
        Figure size
    xmax : int
        Upper x-axis limit in days (default: 50)
    
    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    from matplotlib.ticker import MaxNLocator
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Dataset 1
    N1 = len(durations1)
    D_max1 = int(durations1.max())
    d_values1 = np.arange(1, D_max1 + 1)
    S_values1 = np.array([np.sum(durations1 >= d) / N1 for d in d_values1])
    
    # Dataset 2
    N2 = len(durations2)
    D_max2 = int(durations2.max())
    d_values2 = np.arange(1, D_max2 + 1)
    S_values2 = np.array([np.sum(durations2 >= d) / N2 for d in d_values2])
    
    # Print debug info
    print(f"  {label1}: N={N1}, D_max={D_max1} days")
    print(f"  {label2}: N={N2}, D_max={D_max2} days")
    
    # Colors
    color1, color2 = '#1E88E5', '#E53935'
    
    # Plot survival curves
    ax.plot(d_values1, S_values1, 'o-', color=color1, linewidth=2, markersize=5,
            markerfacecolor='white', markeredgecolor=color1, markeredgewidth=1.5,
            label=f'{label1} (N={N1})')
    ax.plot(d_values2, S_values2, 's--', color=color2, linewidth=2, markersize=5,
            markerfacecolor='white', markeredgecolor=color2, markeredgewidth=1.5,
            label=f'{label2} (N={N2})')
    
    # Log scale on y-axis
    ax.set_yscale('log')
    
    # Configure x-axis
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins='auto'))
    ax.set_xlim(1, xmax)
    
    # Configure y-axis
    S_min = min(S_values1.min(), S_values2.min())
    ax.set_ylim(bottom=max(S_min * 0.5, 1e-3), top=1.2)
    
    # Labels and title
    ax.set_xlabel('Event Duration $d$ (days)', fontsize=12, fontweight='bold')
    ax.set_ylabel(r'Fraction of Events with Duration $\geq d$', fontsize=12, fontweight='bold')
    ax.set_title(f'Blocking Event Duration Survival Curve ({season})\n'
                 f'{label1} vs {label2}',
                 fontsize=13, fontweight='bold')
    
    # Grid
    ax.grid(True, which='major', linestyle='-', alpha=0.4)
    ax.grid(True, which='minor', linestyle=':', alpha=0.2)
    
    # Legend
    ax.legend(loc='lower left', fontsize=11)
    
    # Statistics annotation
    mean1, median1 = durations1.mean(), np.median(durations1)
    mean2, median2 = durations2.mean(), np.median(durations2)
    stats_text = (f'{label1}: Mean={mean1:.1f}d, Median={median1:.0f}d\n'
                  f'{label2}: Mean={mean2:.1f}d, Median={median2:.0f}d')
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, edgecolor='tan'))
    
    plt.tight_layout()
    return fig, ax

