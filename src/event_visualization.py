import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import os
import pickle
import glob
import warnings
import xarray as xr
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from cartopy.util import add_cyclic_point

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
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 0.91, 0.96])
    
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


# =============================================================================
# Extreme Trajectory Visualization (AI-RES experiments)
# =============================================================================

def _auto_detect_K(exp_path):
    """Find the highest resampling step K from thetas_step_*.npy files."""
    pattern = os.path.join(exp_path, "resampling", "thetas_step_*.npy")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(
            f"No resampling/thetas_step_*.npy found under {exp_path}"
        )
    return max(
        int(os.path.basename(f).split("_")[-1].split(".")[0]) for f in files
    )


def _get_ranked_particle(exp_path, K, rank):
    """
    Load thetas at step K, sort descending, return (particle_index, score)
    for the given rank (1 = highest).
    """
    thetas_path = os.path.join(exp_path, "resampling", f"thetas_step_{K}.npy")
    if not os.path.exists(thetas_path):
        raise FileNotFoundError(f"Missing {thetas_path}")
    thetas = np.load(thetas_path)
    if thetas.ndim != 1:
        raise ValueError(
            f"Expected 1-D thetas at {thetas_path}, got shape {thetas.shape}"
        )
    n = len(thetas)
    if rank < 1 or rank > n:
        raise ValueError(
            f"rank={rank} out of range [1, {n}] for {n} particles"
        )
    sorted_indices = np.argsort(thetas)[::-1]  # descending
    idx = int(sorted_indices[rank - 1])
    return idx, float(thetas[idx])


def rank_particles_by_plasim_score(
    exp_path,
    climatology_path="/glade/u/home/zhil/project/AI-RES/Blocking/data/ano_climatology_thresholds.nc",
    K=None,
    top_n=None,
):
    """
    Rank particles at step K by actual PlaSim trajectory score.

    Preferred source: ``resampling/plasim_scores_step_K.npy`` (pre-computed
    by ``compute_plasim_scores.py`` using the experiment's scorer pipeline).
    Fallback: recompute a simple max-anomaly proxy from the ``.nc`` files.

    Parameters
    ----------
    exp_path : str
        Experiment directory path.
    climatology_path : str
        Path to climatology .nc file with z500_climatology(dayofyear, lat, lon).
        Only used for the fallback computation.
    K : int or None
        Resampling step to score at. Auto-detected if None.
    top_n : int or None
        If set, only return the top N particles. None returns all.

    Returns
    -------
    list of (particle_idx, plasim_score) tuples, sorted descending by score.
        Index 0 = highest actual PlaSim score.
    """
    if K is None:
        K = _auto_detect_K(exp_path)

    # --- Try pre-computed scores first -----------------------------------
    saved_path = os.path.join(exp_path, "resampling", f"plasim_scores_step_{K}.npy")
    if os.path.exists(saved_path):
        print(f"Loading saved PlaSim scores from {saved_path}")
        scores = np.load(saved_path)
        plasim_scores = {
            i: float(scores[i])
            for i in range(len(scores))
            if np.isfinite(scores[i])
        }
        ranked = sorted(plasim_scores.items(), key=lambda x: x[1], reverse=True)
        if top_n is not None:
            ranked = ranked[:top_n]
        return ranked

    # --- Fallback: recompute max-anomaly from .nc files ------------------
    print(
        f"No saved PlaSim scores at {saved_path}; "
        "falling back to max-anomaly computation from .nc files"
    )
    clim_ds = xr.open_dataset(climatology_path)
    clim = clim_ds["z500_climatology"]

    # Find all .nc files at step K
    nc_patterns = [
        os.path.join(exp_path, f"step_{K}", "particle_*", "output", "panguplasim_in.*.nc"),
        os.path.join(exp_path, f"step_{K}", "particle_*", "output", "plasim_out.*.nc"),
    ]
    nc_files = []
    for pattern in nc_patterns:
        nc_files.extend(glob.glob(pattern))
    if not nc_files:
        raise FileNotFoundError(
            f"No .nc files found at step_{K}. Available patterns tried: {nc_patterns}"
        )

    # Deduplicate by particle index (prefer panguplasim_in over plasim_out)
    seen = {}
    for nc_path in nc_files:
        particle_idx = int(nc_path.split("particle_")[1].split("/")[0])
        if particle_idx not in seen or "panguplasim_in" in nc_path:
            seen[particle_idx] = nc_path

    plasim_scores = {}
    for particle_idx, nc_path in seen.items():
        try:
            ds = xr.open_dataset(nc_path, use_cftime=True)
            # Handle both Pa and hPa plev units
            if 50000.0 in ds.plev.values:
                z500 = ds["zg"].sel(plev=50000.0)
            elif 500.0 in ds.plev.values:
                z500 = ds["zg"].sel(plev=500.0)
            else:
                ds.close()
                continue
            z500_daily = z500.resample(time="1D").mean()
            # Compute max anomaly across all days
            max_anom = 0.0
            for t in range(len(z500_daily.time)):
                cft = z500_daily.time.values[t]
                doy = cft.timetuple().tm_yday
                doy = min(doy, int(clim.dayofyear.max()))
                doy = max(doy, int(clim.dayofyear.min()))
                clim_slice = clim.sel(dayofyear=doy).interp(
                    lat=z500_daily.lat, lon=z500_daily.lon, method="linear",
                )
                anom = z500_daily.isel(time=t).values - clim_slice.values
                max_anom = max(max_anom, float(np.nanmax(anom)))
            plasim_scores[particle_idx] = max_anom
            ds.close()
        except Exception as e:
            warnings.warn(f"Could not score particle {particle_idx}: {e}")

    clim_ds.close()

    ranked = sorted(plasim_scores.items(), key=lambda x: x[1], reverse=True)
    if top_n is not None:
        ranked = ranked[:top_n]
    return ranked


def _trace_lineage(exp_path, K, particle_idx):
    """
    Load working_tree.pkl and walk parent pointers from the given node
    back to the root.  Returns a list of dicts [{'step': k, 'particle': m}, ...]
    ordered from step 0 to step K.
    """
    from anytree.search import findall

    tree_path = os.path.join(exp_path, "working_tree.pkl")
    with open(tree_path, "rb") as f:
        working_tree = pickle.load(f)

    name = f"step_{K}_particle_{particle_idx}"
    matches = findall(working_tree, filter_=lambda node: node.name == name)
    if len(matches) != 1:
        raise ValueError(
            f"Expected one node named '{name}' in working tree, found {len(matches)}"
        )
    node = matches[0]

    lineage = []
    while node.name != "root":
        parts = node.name.split("_")
        lineage.append({"step": int(parts[1]), "particle": int(parts[3])})
        node = node.parent
    lineage.reverse()
    return lineage


def visualize_extreme_trajectory(
    exp_path,
    viz_type,
    extreme_rank,
    climatology_path,
    K=None,
    steps_to_show=None,
    blocking_overlay=False,
    save_path=None,
    fps=2,
    dpi=100,
    figsize=None,
):
    """
    Visualize the Z500 anomaly trajectory of an extreme-scoring particle
    through its full genealogy in an AI-RES experiment.

    Parameters
    ----------
    exp_path : str
        Experiment directory (contains working_tree.pkl, resampling/, step_*/).
    viz_type : str
        'animation' for GIF or 'image' for 2x4 panel PNG.
    extreme_rank : int
        Rank of the particle to visualize (1 = highest score).
    climatology_path : str
        Path to .nc file containing z500_climatology(dayofyear, lat, lon).
    K : int, optional
        Final resampling step. Auto-detected from thetas files if None.
    steps_to_show : list[int], optional
        Subset of steps to include. Default: all steps in the lineage.
    blocking_overlay : bool
        If True, overlay ANO blocking contours (orange dashed).
    save_path : str, optional
        Output file path (.gif or .png). Auto-generated if None.
    fps : int
        Frames per second for animation (default 2).
    dpi : int
        Output resolution (default 100).
    figsize : tuple, optional
        Figure size. Defaults to (12, 10) for animation, (24, 12) for image.

    Returns
    -------
    str
        Path to the saved file.
    """
    # ── Validate inputs ──────────────────────────────────────────────────
    if viz_type not in ("animation", "image"):
        raise ValueError(f"viz_type must be 'animation' or 'image', got '{viz_type}'")
    if figsize is None:
        figsize = (12, 10) if viz_type == "animation" else (24, 12)

    # ── Auto-detect K ────────────────────────────────────────────────────
    if K is None:
        K = _auto_detect_K(exp_path)
    print(f"Using resampling step K={K}")

    # ── Rank and select particle (by actual PlaSim score) ──────────────
    ranked = rank_particles_by_plasim_score(exp_path, climatology_path, K=K)
    if extreme_rank < 1 or extreme_rank > len(ranked):
        raise ValueError(
            f"extreme_rank={extreme_rank} out of range [1, {len(ranked)}]"
        )
    particle_idx, score = ranked[extreme_rank - 1]
    print(f"Rank {extreme_rank}: particle {particle_idx}, PlaSim score {score:.6f}")

    # ── Trace genealogy ──────────────────────────────────────────────────
    lineage = _trace_lineage(exp_path, K, particle_idx)
    print(f"Lineage ({len(lineage)} steps): "
          + " -> ".join(f"s{e['step']}p{e['particle']}" for e in lineage))

    # ── Filter by steps_to_show ──────────────────────────────────────────
    if steps_to_show is not None:
        steps_set = set(steps_to_show)
        lineage = [e for e in lineage if e["step"] in steps_set]
        if not lineage:
            raise ValueError(
                f"No lineage entries match steps_to_show={steps_to_show}"
            )
        print(f"Filtered to {len(lineage)} steps: {[e['step'] for e in lineage]}")

    # ── Load Z500 trajectory ─────────────────────────────────────────────
    datasets = []
    for entry in lineage:
        k, m = entry["step"], entry["particle"]
        nc_path = os.path.join(
            exp_path, f"step_{k}", f"particle_{m}", "output",
            f"panguplasim_in.step_{k}.particle_{m}.nc",
        )
        if not os.path.exists(nc_path):
            # Fallback: re-simulated PlaSim output (from resimulate_lineage.py)
            nc_path = os.path.join(
                exp_path, f"step_{k}", f"particle_{m}", "output",
                f"plasim_out.step_{k}.particle_{m}.nc",
            )
        if not os.path.exists(nc_path):
            warnings.warn(f"Missing file, skipping: {nc_path}")
            continue
        ds = xr.open_dataset(nc_path, use_cftime=True)
        # Select 500 hPa level — handle both Pa (50000) and hPa (500) units
        if 50000.0 in ds.plev.values:
            z500 = ds["zg"].sel(plev=50000.0)
        elif 500.0 in ds.plev.values:
            z500 = ds["zg"].sel(plev=500.0)
        else:
            raise KeyError(
                f"Cannot find 500 hPa level. Available plev values: {ds.plev.values}"
            )
        # Tag each timestep with its lineage step
        z500 = z500.assign_coords(lineage_step=("time", [k] * len(z500.time)))
        datasets.append(z500)
        ds.close()

    if not datasets:
        raise FileNotFoundError(
            "No panguplasim_in files found for any lineage entry. "
            "Check that step directories exist on disk."
        )

    z500_traj = xr.concat(datasets, dim="time")
    n_total = len(z500_traj.time)
    print(f"Loaded {n_total} timesteps ({len(datasets)} files)")

    # ── Standardize coordinates (NH, lat ascending, 0-360, cyclic) ───────
    # Subset to NH
    z500_traj = z500_traj.sel(lat=z500_traj.lat[z500_traj.lat >= 0])

    # Ensure lat ascending
    if len(z500_traj.lat) > 1 and float(z500_traj.lat[0]) > float(z500_traj.lat[-1]):
        z500_traj = z500_traj.sortby("lat")

    # Ensure lon 0-360
    lon_vals = z500_traj.lon.values
    if np.any(lon_vals < 0):
        lon_vals_adj = np.where(lon_vals < 0, lon_vals + 360, lon_vals)
        z500_traj = z500_traj.assign_coords(lon=lon_vals_adj).sortby("lon")

    lat = z500_traj.lat.values
    lon = z500_traj.lon.values

    # Add cyclic point for plotting
    z500_vals = z500_traj.values  # (time, lat, lon)
    z500_cyc, lon_cyc = add_cyclic_point(z500_vals[0], coord=lon)
    # We only needed lon_cyc; build full cyclic array
    z500_all_cyc = np.empty((n_total, len(lat), len(lon_cyc)))
    for t in range(n_total):
        z500_all_cyc[t], _ = add_cyclic_point(z500_vals[t], coord=lon)

    # ── Compute anomalies ────────────────────────────────────────────────
    clim_ds = xr.open_dataset(climatology_path)
    clim = clim_ds["z500_climatology"]  # (dayofyear, lat, lon)

    # Build anomaly array
    z500_anom_cyc = np.empty_like(z500_all_cyc)
    time_coords = z500_traj.time.values

    for t in range(n_total):
        # Extract day-of-year from cftime
        cft = time_coords[t]
        doy = cft.timetuple().tm_yday
        # Clamp to available range in climatology
        doy = min(doy, int(clim.dayofyear.max()))
        doy = max(doy, int(clim.dayofyear.min()))

        clim_slice = clim.sel(dayofyear=doy)  # (lat, lon) on climatology grid

        # Interpolate climatology to trajectory grid if shapes differ
        if clim_slice.shape != (len(lat), len(lon)):
            clim_interp = clim_slice.interp(
                lat=xr.DataArray(lat, dims="lat"),
                lon=xr.DataArray(lon, dims="lon"),
                method="linear",
            ).values
        else:
            clim_interp = clim_slice.values

        # Add cyclic point to climatology slice
        clim_cyc, _ = add_cyclic_point(clim_interp, coord=lon)
        z500_anom_cyc[t] = z500_all_cyc[t] - clim_cyc

    clim_ds.close()

    # ── Optional blocking detection ──────────────────────────────────────
    blocked_cyc = None
    if blocking_overlay:
        try:
            from ANO_PlaSim import create_blocking_mask_fast, identify_blocking_events

            # Need daily means for blocking detection (data is 6-hourly, 4 per day)
            # Work on the non-cyclic data in xarray for proper time handling
            z500_anom_da = xr.DataArray(
                z500_anom_cyc[:, :, :-1],  # drop cyclic point for detection
                dims=["time", "lat", "lon"],
                coords={"time": z500_traj.time, "lat": lat, "lon": lon},
            )
            z500_daily = z500_anom_da.resample(time="1D").mean()

            # Load threshold from climatology file
            clim_ds2 = xr.open_dataset(climatology_path)
            if "threshold_90" in clim_ds2:
                threshold = float(clim_ds2["threshold_90"].values)
            else:
                # Estimate from data
                threshold = float(np.nanpercentile(z500_daily.values, 90))
            clim_ds2.close()

            blocked_mask = create_blocking_mask_fast(z500_daily, threshold)
            # Expand daily mask back to 6-hourly for overlay
            n_daily = len(blocked_mask.time)
            blocked_vals = blocked_mask.values  # (n_daily, lat, lon)
            blocked_6h = np.repeat(blocked_vals, 4, axis=0)[:n_total]
            # Add cyclic point
            blocked_cyc = np.empty((n_total, len(lat), len(lon_cyc)))
            for t in range(n_total):
                blocked_cyc[t], _ = add_cyclic_point(blocked_6h[t], coord=lon)

            print("Blocking overlay computed successfully")
        except Exception as e:
            warnings.warn(f"Could not compute blocking overlay: {e}")
            blocked_cyc = None

    # ── Resample to daily means ──────────────────────────────────────────
    lineage_steps_6h = z500_traj.coords["lineage_step"].values

    # Group 6-hourly timesteps by (year, month, day) and average
    from collections import OrderedDict
    daily_groups = OrderedDict()  # key=(y,m,d) -> list of 6h indices
    for t in range(n_total):
        cft = time_coords[t]
        day_key = (cft.year, cft.month, cft.day)
        daily_groups.setdefault(day_key, []).append(t)

    n_daily = len(daily_groups)
    z500_anom_daily_cyc = np.empty((n_daily, len(lat), len(lon_cyc)))
    daily_lineage_steps = np.empty(n_daily, dtype=int)
    daily_time_coords = []
    if blocked_cyc is not None:
        blocked_daily_cyc = np.empty((n_daily, len(lat), len(lon_cyc)))
    else:
        blocked_daily_cyc = None

    for d, (day_key, indices) in enumerate(daily_groups.items()):
        z500_anom_daily_cyc[d] = np.mean(z500_anom_cyc[indices], axis=0)
        daily_lineage_steps[d] = int(lineage_steps_6h[indices[-1]])  # step of last 6h in the day
        daily_time_coords.append(time_coords[indices[0]])  # use first 6h time for the date label
        if blocked_daily_cyc is not None:
            blocked_daily_cyc[d] = np.mean(blocked_cyc[indices], axis=0)

    # Replace 6-hourly arrays with daily arrays for all downstream plotting
    z500_anom_cyc = z500_anom_daily_cyc
    blocked_cyc = blocked_daily_cyc
    lineage_steps = daily_lineage_steps
    time_coords = daily_time_coords
    n_total = n_daily
    print(f"Resampled to {n_daily} daily-mean frames")

    # ── Build frame metadata ─────────────────────────────────────────────
    frame_info = []
    for t in range(n_total):
        cft = time_coords[t]
        date_str = f"{cft.year:04d}-{cft.month:02d}-{cft.day:02d}"
        frame_info.append({
            "step": int(lineage_steps[t]),
            "date": date_str,
            "frame": t,
        })

    # ── Default save path ────────────────────────────────────────────────
    if save_path is None:
        ext = "gif" if viz_type == "animation" else "png"
        save_path = os.path.join(
            exp_path,
            f"extreme_rank{extreme_rank}_K{K}.{ext}",
        )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # ── Shared plotting parameters ───────────────────────────────────────
    levels_anom = np.arange(-300, 301, 30)
    # Compute global max for the black-shading range
    global_anom_max = float(np.nanmax(z500_anom_cyc))
    print(f"Colorbar range: ±300 m  (data max anom = {global_anom_max:.0f} m)")

    # Black-shading levels for anomalies > 300 m.
    # Each band gets progressively more opaque black, so the viewer can
    # distinguish 320 m from 540 m at a glance.
    _SHADE_FLOOR = 300  # start of black overlay
    _N_SHADE_BANDS = 8  # number of opacity bands
    _ALPHA_MIN = 0.10   # lightest black band (just above 300 m)
    _ALPHA_MAX = 0.85   # darkest black band (at the data maximum)

    if global_anom_max > _SHADE_FLOOR:
        shade_ceil = int(np.ceil(global_anom_max / 50) * 50)  # round up to nearest 50
        shade_levels = np.linspace(_SHADE_FLOOR, shade_ceil, _N_SHADE_BANDS + 1)
        shade_alphas = np.linspace(_ALPHA_MIN, _ALPHA_MAX, _N_SHADE_BANDS)
        shade_colors = [(0, 0, 0, a) for a in shade_alphas]
        print(f"Black shading: {_SHADE_FLOOR}–{shade_ceil} m "
              f"({_N_SHADE_BANDS} bands, alpha {_ALPHA_MIN}–{_ALPHA_MAX})")
    else:
        shade_levels = None  # nothing to shade

    # Legend
    from matplotlib.patches import Patch
    legend_handles = []
    if shade_levels is not None:
        legend_handles.append(
            Patch(facecolor=(0, 0, 0, 0.5), edgecolor="black",
                  label=f"> {_SHADE_FLOOR} m (black shading)")
        )
    if blocked_cyc is not None:
        legend_handles.append(
            Line2D([0], [0], color="orange", lw=1.5, ls="--", label="Blocked region")
        )

    def _add_extreme_shading(ax, data_2d):
        """Overlay graduated black shading for anomalies exceeding 300 m."""
        if shade_levels is None:
            return
        if float(np.nanmax(data_2d)) <= _SHADE_FLOOR:
            return
        ax.contourf(
            lon_cyc, lat, data_2d,
            levels=shade_levels, colors=shade_colors,
            transform=ccrs.PlateCarree(), extend="max",
        )

    def _add_peak_annotation(ax, data_2d):
        """Annotate the map with the peak anomaly value at its location."""
        peak_val = float(np.nanmax(data_2d))
        if peak_val < 100:
            return  # skip annotation for weak anomalies
        peak_idx = np.unravel_index(np.nanargmax(data_2d), data_2d.shape)
        peak_lat = lat[peak_idx[0]]
        peak_lon = lon_cyc[peak_idx[1]]
        ax.text(
            peak_lon, peak_lat, f"{peak_val:.0f} m",
            transform=ccrs.PlateCarree(),
            fontsize=15, fontweight="bold", color="white",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="darkred", alpha=0.8),
            zorder=10,
        )

    # ══════════════════════════════════════════════════════════════════════
    #  ANIMATION
    # ══════════════════════════════════════════════════════════════════════
    if viz_type == "animation":
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.NorthPolarStereo())
        ax.set_extent([-180, 180, 30, 90], ccrs.PlateCarree())
        ax.coastlines(linewidth=0.5, color="black")
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
        ax.gridlines(draw_labels=False, linewidth=0.5, color="gray",
                     alpha=0.3, linestyle="-")

        # Initial frame
        cf = ax.contourf(lon_cyc, lat, z500_anom_cyc[0],
                         levels=levels_anom, cmap="RdBu_r",
                         transform=ccrs.PlateCarree(), alpha=0.6, extend="both")
        # Overlay graduated black shading for extreme positive anomalies
        _add_extreme_shading(ax, z500_anom_cyc[0])
        # Annotate peak anomaly value
        _add_peak_annotation(ax, z500_anom_cyc[0])

        cbar = fig.colorbar(cf, ax=ax, orientation="horizontal",
                            pad=0.05, shrink=0.8, label="Z500 Anomaly (m)")
        cbar.set_label("Z500 Anomaly (m)", fontsize=15)
        cbar.ax.tick_params(labelsize=15)

        info0 = frame_info[0]
        title_text = (
            f"Rank {extreme_rank} (score={score:.4f})\n"
            f"Step {info0['step']} | {info0['date']} | "
            f"Frame 1/{n_total}"
        )
        title_obj = ax.set_title(title_text, fontsize=18, fontweight="bold", pad=10)
        ax.legend(handles=legend_handles, loc="lower left", fontsize=14, framealpha=0.9)

        def update(frame):
            # Clear collections and text annotations from previous frame
            for coll in list(ax.collections):
                coll.remove()
            for txt in list(ax.texts):
                txt.remove()

            ax.contourf(lon_cyc, lat, z500_anom_cyc[frame],
                        levels=levels_anom, cmap="RdBu_r",
                        transform=ccrs.PlateCarree(), alpha=0.6, extend="both")
            # Overlay graduated black shading for extreme positive anomalies
            _add_extreme_shading(ax, z500_anom_cyc[frame])
            # Annotate peak anomaly value
            _add_peak_annotation(ax, z500_anom_cyc[frame])

            if blocked_cyc is not None and blocked_cyc[frame].any():
                ax.contour(lon_cyc, lat, blocked_cyc[frame],
                           levels=[0.5], colors="orange", linewidths=1.5,
                           linestyles="--", transform=ccrs.PlateCarree(), alpha=0.5)

            ax.coastlines(linewidth=0.5, color="black")
            ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)

            info = frame_info[frame]
            title_obj.set_text(
                f"Rank {extreme_rank} (score={score:.4f})\n"
                f"Step {info['step']} | {info['date']} | "
                f"Frame {frame + 1}/{n_total}"
            )
            if (frame + 1) % max(1, n_total // 10) == 0:
                print(f"  Frame {frame + 1}/{n_total}")
            return ax.collections + list(ax.texts) + [title_obj]

        print(f"Generating {n_total}-frame animation...")
        anim = FuncAnimation(fig, update, frames=n_total,
                             interval=1000 / fps, blit=False, repeat=True)
        writer = PillowWriter(fps=fps)
        anim.save(save_path, writer=writer, dpi=dpi)
        plt.close(fig)

        size_mb = os.path.getsize(save_path) / (1024 * 1024)
        print(f"Saved animation: {save_path} ({size_mb:.1f} MB, "
              f"{n_total} frames, {n_total / fps:.1f}s at {fps} FPS)")

    # ══════════════════════════════════════════════════════════════════════
    #  STATIC IMAGE (2x4 panels)
    # ══════════════════════════════════════════════════════════════════════
    else:
        n_panels = 8
        # Pick one representative frame per unique lineage step (last timestep
        # of each step) so that every panel shows a distinct step.
        unique_steps = []
        seen_steps = set()
        for t in range(n_total):
            s = int(lineage_steps[t])
            if s not in seen_steps:
                seen_steps.add(s)
                unique_steps.append(s)
        # For each unique step, take its last timestep as the representative
        step_last_frame = {}
        for t in range(n_total):
            step_last_frame[int(lineage_steps[t])] = t
        representative_frames = [step_last_frame[s] for s in unique_steps]
        # If more steps than panels, subsample evenly; if fewer, use all
        if len(representative_frames) > n_panels:
            pick = np.linspace(0, len(representative_frames) - 1, n_panels, dtype=int)
            panel_indices = [representative_frames[i] for i in pick]
        else:
            panel_indices = representative_frames
        n_panels = len(panel_indices)

        n_cols = min(4, n_panels)
        n_rows = int(np.ceil(n_panels / n_cols))
        fig = plt.figure(figsize=figsize)
        for subplot_idx, t_idx in enumerate(panel_indices, start=1):
            ax = fig.add_subplot(n_rows, n_cols, subplot_idx,
                                 projection=ccrs.NorthPolarStereo())
            ax.set_extent([-180, 180, 30, 90], ccrs.PlateCarree())

            cf = ax.contourf(lon_cyc, lat, z500_anom_cyc[t_idx],
                             levels=levels_anom, cmap="RdBu_r",
                             transform=ccrs.PlateCarree(), alpha=0.6, extend="both")
            # Overlay graduated black shading for extreme positive anomalies
            _add_extreme_shading(ax, z500_anom_cyc[t_idx])
            # Annotate peak anomaly value
            _add_peak_annotation(ax, z500_anom_cyc[t_idx])

            if blocked_cyc is not None and blocked_cyc[t_idx].any():
                ax.contour(lon_cyc, lat, blocked_cyc[t_idx],
                           levels=[0.5], colors="orange", linewidths=1.5,
                           linestyles="--", transform=ccrs.PlateCarree(), alpha=0.5)

            ax.coastlines(linewidth=0.5, color="black")
            ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.5)
            ax.gridlines(draw_labels=False, linewidth=0.5,
                         color="gray", alpha=0.3, linestyle="-")

            info = frame_info[t_idx]
            ax.set_title(
                f"Step {info['step']}\n{info['date']}",
                fontsize=17, fontweight="bold",
            )
            if subplot_idx == 1:
                ax.legend(handles=legend_handles, loc="lower left", fontsize=12, framealpha=0.9)

        # Colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        cbar = fig.colorbar(cf, cax=cbar_ax, label="Z500 Anomaly (m)")
        cbar.set_label("Z500 Anomaly (m)", fontsize=15)
        cbar.ax.tick_params(labelsize=15)

        # Suptitle — raised above the subplot area to avoid overlapping panel titles
        fig.suptitle(
            f"Extreme Trajectory — Rank {extreme_rank} (score={score:.4f})",
            fontsize=21, fontweight="bold", y=1.02,
        )
        # Use explicit spacing because the manual colorbar axis is incompatible
        # with tight_layout in some Matplotlib versions.
        plt.subplots_adjust(
            left=0.04, right=0.90, bottom=0.06, top=0.88, hspace=0.22, wspace=0.10
        )

        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved image: {save_path}")

    return save_path
