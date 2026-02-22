#!/usr/bin/env python3
"""
Shared Utilities for ANO-based Blocking Scorers.

This module extracts common utility functions used by ANO-based scorers,
including centroid computation, event selection, block tracking, and
integration helpers.

Author: AI-RES Project
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr
from scipy import ndimage


# ---------------------------------------------------------------------------
# Region Bounds
# ---------------------------------------------------------------------------

# Predefined region boundaries (lon_min, lon_max, lat_min, lat_max)
REGION_BOUNDS = {
    "Eurasia": (30.0, 100.0, 55.0, 75.0),
    "NorthAtlantic": (-59.0625, -2.8125, 57.20397269, 73.94716987),
}


# ---------------------------------------------------------------------------
# Spatial Weight Mask
# ---------------------------------------------------------------------------

def create_spatial_weight_mask(
    ds_coords: xr.DataArray,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float
) -> xr.DataArray:
    """
    Construct the 2D spatial weight mask W(lambda, phi).

    W = 1 if lon in [lon_min, lon_max] and lat in [lat_min, lat_max], else 0.
    """
    mask = (
        (ds_coords.lat >= lat_min) &
        (ds_coords.lat <= lat_max) &
        (ds_coords.lon >= lon_min) &
        (ds_coords.lon <= lon_max)
    )
    return mask.astype(float)


# ---------------------------------------------------------------------------
# ROI Helpers
# ---------------------------------------------------------------------------

def _centroid_in_roi(
    lon: float,
    lat: float,
    region_lon_min: float,
    region_lon_max: float,
    region_lat_min: float,
    region_lat_max: float,
) -> bool:
    """Check whether a centroid (lon, lat) lies inside the target ROI."""
    if not (region_lat_min <= lat <= region_lat_max):
        return False
    if region_lon_min < 0:
        lon_min_360 = region_lon_min % 360
        lon_max_360 = region_lon_max % 360
        if lon_min_360 > lon_max_360:
            return lon >= lon_min_360 or lon <= lon_max_360
        else:
            return lon_min_360 <= lon <= lon_max_360
    else:
        return region_lon_min <= lon <= region_lon_max


def _build_roi_mask(
    LON_grid: np.ndarray,
    LAT_grid: np.ndarray,
    region_lon_min: float,
    region_lon_max: float,
    region_lat_min: float,
    region_lat_max: float,
) -> np.ndarray:
    """Build a boolean (lat, lon) mask for the target ROI."""
    if region_lon_min < 0:
        lon_min_360 = region_lon_min % 360
        lon_max_360 = region_lon_max % 360
        if lon_min_360 > lon_max_360:
            roi_lon_mask = (LON_grid >= lon_min_360) | (LON_grid <= lon_max_360)
        else:
            roi_lon_mask = (LON_grid >= lon_min_360) & (LON_grid <= lon_max_360)
    else:
        roi_lon_mask = (LON_grid >= region_lon_min) & (LON_grid <= region_lon_max)
    roi_lat_mask = (LAT_grid >= region_lat_min) & (LAT_grid <= region_lat_max)
    return roi_lon_mask & roi_lat_mask


# ---------------------------------------------------------------------------
# Event / Block Selection
# ---------------------------------------------------------------------------

def select_event_at_onset(
    event_mask: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    onset_time_idx: int,
    region_lon_min: float,
    region_lon_max: float,
    region_lat_min: float,
    region_lat_max: float,
    n_days: int = 5,
) -> Optional[Tuple[int, Dict]]:
    """
    Select a blocking event from the ANO event_mask at onset time.

    Uses the same centroid-in-ROI + largest-area logic as select_block_at_onset,
    but operates on the labeled event_mask produced by
    ANO_PlaSim.identify_blocking_events (which already enforces area, duration,
    and overlap criteria).

    The selected event must also persist through all *n_days* timesteps
    starting from *onset_time_idx*.  If it does not, None is returned so the
    caller can assign B_int = 0.

    Parameters
    ----------
    event_mask : np.ndarray
        3-D array (time, lat, lon) of integer event IDs from
        identify_blocking_events.  0 = no event.
    lats, lons : np.ndarray
        1-D coordinate arrays.
    onset_time_idx : int
        Time index of blocking onset.
    region_lon_min, region_lon_max : float
        Longitude bounds for the target ROI.
    region_lat_min, region_lat_max : float
        Latitude bounds for the target ROI.
    n_days : int
        Number of integration days (default 5).

    Returns
    -------
    (event_id, info) or None
        event_id : int  - the selected event ID
        info : dict     - centroid_lon, centroid_lat, size
        Returns None when no qualifying event is found or the event does
        not persist through the integration window.
    """
    # Time bounds check
    if onset_time_idx + n_days - 1 >= event_mask.shape[0]:
        return None

    mask_t = event_mask[onset_time_idx]
    event_ids = np.unique(mask_t)
    event_ids = event_ids[event_ids > 0]

    if len(event_ids) == 0:
        return None

    LON_grid, LAT_grid = np.meshgrid(lons, lats)
    cos_lat_weights = np.cos(np.deg2rad(lats))
    COS_LAT_grid = np.broadcast_to(cos_lat_weights[:, None], (len(lats), len(lons)))

    roi_mask = _build_roi_mask(
        LON_grid, LAT_grid,
        region_lon_min, region_lon_max,
        region_lat_min, region_lat_max,
    )

    candidates = []
    for eid in event_ids:
        comp_mask = (mask_t == eid)
        comp_lons = LON_grid[comp_mask]
        comp_lats = LAT_grid[comp_mask]
        comp_weights = COS_LAT_grid[comp_mask]

        total_weight = np.sum(comp_weights)
        centroid_lon = np.sum(comp_weights * comp_lons) / total_weight
        centroid_lat = np.sum(comp_weights * comp_lats) / total_weight

        if not _centroid_in_roi(
            centroid_lon, centroid_lat,
            region_lon_min, region_lon_max,
            region_lat_min, region_lat_max,
        ):
            continue

        # Area overlap with ROI (used for tie-breaking)
        overlap_weights = COS_LAT_grid[comp_mask & roi_mask]
        roi_overlap = np.sum(overlap_weights) if overlap_weights.size > 0 else 0.0

        candidates.append({
            'event_id': int(eid),
            'centroid_lon': float(centroid_lon),
            'centroid_lat': float(centroid_lat),
            'size': float(total_weight),
            'roi_overlap': float(roi_overlap),
        })

    if len(candidates) == 0:
        return None

    # Select the candidate with largest ROI overlap
    selected = max(candidates, key=lambda c: c['roi_overlap'])

    # Verify persistence: the event must be present at every timestep in
    # [onset_time_idx, onset_time_idx + n_days)
    eid = selected['event_id']
    for offset in range(n_days):
        t = onset_time_idx + offset
        if not np.any(event_mask[t] == eid):
            # Event does not span the full integration window -> score = 0
            return None

    return eid, {
        'centroid_lon': selected['centroid_lon'],
        'centroid_lat': selected['centroid_lat'],
        'size': selected['size'],
    }


def select_block_at_onset(
    blocked_mask: xr.DataArray,
    onset_time_idx: int,
    region_lon_min: float,
    region_lon_max: float,
    region_lat_min: float,
    region_lat_max: float,
    nh_lat_min: float = 30.0,
) -> Optional[Tuple[np.ndarray, Dict]]:
    """
    Select the blocking object at onset time whose centroid lies within the
    target ROI.  If multiple blocks have centroids in the ROI, select the one
    with largest area.

    Returns None (instead of raising) when no qualifying block is found so
    that the caller can assign B_int = 0.

    Parameters
    ----------
    blocked_mask : xr.DataArray
        Binary blocking mask with dimensions (time, lat, lon).
    onset_time_idx : int
        Time index of blocking onset.
    region_lon_min, region_lon_max : float
        Longitude bounds for the target region of interest.
    region_lat_min, region_lat_max : float
        Latitude bounds for the target region of interest.
    nh_lat_min : float, optional
        Minimum latitude for Northern Hemisphere.  Default: 30.0 N.

    Returns
    -------
    (component_mask, component_info) or None
    """
    mask_t = blocked_mask.isel(time=onset_time_idx).values.copy()
    lats = blocked_mask.lat.values
    lons = blocked_mask.lon.values

    LON_grid, LAT_grid = np.meshgrid(lons, lats)
    cos_lat_weights = np.cos(np.deg2rad(lats))
    COS_LAT_grid = np.broadcast_to(cos_lat_weights[:, None], (len(lats), len(lons)))

    nh_mask = LAT_grid >= nh_lat_min
    mask_nh = mask_t * nh_mask

    if mask_nh.sum() == 0:
        return None

    labeled, num_features = ndimage.label(mask_nh)
    if num_features == 0:
        return None

    roi_mask = _build_roi_mask(
        LON_grid, LAT_grid,
        region_lon_min, region_lon_max,
        region_lat_min, region_lat_max,
    )

    components = []
    for label_id in range(1, num_features + 1):
        component_mask = (labeled == label_id)
        comp_lons = LON_grid[component_mask]
        comp_lats = LAT_grid[component_mask]
        comp_weights = COS_LAT_grid[component_mask]

        total_weight = np.sum(comp_weights)
        centroid_lon = np.sum(comp_weights * comp_lons) / total_weight
        centroid_lat = np.sum(comp_weights * comp_lats) / total_weight

        overlap_weights = COS_LAT_grid[component_mask & roi_mask]
        roi_overlap = np.sum(overlap_weights) if overlap_weights.size > 0 else 0.0

        components.append({
            'label': label_id,
            'centroid_lon': centroid_lon,
            'centroid_lat': centroid_lat,
            'size': total_weight,
            'roi_overlap': roi_overlap,
            'mask': component_mask,
        })

    components_in_roi = [
        c for c in components
        if _centroid_in_roi(
            c['centroid_lon'], c['centroid_lat'],
            region_lon_min, region_lon_max,
            region_lat_min, region_lat_max,
        )
    ]

    if len(components_in_roi) == 0:
        return None

    selected = max(components_in_roi, key=lambda c: c['roi_overlap'])
    return selected['mask'], {
        'centroid_lon': selected['centroid_lon'],
        'centroid_lat': selected['centroid_lat'],
        'size': selected['size'],
        'label': selected['label'],
    }


def track_block_through_time(
    blocked_mask: xr.DataArray,
    onset_time_idx: int,
    n_days: int,
    initial_component_mask: np.ndarray,
    nh_lat_min: float = 30.0,
) -> Optional[List[np.ndarray]]:
    """
    Track a blocking object through time using spatial overlap matching.

    Returns None (instead of raising) when the block dissipates before
    *n_days* so the caller can assign B_int = 0.

    Parameters
    ----------
    blocked_mask : xr.DataArray
        Binary blocking mask with dimensions (time, lat, lon).
    onset_time_idx : int
        Time index of blocking onset.
    n_days : int
        Number of days to track (including onset day).
    initial_component_mask : np.ndarray
        Binary mask (lat, lon) of the component at onset time.
    nh_lat_min : float, optional
        Minimum latitude for Northern Hemisphere.  Default: 30.0 N.

    Returns
    -------
    tracked_masks : list of np.ndarray, or None
        List of binary masks for each timestep, or None if the block
        dissipates or data runs out before n_days.
    """
    lats = blocked_mask.lat.values
    lons = blocked_mask.lon.values
    LON_grid, LAT_grid = np.meshgrid(lons, lats)

    cos_lat_weights = np.cos(np.deg2rad(lats))
    COS_LAT_grid = np.broadcast_to(cos_lat_weights[:, None], (len(lats), len(lons)))

    nh_mask = LAT_grid >= nh_lat_min

    tracked_masks = [initial_component_mask]
    prev_mask = initial_component_mask

    for day_offset in range(1, n_days):
        time_idx = onset_time_idx + day_offset

        if time_idx >= len(blocked_mask.time):
            return None

        mask_t = blocked_mask.isel(time=time_idx).values.copy()
        mask_nh = mask_t * nh_mask

        if mask_nh.sum() == 0:
            return None

        labeled, num_features = ndimage.label(mask_nh)
        if num_features == 0:
            return None

        best_overlap = 0.0
        best_mask = None

        for label_id in range(1, num_features + 1):
            component_mask = (labeled == label_id)
            overlap = prev_mask & component_mask
            overlap_weight = np.sum(COS_LAT_grid[overlap])

            if overlap_weight > best_overlap:
                best_overlap = overlap_weight
                best_mask = component_mask

        if best_mask is None or best_overlap == 0:
            return None

        tracked_masks.append(best_mask)
        prev_mask = best_mask

    return tracked_masks


# ---------------------------------------------------------------------------
# Integration Helpers
# ---------------------------------------------------------------------------

def _integrate_over_masks(
    z500_anom: xr.DataArray,
    tracked_masks: List[np.ndarray],
    onset_time_idx: int,
) -> Tuple[float, List[float]]:
    """
    Compute B_int by integrating Z'_+ * cos(lat) over the tracked masks.

    Returns (B_int, daily_contributions).
    """
    lats = z500_anom.lat.values
    cos_lat_weights = np.cos(np.deg2rad(lats))

    B_int = 0.0
    daily_contributions = []

    for day_offset, component_mask in enumerate(tracked_masks):
        time_idx = onset_time_idx + day_offset
        z_anom_t = z500_anom.isel(time=time_idx).values
        z_plus = np.maximum(z_anom_t, 0)
        integrand = component_mask * z_plus * cos_lat_weights[:, None]
        daily_value = float(np.sum(integrand))
        daily_contributions.append(daily_value)
        B_int += daily_value

    return float(B_int), daily_contributions


# ---------------------------------------------------------------------------
# Centroid Computation (from rmse_scorer)
# ---------------------------------------------------------------------------

def compute_blocking_centroid(
    blocked_mask: xr.DataArray,
    time_idx: int,
    region_lon_min: Optional[float] = None,
    region_lon_max: Optional[float] = None,
    region_lat_min: Optional[float] = None,
    region_lat_max: Optional[float] = None,
    nh_lat_min: float = 30.0,
) -> Tuple[float, float]:
    """
    Compute the area-weighted centroid of blocking at a specific time.

    Algorithm:
    1. Identify ALL blocking objects over the Northern Hemisphere using
       connected-component labeling on the full NH blocked_mask (no ROI masking)
    2. For each connected component, compute its centroid using ALL grid points
       in that component (not truncated to ROI) and its size (area-weighted)
    3. Select components whose centroid lies inside the target ROI
    4. If multiple centroids are in ROI, choose the largest component
    5. Return the centroid of that selected component

    The centroid is weighted by cos(lat) to account for grid cell area variation.

    Parameters
    ----------
    blocked_mask : xr.DataArray
        Binary blocking mask with dimensions (time, lat, lon).
    time_idx : int
        Time index at which to compute the centroid.
    region_lon_min, region_lon_max : float, optional
        Longitude bounds for target region of interest. If None, uses NH.
    region_lat_min, region_lat_max : float, optional
        Latitude bounds for target region of interest. If None, uses NH.
    nh_lat_min : float, optional
        Minimum latitude for Northern Hemisphere. Default: 30.0 N.

    Returns
    -------
    lon_centroid : float
        Longitude of the centroid (degrees).
    lat_centroid : float
        Latitude of the centroid (degrees).

    Raises
    ------
    ValueError
        If no blocking objects have their centroid within the target ROI.
    """
    mask_t = blocked_mask.isel(time=time_idx).values.copy()
    lats = blocked_mask.lat.values
    lons = blocked_mask.lon.values

    # Create coordinate grids
    LON_grid, LAT_grid = np.meshgrid(lons, lats)

    # Create cos(lat) weight grid for area weighting
    cos_lat_weights = np.cos(np.deg2rad(lats))
    COS_LAT_grid = np.broadcast_to(cos_lat_weights[:, None], (len(lats), len(lons)))

    # Step 1: Mask to Northern Hemisphere only (no ROI masking yet)
    nh_mask = LAT_grid >= nh_lat_min
    mask_nh = mask_t * nh_mask

    if mask_nh.sum() == 0:
        raise ValueError(f"No blocked grid points in Northern Hemisphere at time index {time_idx}")

    # Step 2: Find ALL connected components in NH
    labeled, num_features = ndimage.label(mask_nh)

    if num_features == 0:
        raise ValueError(f"No blocking objects found at time index {time_idx}")

    # Step 3: For each component, compute centroid (using ALL its points) and size
    components = []
    for label_id in range(1, num_features + 1):
        component_mask = (labeled == label_id)

        # Get all points in this component
        comp_lons = LON_grid[component_mask]
        comp_lats = LAT_grid[component_mask]
        comp_weights = COS_LAT_grid[component_mask]

        # Compute area-weighted centroid of this component
        total_weight = np.sum(comp_weights)
        centroid_lon = np.sum(comp_weights * comp_lons) / total_weight
        centroid_lat = np.sum(comp_weights * comp_lats) / total_weight

        # Compute size (area-weighted grid point count)
        size = total_weight

        components.append({
            'label': label_id,
            'centroid_lon': centroid_lon,
            'centroid_lat': centroid_lat,
            'size': size,
            'mask': component_mask,
        })

    # Step 4: Select components whose centroid lies inside the target ROI
    # If no ROI specified, use all NH components
    if all(v is not None for v in [region_lon_min, region_lon_max, region_lat_min, region_lat_max]):
        # Build ROI mask for computing overlap
        roi_mask = _build_roi_mask(
            LON_grid, LAT_grid,
            region_lon_min, region_lon_max,
            region_lat_min, region_lat_max,
        )

        # Compute overlap (area-weighted grid points inside ROI) for each component
        for comp in components:
            comp_mask = comp['mask']
            overlap_mask = comp_mask & roi_mask
            overlap_weights = COS_LAT_grid[overlap_mask]
            comp['roi_overlap'] = np.sum(overlap_weights) if overlap_weights.size > 0 else 0.0

        components_in_roi = [
            c for c in components
            if _centroid_in_roi(
                c['centroid_lon'], c['centroid_lat'],
                region_lon_min, region_lon_max,
                region_lat_min, region_lat_max,
            )
        ]

        if len(components_in_roi) == 0:
            raise ValueError(
                f"No blocking objects have their centroid within ROI "
                f"({region_lon_min}-{region_lon_max} E, {region_lat_min}-{region_lat_max} N) "
                f"at time index {time_idx}. "
                f"Found {num_features} blocking objects with centroids at: "
                + ", ".join([f"({c['centroid_lon']:.1f} E, {c['centroid_lat']:.1f} N)" for c in components])
            )
    else:
        # No ROI specified, use all NH components
        # Set roi_overlap to total size for selection
        for comp in components:
            comp['roi_overlap'] = comp['size']
        components_in_roi = components

    # Step 5: If multiple, choose the one with largest overlap inside ROI
    if len(components_in_roi) == 1:
        selected = components_in_roi[0]
    else:
        selected = max(components_in_roi, key=lambda c: c['roi_overlap'])

    return selected['centroid_lon'], selected['centroid_lat']


# Alias for backward compatibility
compute_unweighted_centroid = compute_blocking_centroid


# ---------------------------------------------------------------------------
# 3x3 Patch Extraction (from rmse_scorer)
# ---------------------------------------------------------------------------

def extract_3x3_patch(
    z500_anom: xr.DataArray,
    center_lon: float,
    center_lat: float,
    time_idx: int
) -> np.ndarray:
    """
    Extract a 3x3 patch of Z500 anomalies centered on the given location.

    Uses native grid points: finds the nearest grid point to (center_lon, center_lat),
    then extracts the 3x3 neighborhood (+/-1 grid index in each direction).

    Handles longitude wrapping for cyclic grids (where lon[0] and lon[-1] represent
    the same physical point, e.g., 0 and 360).

    Parameters
    ----------
    z500_anom : xr.DataArray
        Z500 anomaly data with dimensions (time, lat, lon).
    center_lon : float
        Center longitude (degrees).
    center_lat : float
        Center latitude (degrees).
    time_idx : int
        Time index at which to extract the patch.

    Returns
    -------
    patch : np.ndarray
        Flattened array of 9 values (3x3 patch).

    Raises
    ------
    ValueError
        If the 3x3 patch extends beyond the latitude boundaries.
    """
    lats = z500_anom.lat.values
    lons = z500_anom.lon.values
    n_lon = len(lons)

    # Detect cyclic grid: last lon point is a duplicate of the first (e.g., 0 and 360)
    lon_spacing = float(lons[1] - lons[0])
    has_cyclic_point = abs((lons[-1] - lons[0]) - 360.0) < lon_spacing * 0.5

    # Find nearest grid indices
    lat_idx = int(np.argmin(np.abs(lats - center_lat)))
    lon_idx = int(np.argmin(np.abs(lons - center_lon)))

    # Check latitude boundaries (no wrapping for latitude)
    if lat_idx < 1 or lat_idx >= len(lats) - 1:
        raise ValueError(
            f"Latitude index {lat_idx} too close to boundary for 3x3 patch. "
            f"Center lat: {center_lat}, grid range: [{lats[0]}, {lats[-1]}]"
        )

    if has_cyclic_point:
        # Number of unique longitude points (excluding the duplicate cyclic point)
        n_unique = n_lon - 1

        # If lon_idx is the cyclic duplicate (last index), normalize to index 0
        if lon_idx == n_unique:
            lon_idx = 0

        # Compute wrapped longitude indices within the unique range [0, n_unique)
        lon_indices = [(lon_idx + offset) % n_unique for offset in (-1, 0, 1)]

        # Extract patch using explicit index selection
        data_t = z500_anom.isel(time=time_idx)
        patch = data_t.isel(
            lat=slice(lat_idx - 1, lat_idx + 2),
            lon=lon_indices
        ).values.flatten()
    else:
        # Non-cyclic grid: check longitude boundaries
        if lon_idx < 1 or lon_idx >= n_lon - 1:
            raise ValueError(
                f"Longitude index {lon_idx} too close to boundary for 3x3 patch. "
                f"Center lon: {center_lon}, grid range: [{lons[0]}, {lons[-1]}]"
            )

        # Extract 3x3 patch (+/-1 in each direction)
        patch = z500_anom.isel(
            time=time_idx,
            lat=slice(lat_idx - 1, lat_idx + 2),
            lon=slice(lon_idx - 1, lon_idx + 2)
        ).values.flatten()

    return patch
