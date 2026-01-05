"""
Event Duration GIF Generation Module

This module provides functions for creating animated GIFs of blocking events
filtered by specific durations, organized by method (ABS/ANO) and season.

Author: Generated for AIRES project
Date: 2025-11-14
"""

import os
from typing import List, Dict, Optional, Union
import xarray as xr
from event_animation import convert_abs_events_to_ano_format, create_event_animation_gif


def create_duration_specific_gifs(
    event_info: Dict,
    z500_data: xr.DataArray,
    method: str,
    season: str,
    target_durations: List[int],
    output_base_dir: str = '/home/zhixingliu/projects/aires/figures',
    fps: int = 2,
    dpi: int = 100,
    selection_criterion: str = 'first'
) -> Dict[int, Optional[str]]:
    """
    Create animated GIFs for blocking events with specific durations.
    
    This function filters events by duration and creates one GIF per target duration,
    organizing outputs in method/season-specific folders.
    
    Parameters
    ----------
    event_info : dict
        Event information dictionary from blocking detection.
        Can be either:
        - ABS format: {'event_count': int, 'persistent_events': [...], ...}
        - ANO format: {'num_events': int, 'event_durations': {...}, ...}
    z500_data : xarray.DataArray
        The Z500 data used for blocking detection (needed for animation background).
        Should have dimensions (time, lat, lon).
    method : str
        Blocking detection method. Must be 'ABS' or 'ANO'.
    season : str
        Season identifier. Typically 'DJF', 'JJA', 'MAM', or 'SON'.
    target_durations : list of int
        List of target durations (in days) to create GIFs for.
        Example: [5, 6, 10, 15]
    output_base_dir : str, default='/home/zhixingliu/projects/aires/figures'
        Base directory for output files. GIFs will be saved in subdirectories
        organized as: {output_base_dir}/{method} {season}/
    fps : int, default=2
        Frames per second for the animation.
    dpi : int, default=100
        Resolution (dots per inch) for the saved figures.
    selection_criterion : str, default='first'
        How to select an event when multiple events have the same duration.
        Options:
        - 'first': Use the first event found
        - 'largest': Use the event with the largest mean area
        - 'longest_lived': Use the event with the longest duration (redundant here)
    
    Returns
    -------
    dict
        Dictionary mapping target_duration -> gif_path (or None if no event found).
        Example: {5: '/path/to/5days.gif', 6: None, 10: '/path/to/10days.gif', ...}
    
    Examples
    --------
    >>> # For ABS DJF events
    >>> gif_paths = create_duration_specific_gifs(
    ...     event_info=abs_event_info,
    ...     z500_data=z500_djf_100years,
    ...     method='ABS',
    ...     season='DJF',
    ...     target_durations=[5, 6, 10, 15]
    ... )
    
    >>> # For ANO JJA events
    >>> gif_paths = create_duration_specific_gifs(
    ...     event_info=ano_stats_jja,
    ...     z500_data=z500_jja_100years,
    ...     method='ANO',
    ...     season='JJA',
    ...     target_durations=[5, 10, 15, 20]
    ... )
    
    Notes
    -----
    - Creates output directory automatically if it doesn't exist
    - For ABS events, converts to ANO format internally for animation compatibility
    - Prints progress information for each duration
    - Returns None for durations where no matching events are found
    """
    # Validate method
    method = method.upper()
    if method not in ['ABS', 'ANO']:
        raise ValueError(f"method must be 'ABS' or 'ANO', got '{method}'")
    
    # Create output folder
    output_folder = os.path.join(output_base_dir, f'{method} {season}')
    os.makedirs(output_folder, exist_ok=True)
    print(f"\n📁 Output folder: {output_folder}")
    
    # Detect format and check for events
    if 'event_count' in event_info and 'persistent_events' in event_info:
        # ABS format
        event_count = event_info.get('event_count', 0)
        if event_count == 0:
            print(f"\n⚠️  No {method} events detected for {season}. Cannot create GIFs.")
            return {dur: None for dur in target_durations}
        
        # Convert ABS format to ANO format for animation
        print(f"\n🔄 Converting {method} format to ANO format for animation...")
        ano_stats = convert_abs_events_to_ano_format(event_info, z500_data)
        persistent_events = event_info['persistent_events']
        
    elif 'num_events' in event_info and 'event_durations' in event_info:
        # ANO format - already in correct format
        num_events = event_info.get('num_events', 0)
        if num_events == 0:
            print(f"\n⚠️  No {method} events detected for {season}. Cannot create GIFs.")
            return {dur: None for dur in target_durations}
        
        ano_stats = event_info
        # Convert ANO format to list of events for filtering
        persistent_events = []
        for event_id in event_info['all_event_ids']:
            duration = event_info['event_durations'][event_id]
            persistent_events.append({
                'event_id': event_id,
                'duration': duration
            })
    else:
        available_keys = list(event_info.keys())
        raise ValueError(
            f"event_info format not recognized. "
            f"Expected ABS format (with 'event_count' and 'persistent_events') "
            f"or ANO format (with 'num_events' and 'event_durations'). "
            f"Available keys: {available_keys}"
        )
    
    # Create GIFs for each target duration
    gif_paths = {}
    
    print(f"\n🎬 Creating GIFs for {method} {season} events with specific durations...")
    print(f"   Target durations: {target_durations}")
    print(f"   Total events available: {len(persistent_events)}")
    
    for target_dur in target_durations:
        # Find all events with exactly this duration
        if 'event_count' in event_info:
            # ABS format - enumerate from persistent_events list
            events_with_duration = [
                (idx, event) 
                for idx, event in enumerate(persistent_events) 
                if event['duration'] == target_dur
            ]
        else:
            # ANO format - filter by event_id
            events_with_duration = [
                (event['event_id'], event)
                for event in persistent_events
                if event['duration'] == target_dur
            ]
        
        if len(events_with_duration) > 0:
            # Select event based on criterion
            if selection_criterion == 'first':
                event_idx_or_id, event = events_with_duration[0]
            elif selection_criterion == 'largest' and 'event_count' in event_info:
                # For ABS: select by mean area
                event_idx_or_id, event = max(
                    events_with_duration,
                    key=lambda x: x[1].get('mean_area', 0)
                )
            else:
                # Default to first
                event_idx_or_id, event = events_with_duration[0]
            
            # Determine event_id
            if 'event_count' in event_info:
                # ABS format: event IDs start at 1
                event_id = event_idx_or_id + 1
            else:
                # ANO format: use the event_id directly
                event_id = event_idx_or_id
            
            print(f"\n{'='*70}")
            print(f"{method} {season} Event with Duration = {target_dur} days")
            print(f"   Event ID: {event_id}")
            print(f"   Total events with this duration: {len(events_with_duration)}")
            print(f"   Duration: {event['duration']} days")
            if 'start_time' in event:
                print(f"   Start time: {event['start_time']}")
                print(f"   End time: {event['end_time']}")
            print(f"{'='*70}")
            
            # Create animation filename
            filename = f'{method}_Event_{season}_duration_{target_dur}days_animation.gif'
            save_path = os.path.join(output_folder, filename)
            
            # Create animation
            try:
                gif_path = create_event_animation_gif(
                    event_id=event_id,
                    ano_stats=ano_stats,
                    save_path=save_path,
                    title_prefix=f'{method} {season} PlaSim (Duration = {target_dur} days)',
                    fps=fps,
                    dpi=dpi
                )
                gif_paths[target_dur] = gif_path
            except Exception as e:
                print(f"\n❌ Error creating GIF for duration {target_dur}: {e}")
                gif_paths[target_dur] = None
        else:
            print(f"\n⚠️  No events found with duration = {target_dur} days for {method} {season}")
            gif_paths[target_dur] = None
    
    # Summary
    successful = sum(1 for path in gif_paths.values() if path is not None)
    print(f"\n{'='*70}")
    print(f"✅ GIF Creation Complete for {method} {season}")
    print(f"   Successfully created: {successful}/{len(target_durations)} GIFs")
    print(f"   Output folder: {output_folder}")
    print(f"{'='*70}\n")
    
    return gif_paths


def create_duration_gifs_batch(
    event_info_dict: Dict[str, Dict],
    z500_data_dict: Dict[str, xr.DataArray],
    target_durations: List[int],
    output_base_dir: str = '/home/zhixingliu/projects/aires/figures',
    fps: int = 2,
    dpi: int = 100
) -> Dict[str, Dict[int, Optional[str]]]:
    """
    Create GIFs for multiple method/season combinations in batch.
    
    Parameters
    ----------
    event_info_dict : dict
        Dictionary mapping '{method}_{season}' to event_info.
        Example: {
            'ABS_DJF': abs_event_info,
            'ABS_JJA': abs_event_info_jja,
            'ANO_DJF': ano_stats,
            'ANO_JJA': ano_stats_jja
        }
    z500_data_dict : dict
        Dictionary mapping '{method}_{season}' to z500_data.
        Example: {
            'ABS_DJF': z500_djf_100years,
            'ABS_JJA': z500_jja_100years,
            'ANO_DJF': z500_djf_100years_ANO,
            'ANO_JJA': z500_jja_100years_ANO
        }
    target_durations : list of int
        List of target durations to create GIFs for.
    output_base_dir : str, default='/home/zhixingliu/projects/aires/figures'
        Base directory for output files.
    fps : int, default=2
        Frames per second for animations.
    dpi : int, default=100
        Resolution for saved figures.
    
    Returns
    -------
    dict
        Nested dictionary: {'{method}_{season}': {duration: gif_path, ...}, ...}
    
    Examples
    --------
    >>> results = create_duration_gifs_batch(
    ...     event_info_dict={
    ...         'ABS_DJF': abs_event_info,
    ...         'ABS_JJA': abs_event_info_jja
    ...     },
    ...     z500_data_dict={
    ...         'ABS_DJF': z500_djf_100years,
    ...         'ABS_JJA': z500_jja_100years
    ...     },
    ...     target_durations=[5, 6, 10, 15]
    ... )
    """
    all_results = {}
    
    for key in event_info_dict.keys():
        if key not in z500_data_dict:
            print(f"⚠️  Warning: No z500_data found for {key}, skipping...")
            continue
        
        # Parse method and season from key (e.g., 'ABS_DJF' -> 'ABS', 'DJF')
        parts = key.split('_')
        if len(parts) != 2:
            print(f"⚠️  Warning: Invalid key format '{key}', expected 'METHOD_SEASON'")
            continue
        
        method, season = parts
        
        gif_paths = create_duration_specific_gifs(
            event_info=event_info_dict[key],
            z500_data=z500_data_dict[key],
            method=method,
            season=season,
            target_durations=target_durations,
            output_base_dir=output_base_dir,
            fps=fps,
            dpi=dpi
        )
        
        all_results[key] = gif_paths
    
    return all_results

