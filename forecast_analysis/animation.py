#!/usr/bin/env python3
"""
Animation Generation for Forecast Blocking Analysis.

This module provides functions for creating GIF animations of blocking events.

Author: AI-RES Project
"""

import os
import sys
from typing import Dict, Optional

# Add src/ directory to path for importing core modules
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(os.path.dirname(_THIS_DIR), 'src')
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


def create_forecast_animation(
    event_info: Dict,
    output_path: str,
    title_prefix: str = "Forecast",
    region_lon_min: float = 30.0,
    region_lon_max: float = 100.0,
    region_lat_min: float = 55.0,
    region_lat_max: float = 75.0,
    fps: float = 0.5,
    show_diagnostics: bool = True
) -> Optional[str]:
    """
    Create an animation GIF for forecast blocking analysis.

    Parameters
    ----------
    event_info : dict
        Event information dictionary from process_single_forecast().
        Must contain 'z500_anom', 'blocked_mask', 'threshold_90'.
    output_path : str
        Path to save the output GIF file.
    title_prefix : str, optional
        Prefix for the animation title. Default: "Forecast".
    region_lon_min, region_lon_max : float, optional
        Longitude bounds for diagnostic region. Default: 30-100°E.
    region_lat_min, region_lat_max : float, optional
        Latitude bounds for diagnostic region. Default: 55-75°N.
    fps : float, optional
        Frames per second (0.5 = 2 seconds per frame). Default: 0.5.
    show_diagnostics : bool, optional
        Whether to show diagnostic panels. Default: True.

    Returns
    -------
    output_path : str or None
        Path to the saved GIF file, or None if animation module not available.

    Notes
    -----
    Requires the event_animation module to be available in src/.
    """
    try:
        from event_animation import create_event_animation_gif_fast
    except ImportError:
        print("Warning: event_animation not available. Skipping animation.")
        return None

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create full simulation animation
    create_event_animation_gif_fast(
        event_id=320,  # Placeholder event ID for title
        ano_stats=event_info,
        save_path=output_path,
        title_prefix=title_prefix,
        event_start_times=None,  # Not needed for full simulation
        show_diagnostics=show_diagnostics,
        region_lon_min=region_lon_min,
        region_lon_max=region_lon_max,
        region_lat_min=region_lat_min,
        region_lat_max=region_lat_max,
        fps=fps,
        animate_full_simulation=True,
    )

    return output_path
