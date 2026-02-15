# Atmospheric Blocking Detection - src

> **📚 Full Documentation:** See [`../.skills/overview/blocking_detection.md`](../.skills/overview/blocking_detection.md) for the centralized version of this documentation.

This directory contains Python code for detecting and analyzing atmospheric blocking events using 500 hPa geopotential height (Z500) data.

## Project Overview

Atmospheric blocking refers to persistent high-pressure systems that disrupt the normal westerly flow in the mid-latitudes. This codebase implements multiple detection algorithms and visualization tools for analyzing blocking in climate model output (primarily PlaSim) and reanalysis data (ERA5).

## Key Modules

### Data Loading & Preprocessing

- **Process.py** - Core data loading for PlaSim model output. Key functions:
  - `z500_for_season()` - Load Z500 for DJF/JJA seasons
  - `process_multiple_files()` - Batch load multiple years
  - `standardize_coordinates()` - Ensure consistent lat/lon orientation

- **Process_ERA5.py** - ERA5 reanalysis data loading with similar interface

- **plasim_z500_loader.py** - Specialized PlaSim Z500 loader with extended options

### Blocking Detection Methods

Three detection algorithms are implemented, all returning blocking frequency maps and event information:

- **ANO_PlaSim.py** - Anomaly-based method (Woollings et al., 2018)
  - Uses 90th percentile threshold on Z500 anomalies
  - Explicit blob tracking with area/duration/overlap criteria
  - Main function: `blocking_detection_complete()`

- **DG83p_PlaSim.py** - Dole-Gordon 1983 with persistence factor
  - Exponential smoothing (p=0.97) encodes persistence
  - Model-specific sigma threshold calibration
  - See `DG83p_method_summary.md` for detailed documentation
  - Main function: `dg83p_blocking_complete()`

- **ABS_PlaSim.py** - Absolute threshold method
  - Uses absolute Z500 values rather than anomalies
  - Main function: `prepare_for_blocking_analysis()`

### Visualization & Animation

- **event_animation.py** - Create animated visualizations of blocking events
- **event_visualization.py** - Static event visualizations
- **event_duration_gifs.py** - Generate GIFs showing event evolution
- **duration_plots.py** - Duration distribution and timeline plots
- **Plot.py** - General plotting utilities

### Analysis

- **event_statistics.py** - Statistical analysis of blocking events
- **compute_blocking_scores.py** - Blocking index/score computation
- **forecast_blocking_analysis.py** - Analysis tools for forecast data

## Data Format

### Input
- NetCDF files with Z500 (variable: `zg`) on pressure levels
- Expected dimensions: `(time, plev, lat, lon)`
- PlaSim files typically named: `{year}_gaussian.nc`

### Output
Detection functions return:
```python
blocking_frequency : xr.DataArray  # (lat, lon) - fraction of blocked time
event_info : dict                  # Event masks, statistics, parameters
```

## Coordinate Conventions

- Latitude: North-to-South (90 to -90) after standardization
- Longitude: 0-360 degrees with cyclic point at 360
- Time: Daily resolution, decoded as datetime

## Key Parameters

### ANO Method
- Threshold: 90th percentile of anomalies
- Minimum area: 2×10⁶ km²
- Minimum duration: 5 days
- Overlap criterion: 50% between consecutive days

### DG83p Method
- Persistence factor (p): 0.97 (e-folding ~33 days)
- Threshold: n_sigma × σ_model (default n_sigma=3.0)
- Target window: DOY 171-230 (June 20 - August 18)

## Dependencies

- numpy, xarray, scipy (core computation)
- matplotlib (visualization)
- tqdm (progress bars)
- cartopy (map projections, for visualization)

## Usage Example

### Path Convention & Importing
When working in Jupyter notebooks (e.g., in the `script/` directory), note that the kernel location may differ from the notebook location. It is recommended to use **absolute paths** to add the `src` directory to `sys.path` to ensure reliable imports.

```python
import sys
import os

# Use absolute path to the project src directory
src_path = '/glade/u/home/zhil/project/AI-RES/Blocking/src'
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from Process import process_multiple_files
from ANO_PlaSim import ano_blocking_complete
```

### Analysis Example
```python
# Load 50 years of JJA data
z500 = process_multiple_files(range(1000, 1050), season='JJA')

# Detect blocking events
blocking_freq, events = blocking_detection_complete(z500)

# events contains:
#   - event_mask: 3D labeled array (time, lat, lon)
#   - num_events: total number of blocking events
#   - event_durations: dict mapping event_id -> duration in days
```

## File Organization

```
src/
├── Process.py              # PlaSim data loading
├── Process_ERA5.py         # ERA5 data loading
├── plasim_z500_loader.py   # Extended PlaSim loader
├── ANO_PlaSim.py           # Anomaly method
├── DG83p_PlaSim.py         # DG83p method
├── ABS_PlaSim.py           # Absolute method
├── DG83p_method_summary.md # DG83p documentation
├── event_animation.py      # Event animations
├── event_visualization.py  # Static visualizations
├── event_duration_gifs.py  # Duration GIFs
├── event_statistics.py     # Statistical analysis
├── duration_plots.py       # Duration plots
├── compute_blocking_scores.py
├── forecast_blocking_analysis.py
├── Plot.py                 # Plotting utilities
└── __init__.py
```
