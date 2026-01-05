# Blocking Function: Atmospheric Blocking Analysis Tools

This repository contains a collection of Python tools and scripts for the detection, analysis, and visualization of atmospheric blocking events. It is designed to work with both ERA5 reanalysis data and PlaSim (Planet Simulator) model output.

## Features

- **Blocking Detection Algorithms**:
  - `ABS` (Absolute Blocking Scale) method.
  - `ANO` (Anomaly) method for blocking detection.
- **Data Preprocessing**: Standardized processing for ERA5 and PlaSim datasets, including coordinate standardization and subsetting.
- **Statistical Analysis**: Tools to compute blocking scores, event durations, areas, and other lifecycle statistics.
- **Visualization**:
  - Event animations (cinematic and fast versions).
  - Spatial evolution plots.
  - Duration and area histograms.
  - Diagnostic plots for blocking events.

## Repository Structure

- `src/`: Core Python modules for detection and analysis.
- `script/`: Jupyter notebooks demonstrating usage and specific analyses (ERA5, PlaSim, climatology).
- `bash_scripts/`: Shell scripts for batch processing and data downloading.
- `requirements.txt`: Python dependencies.

## Installation

Ensure you have the required dependencies installed:

```bash
pip install -r requirements.txt
```

## Usage

The core functionality is located in the `src/` directory. You can import these modules in your scripts or Jupyter notebooks:

```python
from src.ABS_PlaSim import prepare_for_blocking_analysis
from src.compute_blocking_scores import compute_blocking_scores
# ... and more
```

Refer to the notebooks in the `script/` directory for detailed examples of the analysis workflow.

