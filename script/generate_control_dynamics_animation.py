"""
Generate a jet-stream overlay animation for the non-blocking control case
at timestamp 0076-01-02 (Atlantic region, DOY ±5, ±7-day buffer).

Uses overlay_mode='dynamics' to show U250 isotachs instead of blocking contours.
Uses pre-computed climatology and thresholds — only loads year 76 data.
"""

import sys
import os
import json
import warnings

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import xarray as xr
from plasim_z500_loader import standardize_coordinates
from ANO_PlaSim import create_blocking_mask
from event_animation import create_event_animation_gif_fast

# ── Configuration ──────────────────────────────────────────────────────────
DATA_DIR = '/glade/u/home/zhil/project/AI-RES/Blocking/data/PlaSim/sim52/plev_data'
REPO_ROOT = '/glade/u/home/zhil/project/AI-RES/Blocking'
CLIM_FILE = os.path.join(REPO_ROOT, 'data', 'ano_climatology_thresholds.nc')
THRESHOLD_FILE = os.path.join(REPO_ROOT, 'data', 'ano_thresholds.json')

CONTROL_TIMESTAMP = '0076-01-02'
ANIMATION_DAYS = 20
OUTPUT_DIR = os.path.join(REPO_ROOT, 'figures')

# U250 source file (year 76 PlaSim file — contains 'ua' on pressure levels)
U250_PATH = os.path.join(DATA_DIR, '76_gaussian.nc')

# ── Step 1: Load Z500 for year 76 only ─────────────────────────────────────
print("=" * 70)
print("Step 1: Loading Z500 for year 76 only (not all 100 years)")
print("=" * 70)

filepath = os.path.join(DATA_DIR, '76_gaussian.nc')
try:
    ds = xr.open_dataset(
        filepath,
        decode_times=xr.coders.CFDatetimeCoder(use_cftime=True)
    )
except (TypeError, AttributeError):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = xr.open_dataset(filepath, use_cftime=True)

# Extract Z500 (500 hPa = 50000 Pa)
z500_raw = ds['zg'].sel(plev=50000.0)

# Subset to Northern Hemisphere
lat_values = ds['lat'].values
nh_mask = lat_values >= 0
z500_nh = z500_raw.isel(lat=nh_mask)

# Daily mean (6-hourly -> daily)
z500_daily = z500_nh.resample(time="1D").mean()

# Standardize coordinates (ascending lat, 0-360 lon, cyclic point)
z500_daily = standardize_coordinates(z500_daily).compute()

ds.close()

print(f"  Z500 shape: {z500_daily.shape}")
print(f"  Time range: {z500_daily.time.values[0]} to {z500_daily.time.values[-1]}")
print(f"  Lat range: {float(z500_daily.lat.min()):.1f} to {float(z500_daily.lat.max()):.1f}")

# ── Step 2: Compute anomalies using pre-saved climatology ──────────────────
print("\n" + "=" * 70)
print("Step 2: Computing anomalies using pre-saved climatology")
print("=" * 70)

clim_ds = xr.open_dataset(CLIM_FILE)
z500_clim = clim_ds['z500_climatology']
print(f"  Climatology shape: {z500_clim.shape}")
print(f"  Climatology source: {z500_clim.attrs.get('source', 'unknown')}")

# Compute anomalies: Z'(t) = Z(t) - Z_clim(doy(t))
dayofyear = z500_daily.time.dt.dayofyear
z500_anom = z500_daily.groupby(dayofyear) - z500_clim
z500_anom = z500_anom.drop_vars('dayofyear', errors='ignore')
z500_anom = z500_anom.astype('float32')

clim_ds.close()

print(f"  Anomaly shape: {z500_anom.shape}")
print(f"  Mean anomaly: {float(z500_anom.mean()):.2f} m")

# ── Step 3: Create blocking mask using pre-saved thresholds ────────────────
print("\n" + "=" * 70)
print("Step 3: Creating blocking mask using pre-saved thresholds")
print("=" * 70)

with open(THRESHOLD_FILE, 'r') as f:
    threshold_data = json.load(f)

# Convert string keys to int
threshold_90 = {int(k): v for k, v in threshold_data['monthly_thresholds'].items()}
print(f"  Thresholds: {threshold_90}")

blocked_mask = create_blocking_mask(z500_anom, threshold_90)
print(f"  Blocked mask shape: {blocked_mask.shape}")
print(f"  Blocked fraction: {float(blocked_mask.mean()) * 100:.2f}%")

# Create a dummy event mask (no events for control case)
event_mask = xr.DataArray(
    np.zeros_like(blocked_mask.values, dtype=int),
    coords=blocked_mask.coords,
    dims=blocked_mask.dims
)

# ── Step 4: Find time window for 0076-01-02 ────────────────────────────────
print("\n" + "=" * 70)
print(f"Step 4: Finding time window for {CONTROL_TIMESTAMP}")
print("=" * 70)

time_coords = z500_anom.time.values
start_time_idx = None
for t_idx, t_val in enumerate(time_coords):
    if str(t_val)[:10] == CONTROL_TIMESTAMP:
        start_time_idx = t_idx
        break

if start_time_idx is None:
    print(f"ERROR: Could not find time index for {CONTROL_TIMESTAMP}")
    print(f"  Available time range: {time_coords[0]} to {time_coords[-1]}")
    sys.exit(1)

end_time_idx = min(start_time_idx + ANIMATION_DAYS, len(time_coords))
actual_days = end_time_idx - start_time_idx

print(f"  Start index: {start_time_idx}")
print(f"  Time: {time_coords[start_time_idx]}")
print(f"  Animation window: {actual_days} days")

# ── Step 5: Build ano_stats subset ─────────────────────────────────────────
print("\n" + "=" * 70)
print("Step 5: Building ano_stats subset for animation")
print("=" * 70)

t_slice = slice(start_time_idx, end_time_idx)

ano_stats_subset = {
    'z500_anom': z500_anom.isel(time=t_slice),
    'blocked_mask': blocked_mask.isel(time=t_slice),
    'event_mask': event_mask.isel(time=t_slice),
    'event_areas': {0: [1e6] * actual_days},
    'all_event_ids': [0],
    'event_durations': {0: actual_days},
}

print(f"  Subset z500_anom shape: {ano_stats_subset['z500_anom'].shape}")

# ── Step 6: Generate animation with dynamics overlay ───────────────────────
print("\n" + "=" * 70)
print("Step 6: Generating animation with jet stream overlay")
print("=" * 70)

os.makedirs(OUTPUT_DIR, exist_ok=True)
save_path = os.path.join(
    OUTPUT_DIR,
    f'control_dynamics_{CONTROL_TIMESTAMP.replace("-", "")}_{ANIMATION_DAYS}days.gif'
)

gif_path = create_event_animation_gif_fast(
    event_id=0,
    ano_stats=ano_stats_subset,
    save_path=save_path,
    title_prefix=f'Non-blocking Control ({CONTROL_TIMESTAMP})',
    fps=1,
    dpi=100,
    overlay_mode='dynamics',
    u250_path=U250_PATH,
    animate_full_simulation=True,
)

print(f"\nAnimation saved: {gif_path}")
