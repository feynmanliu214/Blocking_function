import numpy as np
import pytest
import xarray as xr
import tempfile
import os

def _create_test_u250_file(tmp_path, n_times=10, n_lats=64, n_lons=128):
    """Create a minimal NetCDF file with ua on pressure levels."""
    times = xr.cftime_range("0100-01-01", periods=n_times, freq="D",
                            calendar="proleptic_gregorian")
    lats = np.linspace(-87.86, 87.86, n_lats)
    lons = np.linspace(0, 357.19, n_lons)
    plev = np.array([25000.0, 50000.0])  # 250 hPa and 500 hPa

    ua_data = np.random.randn(n_times, len(plev), n_lats, n_lons).astype(np.float32)
    # Make the 250 hPa level have jet-like values (20-50 m/s) in midlatitudes
    mid_lat_mask = (lats >= 30) & (lats <= 60)
    ua_data[:, 0, mid_lat_mask, :] = 35.0  # strong jet at 250 hPa

    ds = xr.Dataset(
        {"ua": (["time", "plev", "lat", "lon"], ua_data)},
        coords={"time": times, "plev": plev, "lat": lats, "lon": lons},
    )
    filepath = os.path.join(tmp_path, "test_ua.nc")
    ds.to_netcdf(filepath)
    return filepath, times, lats, lons


def test_load_u250_returns_correct_shape(tmp_path):
    """_load_u250 should return (time, lat, lon) DataArray at 250 hPa."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
    from event_animation import _load_u250

    filepath, times, lats, lons = _create_test_u250_file(tmp_path)
    result = _load_u250(filepath)

    assert isinstance(result, xr.DataArray)
    assert result.dims == ("time", "lat", "lon")
    assert len(result.time) == len(times)
    assert len(result.lat) == len(lats)


def test_load_u250_selects_250hpa(tmp_path):
    """_load_u250 should select the 250 hPa (25000 Pa) level."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
    from event_animation import _load_u250

    filepath, _, lats, _ = _create_test_u250_file(tmp_path)
    result = _load_u250(filepath)

    # The 250 hPa level has value 35.0 in midlatitudes
    mid_lat_mask = (result.lat.values >= 30) & (result.lat.values <= 60)
    mid_lat_values = result.isel(time=0).values[mid_lat_mask, :]
    assert np.allclose(mid_lat_values, 35.0)


def test_load_u250_raises_without_path():
    """_load_u250 should raise ValueError when path is None."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
    from event_animation import _load_u250

    with pytest.raises(ValueError, match="u250_path"):
        _load_u250(None)


def _create_minimal_ano_stats(n_times=10, n_lats=32, n_lons=128):
    """Create minimal ano_stats dict for animation testing."""
    import cftime
    times = [cftime.DatetimeProlepticGregorian(100, 1, d+1) for d in range(n_times)]
    lats = np.linspace(0, 87.86, n_lats)
    lons = np.linspace(0, 357.19, n_lons)

    z500_anom = xr.DataArray(
        np.random.randn(n_times, n_lats, n_lons).astype(np.float32) * 100,
        coords={"time": times, "lat": lats, "lon": lons},
        dims=["time", "lat", "lon"],
    )
    event_mask = xr.DataArray(
        np.zeros((n_times, n_lats, n_lons), dtype=int),
        coords={"time": times, "lat": lats, "lon": lons},
        dims=["time", "lat", "lon"],
    )
    # Create a small event (ID=1) spanning 5 days
    event_mask.values[0:5, 15:20, 50:60] = 1

    blocked_mask = xr.DataArray(
        (event_mask.values > 0).astype("uint8"),
        coords={"time": times, "lat": lats, "lon": lons},
        dims=["time", "lat", "lon"],
    )

    return {
        "z500_anom": z500_anom,
        "event_mask": event_mask,
        "blocked_mask": blocked_mask,
        "event_areas": {1: [1e11] * 5},
        "num_events": 1,
        "all_event_ids": [1],
        "event_durations": {1: 5},
    }


def test_fast_animation_dynamics_mode_creates_gif(tmp_path):
    """create_event_animation_gif_fast with overlay_mode='dynamics' should produce a GIF."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
    from event_animation import create_event_animation_gif_fast

    ano_stats = _create_minimal_ano_stats()
    u250_path, _, _, _ = _create_test_u250_file(tmp_path)
    save_path = os.path.join(tmp_path, "test_dynamics.gif")

    result = create_event_animation_gif_fast(
        event_id=1,
        ano_stats=ano_stats,
        save_path=save_path,
        overlay_mode='dynamics',
        u250_path=u250_path,
        fps=1,
        dpi=50,  # low res for speed
    )

    assert result == save_path
    assert os.path.exists(save_path)
    assert os.path.getsize(save_path) > 0


def test_fast_animation_blocking_mode_creates_gif(tmp_path):
    """create_event_animation_gif_fast with overlay_mode='blocking' (default) should work."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
    from event_animation import create_event_animation_gif_fast

    ano_stats = _create_minimal_ano_stats()
    save_path = os.path.join(tmp_path, "test_blocking.gif")

    result = create_event_animation_gif_fast(
        event_id=1,
        ano_stats=ano_stats,
        save_path=save_path,
        overlay_mode='blocking',
        fps=1,
        dpi=50,
    )

    assert result == save_path
    assert os.path.exists(save_path)


def test_dynamics_mode_without_u250_path_raises():
    """overlay_mode='dynamics' without u250_path should raise ValueError."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
    from event_animation import create_event_animation_gif_fast

    ano_stats = _create_minimal_ano_stats()

    with pytest.raises(ValueError, match="u250_path"):
        create_event_animation_gif_fast(
            event_id=1,
            ano_stats=ano_stats,
            overlay_mode='dynamics',
            u250_path=None,
        )
