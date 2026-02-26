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
