"""Fixtures that create minimal Tanager HDF-EOS5 files for testing.

Each fixture writes a small synthetic HDF5 file mimicking the on-disk layout
of a real Tanager product (groups, datasets, attributes, StructMetadata).
The files are written to pytest's tmp_path and cleaned up automatically.
"""

import h5py
import numpy as np
import pytest

NY, NX, NBANDS = 4, 6, 10
WAVELENGTHS = np.linspace(400.0, 2500.0, NBANDS, dtype=np.float32)
FWHM = np.full(NBANDS, 5.5, dtype=np.float32)
RNG = np.random.default_rng(42)

UL_X, UL_Y = 312960.0, 6624060.0
PIXEL_SIZE = 30.0
LR_X = UL_X + NX * PIXEL_SIZE
LR_Y = UL_Y - NY * PIXEL_SIZE
EPSG = 32750


def _add_cube_attrs(ds, *, include_good_wavelengths=False):
    ds.attrs["wavelengths"] = WAVELENGTHS
    ds.attrs["wavelengths_units"] = "nm"
    ds.attrs["fwhm"] = FWHM
    ds.attrs["fwhm_units"] = "nm"
    if include_good_wavelengths:
        ds.attrs["good_wavelengths"] = np.ones(NBANDS, dtype=np.uint8)


def _add_geo_swath(swath):
    geo = swath.create_group("Geolocation Fields")
    lats = np.linspace(40.0, 41.0, NY * NX, dtype=np.float64).reshape(NY, NX)
    lons = np.linspace(-105.0, -104.0, NY * NX, dtype=np.float64).reshape(NY, NX)
    geo.create_dataset("Latitude", data=lats)
    geo.create_dataset("Longitude", data=lons)
    time = geo.create_dataset(
        "Time", data=np.linspace(1.778e9, 1.779e9, NY, dtype=np.float64)
    )
    time.attrs["_FillValue"] = np.float64(-9999.0)


def _add_struct_metadata(f, *, is_grid):
    info = f.create_group("HDFEOS INFORMATION")
    if is_grid:
        sm_text = (
            f"GROUP=GridStructure\n"
            f"\tGROUP=GRID_1\n"
            f'\t\tGridName="HYP"\n'
            f"\t\tXDim={NX}\n"
            f"\t\tYDim={NY}\n"
            f"\t\tUpperLeftPointMtrs=({UL_X},{UL_Y})\n"
            f"\t\tLowerRightMtrs=({LR_X},{LR_Y})\n"
            f"\tEND_GROUP=GRID_1\n"
            f"END_GROUP=GridStructure\n"
        )
    else:
        sm_text = "GROUP=SwathStructure\nEND_GROUP=SwathStructure\n"
    info.create_dataset("StructMetadata.0", data=sm_text)


def _add_ancillary_2d(data_fields, *, include_sr_fields=False):
    for name in (
        "sensor_zenith",
        "sun_zenith",
        "sensor_azimuth",
        "sun_azimuth",
        "sensor_to_ground_path_length",
    ):
        ds = data_fields.create_dataset(
            name, data=RNG.random((NY, NX)).astype(np.float32)
        )
        ds.attrs["_FillValue"] = np.float32(-9999.0)
    if include_sr_fields:
        for name in ("aerosol_optical_depth", "column_water_vapour"):
            ds = data_fields.create_dataset(
                name, data=RNG.random((NY, NX)).astype(np.float32)
            )
            ds.attrs["_FillValue"] = np.float32(-9999.0)
    for mask_name in ("beta_cloud_mask", "beta_cirrus_mask", "nodata_pixels"):
        mask = data_fields.create_dataset(
            mask_name, data=RNG.integers(0, 2, size=(NY, NX), dtype=np.uint8)
        )
        mask.attrs["_FillValue"] = np.uint8(255)


def _add_time_2d(data_fields):
    time = data_fields.create_dataset(
        "Time", data=np.full((NY, NX), 1.778e9, dtype=np.float64)
    )
    time.attrs["_FillValue"] = np.float64(-9999.0)


@pytest.fixture
def basic_radiance_path(tmp_path):
    path = tmp_path / "basic_radiance.h5"
    with h5py.File(path, "w") as f:
        swath = f.create_group("HDFEOS/SWATHS/HYP")
        df = swath.create_group("Data Fields")
        cube = df.create_dataset(
            "toa_radiance",
            data=RNG.random((NBANDS, NY, NX)).astype(np.float32),
        )
        cube.attrs["_FillValue"] = np.float32(-9999.0)
        _add_cube_attrs(cube)
        _add_ancillary_2d(df)
        _add_geo_swath(swath)
        _add_struct_metadata(f, is_grid=False)
    return path


@pytest.fixture
def basic_sr_path(tmp_path):
    path = tmp_path / "basic_sr.h5"
    with h5py.File(path, "w") as f:
        swath = f.create_group("HDFEOS/SWATHS/HYP")
        df = swath.create_group("Data Fields")
        cube = df.create_dataset(
            "surface_reflectance",
            data=RNG.random((NBANDS, NY, NX)).astype(np.float32),
        )
        cube.attrs["_FillValue"] = np.float32(-9999.0)
        _add_cube_attrs(cube, include_good_wavelengths=True)
        unc = df.create_dataset(
            "surface_reflectance_uncertainty",
            data=RNG.random((NBANDS, NY, NX)).astype(np.float32),
        )
        unc.attrs["_FillValue"] = np.float32(-9999.0)
        _add_ancillary_2d(df, include_sr_fields=True)
        _add_geo_swath(swath)
        _add_struct_metadata(f, is_grid=False)
    return path


@pytest.fixture
def ortho_radiance_path(tmp_path):
    path = tmp_path / "ortho_radiance.h5"
    with h5py.File(path, "w") as f:
        grid = f.create_group("HDFEOS/GRIDS/HYP")
        grid.attrs["epsg_code"] = np.int64(EPSG)
        df = grid.create_group("Data Fields")
        cube = df.create_dataset(
            "toa_radiance",
            data=RNG.random((NBANDS, NY, NX)).astype(np.float32),
        )
        cube.attrs["_FillValue"] = np.float32(-9999.0)
        _add_cube_attrs(cube)
        _add_ancillary_2d(df)
        _add_time_2d(df)
        _add_struct_metadata(f, is_grid=True)
    return path


@pytest.fixture
def ortho_sr_path(tmp_path):
    path = tmp_path / "ortho_sr.h5"
    with h5py.File(path, "w") as f:
        grid = f.create_group("HDFEOS/GRIDS/HYP")
        grid.attrs["epsg_code"] = np.int64(EPSG)
        df = grid.create_group("Data Fields")
        cube = df.create_dataset(
            "surface_reflectance",
            data=RNG.random((NBANDS, NY, NX)).astype(np.float32),
        )
        cube.attrs["_FillValue"] = np.float32(-9999.0)
        _add_cube_attrs(cube, include_good_wavelengths=True)
        unc = df.create_dataset(
            "surface_reflectance_uncertainty",
            data=RNG.random((NBANDS, NY, NX)).astype(np.float32),
        )
        unc.attrs["_FillValue"] = np.float32(-9999.0)
        _add_ancillary_2d(df, include_sr_fields=True)
        _add_time_2d(df)
        _add_struct_metadata(f, is_grid=True)
    return path
