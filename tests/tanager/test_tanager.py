"""Tests for the Tanager xarray backend.

Tests cover three layers: HDF-EOS5 layout discovery, the open_tanager()
convenience function, and the xarray BackendEntrypoint integration.
"""

import numpy as np
import pytest
import xarray as xr

from xarray_hyperspectral.tanager._backend import TanagerBackendEntrypoint, open_tanager
from xarray_hyperspectral.tanager._layout import discover_layout

from .conftest import EPSG, NBANDS, NX, NY

import h5py


class TestLayout:
    def test_basic_radiance(self, basic_radiance_path):
        with h5py.File(basic_radiance_path, "r") as f:
            layout = discover_layout(f)
        assert layout.geometry == "swath"
        assert layout.product_kind == "radiance"
        assert layout.product_type == "basic_radiance_hdf5"

    def test_basic_sr(self, basic_sr_path):
        with h5py.File(basic_sr_path, "r") as f:
            layout = discover_layout(f)
        assert layout.geometry == "swath"
        assert layout.product_kind == "reflectance"
        assert layout.product_type == "basic_sr_hdf5"

    def test_ortho_radiance(self, ortho_radiance_path):
        with h5py.File(ortho_radiance_path, "r") as f:
            layout = discover_layout(f)
        assert layout.geometry == "grid"
        assert layout.product_kind == "radiance"
        assert layout.product_type == "ortho_radiance_hdf5"

    def test_ortho_sr(self, ortho_sr_path):
        with h5py.File(ortho_sr_path, "r") as f:
            layout = discover_layout(f)
        assert layout.geometry == "grid"
        assert layout.product_kind == "reflectance"
        assert layout.product_type == "ortho_sr_hdf5"

    def test_raises_on_no_hyp_group(self, tmp_path):
        path = tmp_path / "empty.h5"
        with h5py.File(path, "w") as f:
            f.create_group("HDFEOS/OTHER")
        with h5py.File(path, "r") as f:
            with pytest.raises(ValueError, match="no GRIDS/HYP or SWATHS/HYP"):
                discover_layout(f)

    def test_raises_on_no_cube_dataset(self, tmp_path):
        path = tmp_path / "no_cube.h5"
        with h5py.File(path, "w") as f:
            f.create_group("HDFEOS/SWATHS/HYP/Data Fields")
        with h5py.File(path, "r") as f:
            with pytest.raises(
                ValueError, match="No toa_radiance or surface_reflectance"
            ):
                discover_layout(f)


class TestOpenTanager:
    def test_basic_radiance_dims(self, basic_radiance_path):
        ds = open_tanager(basic_radiance_path)
        assert "radiance" in ds
        assert ds.sizes == {"band": NBANDS, "alongtrack": NY, "crosstrack": NX}

    def test_basic_radiance_coords(self, basic_radiance_path):
        ds = open_tanager(basic_radiance_path)
        expected_coords = {
            "band",
            "wavelength",
            "fwhm",
            "radiometric_coefficients",
            "alongtrack",
            "crosstrack",
            "latitude",
            "longitude",
            "time",
            "sensor_zenith",
            "sun_zenith",
            "sensor_azimuth",
            "sun_azimuth",
            "sensor_to_ground_path_length",
            "beta_cloud_mask",
            "beta_cirrus_mask",
            "nodata_pixels",
        }
        assert expected_coords == set(ds.coords)
        assert ds["alongtrack"].attrs["long_name"] == "Along-Track Index"
        assert ds["alongtrack"].attrs["units"] == "pixel"
        assert ds["alongtrack"].attrs["axis"] == "Y"
        assert ds["crosstrack"].attrs["long_name"] == "Cross-Track Index"
        assert ds["crosstrack"].attrs["units"] == "pixel"
        assert ds["crosstrack"].attrs["axis"] == "X"
        assert ds["time"].dims == ("alongtrack",)
        assert np.issubdtype(ds["time"].dtype, np.datetime64)
        assert ds["beta_cloud_mask"].dtype == bool
        assert ds["beta_cloud_mask"].attrs["flag_meanings"] == "clear cloudy"
        assert ds["beta_cirrus_mask"].dtype == bool
        assert ds["beta_cirrus_mask"].attrs["flag_meanings"] == "clear cirrus"
        assert ds["nodata_pixels"].dtype == bool
        assert ds["nodata_pixels"].attrs["flag_meanings"] == "valid nodata"
        assert "good_wavelengths" not in ds.coords

    def test_basic_radiance_cf_attrs(self, basic_radiance_path):
        ds = open_tanager(basic_radiance_path)
        assert ds["wavelength"].attrs["standard_name"] == "radiation_wavelength"
        assert ds["wavelength"].attrs["units"] == "nm"
        assert (
            ds["radiance"].attrs["standard_name"]
            == "toa_outgoing_radiance_per_unit_wavelength"
        )
        assert ds["sensor_zenith"].attrs["standard_name"] == "sensor_zenith_angle"
        assert ds["sun_zenith"].attrs["standard_name"] == "solar_zenith_angle"
        assert ds["sensor_azimuth"].attrs["standard_name"] == "sensor_azimuth_angle"
        assert ds["sun_azimuth"].attrs["standard_name"] == "solar_azimuth_angle"
        assert ds["time"].attrs["standard_name"] == "time"
        assert "grid_mapping" not in ds["radiance"].attrs
        assert "ancillary_variables" not in ds["radiance"].attrs
        assert ds.attrs["Conventions"] == "CF-1.13"
        assert "opened with xarray-hyperspectral" in ds.attrs["history"]
        assert ds.attrs["product_type"] == "basic_radiance_hdf5"

    def test_basic_sr(self, basic_sr_path):
        ds = open_tanager(basic_sr_path)
        assert "reflectance" in ds
        assert "radiance" not in ds
        assert (
            ds["reflectance"].attrs["standard_name"]
            == "surface_bidirectional_reflectance"
        )
        assert ds.attrs["product_type"] == "basic_sr_hdf5"
        expected_coords = {
            "band",
            "wavelength",
            "fwhm",
            "good_wavelengths",
            "alongtrack",
            "crosstrack",
            "latitude",
            "longitude",
            "time",
            "sensor_zenith",
            "sun_zenith",
            "sensor_azimuth",
            "sun_azimuth",
            "sensor_to_ground_path_length",
            "aerosol_optical_depth",
            "column_water_vapour",
            "beta_cloud_mask",
            "beta_cirrus_mask",
            "nodata_pixels",
        }
        assert expected_coords == set(ds.coords)
        assert ds["good_wavelengths"].dtype == bool
        assert (
            ds["aerosol_optical_depth"].attrs["standard_name"]
            == "atmosphere_optical_thickness_due_to_ambient_aerosol_particles"
        )
        assert (
            ds["column_water_vapour"].attrs["standard_name"]
            == "atmosphere_mass_content_of_water_vapor"
        )
        assert {"reflectance", "reflectance_uncertainty"} == set(ds.data_vars)
        assert ds["reflectance"].attrs["ancillary_variables"] == "reflectance_uncertainty"
        assert ds["reflectance_uncertainty"].dims == (
            "band",
            "alongtrack",
            "crosstrack",
        )

    def test_ortho_radiance(self, ortho_radiance_path):
        ds = open_tanager(ortho_radiance_path)
        assert "radiance" in ds
        assert ds.sizes == {"band": NBANDS, "y": NY, "x": NX}
        expected_coords = {
            "band",
            "wavelength",
            "fwhm",
            "radiometric_coefficients",
            "y",
            "x",
            "spatial_ref",
            "time",
            "sensor_zenith",
            "sun_zenith",
            "sensor_azimuth",
            "sun_azimuth",
            "sensor_to_ground_path_length",
            "beta_cloud_mask",
            "beta_cirrus_mask",
            "nodata_pixels",
        }
        assert expected_coords == set(ds.coords)
        assert ds.coords["spatial_ref"].attrs["epsg_code"] == EPSG
        assert ds.coords["spatial_ref"].attrs["grid_mapping_name"] == "transverse_mercator"
        assert ds.coords["y"].attrs["axis"] == "Y"
        assert ds.coords["x"].attrs["axis"] == "X"
        assert np.issubdtype(ds["time"].dtype, np.datetime64)
        assert ds["time"].dims == ("y", "x")
        assert ds["radiance"].attrs["grid_mapping"] == "spatial_ref"
        assert ds["sensor_zenith"].attrs["grid_mapping"] == "spatial_ref"
        assert {"radiance"} == set(ds.data_vars)

    def test_ortho_sr(self, ortho_sr_path):
        ds = open_tanager(ortho_sr_path)
        assert "reflectance" in ds
        assert ds.sizes == {"band": NBANDS, "y": NY, "x": NX}
        expected_coords = {
            "band",
            "wavelength",
            "fwhm",
            "good_wavelengths",
            "y",
            "x",
            "spatial_ref",
            "time",
            "sensor_zenith",
            "sun_zenith",
            "sensor_azimuth",
            "sun_azimuth",
            "sensor_to_ground_path_length",
            "aerosol_optical_depth",
            "column_water_vapour",
            "beta_cloud_mask",
            "beta_cirrus_mask",
            "nodata_pixels",
        }
        assert expected_coords == set(ds.coords)
        assert ds.coords["spatial_ref"].attrs["epsg_code"] == EPSG
        assert {"reflectance", "reflectance_uncertainty"} == set(ds.data_vars)

    def test_lazy_load_and_shape(self, basic_radiance_path):
        ds = open_tanager(basic_radiance_path)
        values = ds["radiance"].values
        assert values.shape == (NBANDS, NY, NX)
        assert not np.all(np.isnan(values))

    def test_fill_value_masked(self, basic_radiance_path):
        ds = open_tanager(basic_radiance_path)
        assert not np.any(ds["radiance"].values == -9999.0)

    def test_fwhm_coordinate(self, basic_radiance_path):
        ds = open_tanager(basic_radiance_path)
        assert "fwhm" in ds.coords
        assert ds["fwhm"].attrs["units"] == "nm"

    def test_drop_variables(self, basic_radiance_path):
        ds = open_tanager(basic_radiance_path, drop_variables=["latitude"])
        assert "latitude" not in ds.coords


class TestBackendEntrypoint:
    def test_guess_can_open_accepts(self, basic_radiance_path):
        backend = TanagerBackendEntrypoint()
        assert backend.guess_can_open(basic_radiance_path)

    def test_guess_can_open_rejects_txt(self, tmp_path):
        txt = tmp_path / "not_h5.txt"
        txt.write_text("hello")
        backend = TanagerBackendEntrypoint()
        assert not backend.guess_can_open(txt)

    def test_xr_open_dataset(self, basic_radiance_path):
        ds = xr.open_dataset(basic_radiance_path, engine="tanager")
        assert "radiance" in ds

    def test_dask_chunks(self, basic_radiance_path):
        ds = xr.open_dataset(basic_radiance_path, engine="tanager", chunks={})
        assert ds["radiance"].chunks is not None
