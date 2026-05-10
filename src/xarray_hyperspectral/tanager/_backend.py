"""Planet Tanager xarray backend engine.

This module implements lazy loading of Planet Tanager HDF-EOS5 hyperspectral
data cubes as xarray Datasets with CF-compliant metadata.

Provides three components:
- H5BackendArray: lazy array adapter for reading HDF5 datasets on demand.
- open_tanager: reads a Tanager file and returns an xarray Dataset.
- TanagerBackendEntrypoint: registers the engine="tanager" backend with xarray.
"""

import datetime
import importlib.metadata
import os
import re

import h5py
import numpy as np
import xarray as xr

from xarray_hyperspectral.tanager._layout import Layout, discover_layout

# Mapping from Tanager dataset names to more CF-friendly names
RENAME = {
    "toa_radiance": "radiance",
    "surface_reflectance": "reflectance",
    "surface_reflectance_uncertainty": "reflectance_uncertainty",
    "Latitude": "latitude",
    "Longitude": "longitude",
    "Time": "time",
}

# Define CF attributes to attach to variables when opening a Tanager file.
# The HDF-EOS5 data format used by Tanager pre-dates the CF conventions, so we
# need to add them manually. Many modern tools in the xarray ecosystem rely on
# these attributes to auto-discover variables or understand their meaning.
CF_ATTRS = {
    "band": {
        "long_name": "Band Index",
    },
    "wavelength": {
        "standard_name": "radiation_wavelength",
        "long_name": "Wavelength",
        "units": "nm",
    },
    "fwhm": {
        "long_name": "Full Width at Half Maximum",
        "units": "nm",
    },
    "radiance": {
        "standard_name": "toa_outgoing_radiance_per_unit_wavelength",
        "long_name": "Top-of-Atmosphere Radiance",
        "units": "W m-2 sr-1 um-1",
    },
    "reflectance": {
        "standard_name": "surface_bidirectional_reflectance",
        "long_name": "Surface Reflectance",
    },
    "reflectance_uncertainty": {
        "long_name": "Surface Reflectance Uncertainty",
    },
    "latitude": {
        "standard_name": "latitude",
        "long_name": "Latitude",
        "units": "degrees_north",
    },
    "longitude": {
        "standard_name": "longitude",
        "long_name": "Longitude",
        "units": "degrees_east",
    },
    "alongtrack": {
        "long_name": "Along-Track Index",
        "units": "pixel",
        "axis": "Y",
    },
    "crosstrack": {
        "long_name": "Cross-Track Index",
        "units": "pixel",
        "axis": "X",
    },
    "x": {
        "standard_name": "projection_x_coordinate",
        "long_name": "Easting",
        "units": "m",
        "axis": "X",
    },
    "y": {
        "standard_name": "projection_y_coordinate",
        "long_name": "Northing",
        "units": "m",
        "axis": "Y",
    },
    "sensor_zenith": {
        "standard_name": "sensor_zenith_angle",
        "long_name": "Sensor Zenith Angle",
        "units": "degree",
    },
    "sun_zenith": {
        "standard_name": "solar_zenith_angle",
        "long_name": "Solar Zenith Angle",
        "units": "degree",
    },
    "sensor_azimuth": {
        "standard_name": "sensor_azimuth_angle",
        "long_name": "Sensor Azimuth Angle",
        "units": "degree",
    },
    "sun_azimuth": {
        "standard_name": "solar_azimuth_angle",
        "long_name": "Solar Azimuth Angle",
        "units": "degree",
    },
    "sensor_to_ground_path_length": {
        "long_name": "Sensor to Ground Path Length",
        "units": "m",
    },
    "aerosol_optical_depth": {
        "standard_name": "atmosphere_optical_thickness_due_to_ambient_aerosol_particles",
        "long_name": "Aerosol Optical Depth at 550 nm",
        "radiation_wavelength": "550 nm",
    },
    "column_water_vapour": {
        "standard_name": "atmosphere_mass_content_of_water_vapor",
        "long_name": "Column Water Vapour",
        "units": "g cm-2",
    },
    "good_wavelengths": {
        "long_name": "Good Wavelength Indicator",
    },
    "beta_cloud_mask": {
        "long_name": "Cloud Mask (Beta)",
        "flag_values": [0, 1],
        "flag_meanings": "clear cloudy",
    },
    "beta_cirrus_mask": {
        "long_name": "Cirrus Mask (Beta)",
        "flag_values": [0, 1],
        "flag_meanings": "clear cirrus",
    },
    "nodata_pixels": {
        "long_name": "No Data Pixel Mask",
        "flag_values": [0, 1],
        "flag_meanings": "valid nodata",
    },
    "time": {
        "standard_name": "time",
        "long_name": "Time",
    },
}


class H5BackendArray(xr.backends.BackendArray):
    """Lazy array that reads slices from an HDF5 dataset on demand.

    Xarray's built-in HDF5 backends are designed for netCDF4 files -- a
    specific convention for organizing data inside HDF5.  Tanager files
    use a different convention (HDF-EOS5), so we need our own array class
    that reads from h5py directly.

    BackendArray is xarray's interface for lazy data access: subclasses
    provide shape, dtype, and __getitem__.  Instances get wrapped in
    LazilyIndexedArray, which defers all I/O until values are actually
    needed.

    Dask integration comes for free: when a user passes chunks= to
    open_dataset, xarray automatically wraps lazy variables in dask
    arrays -- no dask import needed in this module.  We store the file
    path (not an open h5py.File) so the array can be serialized for
    dask.distributed, and we open/close the file on each read for
    thread safety.
    """

    def __init__(self, path, h5_dataset_path, shape, dtype, fill_value):
        self.path = path
        self.h5_dataset_path = h5_dataset_path
        self.shape = shape
        self.dtype = np.dtype(dtype)
        self.fill_value = fill_value

    def __getitem__(self, key):
        return xr.core.indexing.explicit_indexing_adapter(
            key, self.shape, xr.core.indexing.IndexingSupport.BASIC, self._getitem
        )

    def _getitem(self, key):
        with h5py.File(self.path, "r") as f:
            data = f[self.h5_dataset_path][key]
        if self.fill_value is not None and np.issubdtype(self.dtype, np.floating):
            data = np.where(data == self.fill_value, np.nan, data)
        return data


def _get_fill_value(ds: h5py.Dataset, key: str = "_FillValue") -> float | None:
    """Return the fill value attribute from an HDF5 dataset, or None."""
    return ds.attrs.get(key)


def _parse_struct_metadata(text: str) -> dict[str, str]:
    """Extract key=value pairs from an HDF-EOS5 StructMetadata.0 text blob.

    StructMetadata.0 is a plain-text dataset defined by the HDF-EOS5
    specification that describes grid structure and projection parameters.
    For ortho (gridded) products, we parse it to get the corner coordinates
    (UpperLeftPointMtrs, LowerRightMtrs) needed to compute pixel locations.
    """
    result = {}
    for m in re.finditer(r"(\w+)=(.+)", text):
        result[m.group(1)] = m.group(2).strip('"')
    return result


def _parse_point(val: str) -> tuple[float, float]:
    """Parse a StructMetadata.0 point string like '(123.4,567.8)' into floats."""
    val = val.strip("()")
    parts = val.split(",")
    return float(parts[0]), float(parts[1])


def _build_grid_coords(
    f: h5py.File, layout: Layout, ny: int, nx: int
) -> dict[str, xr.Variable]:
    """Compute pixel-center x/y coordinates and CRS for ortho products.

    Ortho (geometrically corrected) products are stored as HDF-EOS5 Grids.
    The grid's spatial extent is encoded in StructMetadata.0 as corner
    coordinates in a projected coordinate reference system (CRS), identified
    by an EPSG code stored as an attribute on the HYP group.  We divide the
    extent by the number of pixels to get the pixel size, then compute
    pixel-center coordinates (offset by half a pixel from the corners).

    The returned spatial_ref variable follows the rioxarray/GDAL convention
    so that tools like rioxarray can detect the CRS automatically.
    """
    sm_raw = f["HDFEOS INFORMATION/StructMetadata.0"][()]
    if isinstance(sm_raw, bytes):
        sm_raw = sm_raw.decode()
    sm = _parse_struct_metadata(sm_raw)
    ul_x, ul_y = _parse_point(sm["UpperLeftPointMtrs"])
    lr_x, lr_y = _parse_point(sm["LowerRightMtrs"])
    epsg = int(f[layout.hyp_group].attrs["epsg_code"])

    pixel_w = (lr_x - ul_x) / nx
    pixel_h = (lr_y - ul_y) / ny
    return {
        "y": xr.Variable("y", ul_y + (np.arange(ny) + 0.5) * pixel_h, attrs=CF_ATTRS["y"]),
        "x": xr.Variable("x", ul_x + (np.arange(nx) + 0.5) * pixel_w, attrs=CF_ATTRS["x"]),
        "spatial_ref": xr.Variable(
            (),
            0,
            attrs={
                "grid_mapping_name": "transverse_mercator",
                "crs_wkt": f"EPSG:{epsg}",
                "epsg_code": epsg,
            },
        ),
    }


def open_tanager(
    path: str | os.PathLike[str],
    *,
    drop_variables: list[str] | None = None,
) -> xr.Dataset:
    """Open a Tanager HDF-EOS5 file and return a lazy xarray Dataset.

    Steps:
    1. Detect file layout (swath vs grid, radiance vs reflectance).
    2. Build wavelength and fwhm coordinates from cube attributes.
    3. Build spatial coordinates -- pixel-center x/y for ortho products,
       per-pixel lat/lon for basic (swath) products.
    4. Wrap the main data cube in a lazy H5BackendArray.
    5. Collect remaining ancillary fields from Data Fields: 2D arrays
       become coords (e.g. sensor_zenith), 3D arrays become data_vars
       (e.g. reflectance_uncertainty).
    """
    drop = set(drop_variables or [])
    coords = {}
    data_vars = {}

    with h5py.File(path, "r") as f:
        # Step 1: Detect file layout
        layout = discover_layout(f)
        data_fields = f[f"{layout.hyp_group}/Data Fields"]

        cube_ds = data_fields[layout.cube_name]
        nbands, ny, nx = cube_ds.shape
        fill = _get_fill_value(cube_ds)

        # Step 2: Band and spectral coordinates
        wavelengths = np.asarray(cube_ds.attrs["wavelengths"])
        fwhm_vals = np.asarray(cube_ds.attrs["fwhm"])

        coords["band"] = xr.Variable("band", np.arange(nbands), attrs=CF_ATTRS["band"])
        if "wavelength" not in drop:
            coords["wavelength"] = xr.Variable(
                "band",
                wavelengths,
                attrs=CF_ATTRS["wavelength"],
            )

        # Step 3: Spatial dimension coordinates (listed right after wavelength)
        if layout.geometry == "grid":
            dim_y, dim_x = "y", "x"
            coords.update(_build_grid_coords(f, layout, ny, nx))
        else:
            dim_y, dim_x = "alongtrack", "crosstrack"
            coords["alongtrack"] = xr.Variable("alongtrack", np.arange(ny), attrs=CF_ATTRS["alongtrack"])
            coords["crosstrack"] = xr.Variable("crosstrack", np.arange(nx), attrs=CF_ATTRS["crosstrack"])

        # Spectral auxiliary coordinates
        if "fwhm" not in drop:
            coords["fwhm"] = xr.Variable("band", fwhm_vals, attrs=CF_ATTRS["fwhm"])
        if "good_wavelengths" not in drop:
            gw = cube_ds.attrs.get("good_wavelengths")
            if gw is not None:
                coords["good_wavelengths"] = xr.Variable(
                    "band",
                    np.asarray(gw, dtype=bool),
                    attrs=CF_ATTRS["good_wavelengths"],
                )

        # Geolocation coordinates (swath only)
        if layout.geometry != "grid":
            geo_fields = f[f"{layout.hyp_group}/Geolocation Fields"]
            if "latitude" not in drop:
                coords["latitude"] = xr.Variable(
                    (dim_y, dim_x),
                    geo_fields["Latitude"][:],
                    attrs=CF_ATTRS["latitude"],
                )
            if "longitude" not in drop:
                coords["longitude"] = xr.Variable(
                    (dim_y, dim_x),
                    geo_fields["Longitude"][:],
                    attrs=CF_ATTRS["longitude"],
                )
            if "time" not in drop and "Time" in geo_fields:
                raw = geo_fields["Time"][:]
                fill = _get_fill_value(geo_fields["Time"])
                if fill is not None:
                    raw = np.where(raw == fill, np.nan, raw)
                coords["time"] = xr.Variable(
                    dim_y,
                    (raw * 1e9).astype("datetime64[ns]"),
                    attrs=CF_ATTRS["time"],
                )

        dims_3d = ("band", dim_y, dim_x)
        dims_2d = (dim_y, dim_x)
        spatial_attrs = {"grid_mapping": "spatial_ref"} if layout.geometry == "grid" else {}

        # Step 4: Main data cube
        out_name = RENAME.get(layout.cube_name, layout.cube_name)
        if out_name not in drop:
            backend_arr = H5BackendArray(
                path,
                f"{layout.hyp_group}/Data Fields/{layout.cube_name}",
                (nbands, ny, nx),
                cube_ds.dtype,
                fill,
            )
            lazy = xr.core.indexing.LazilyIndexedArray(backend_arr)
            data_vars[out_name] = xr.Variable(
                dims_3d, lazy, attrs={**CF_ATTRS.get(out_name, {}), **spatial_attrs}
            )

        # Step 5: Ancillary fields
        for ds_name in data_fields:
            if ds_name == layout.cube_name:
                continue
            renamed = RENAME.get(ds_name, ds_name)
            if renamed in drop:
                continue
            h5ds = data_fields[ds_name]
            if not isinstance(h5ds, h5py.Dataset):
                continue
            if h5ds.ndim == 2:
                arr = h5ds[:]
                ds_fill = _get_fill_value(h5ds)
                if ds_fill is not None and np.issubdtype(arr.dtype, np.floating):
                    arr = np.where(arr == ds_fill, np.nan, arr)
                if renamed == "time":
                    arr = (arr * 1e9).astype("datetime64[ns]")
                elif arr.dtype == np.uint8 and ds_fill is not None:
                    # Mask fields (e.g. beta_cloud_mask) are uint8 with fill=255; convert to bool
                    arr = np.where(arr == ds_fill, 0, arr).astype(bool)
                coords[renamed] = xr.Variable(
                    dims_2d, arr, attrs={**CF_ATTRS.get(renamed, {}), **spatial_attrs}
                )
            elif h5ds.ndim == 3:
                ds_fill = _get_fill_value(h5ds)
                backend = H5BackendArray(
                    path,
                    f"{layout.hyp_group}/Data Fields/{ds_name}",
                    (nbands, ny, nx),
                    h5ds.dtype,
                    ds_fill,
                )
                data_vars[renamed] = xr.Variable(
                    dims_3d,
                    xr.core.indexing.LazilyIndexedArray(backend),
                    attrs={**CF_ATTRS.get(renamed, {}), **spatial_attrs},
                )

    if "reflectance" in data_vars and "reflectance_uncertainty" in data_vars:
        data_vars["reflectance"].attrs["ancillary_variables"] = "reflectance_uncertainty"

    return xr.Dataset(
        data_vars,
        coords=coords,
        attrs={
            "Conventions": "CF-1.13",
            "history": f"{datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')} "
            f"opened with xarray-hyperspectral v{importlib.metadata.version('xarray-hyperspectral')}",
            "source": os.path.basename(path),
            "product_type": layout.product_type,
            "instrument": "Tanager",
            "references": "https://docs.planet.com/data/imagery/tanager/techspec/"
        },
    )


class TanagerBackendEntrypoint(xr.backends.BackendEntrypoint):
    """Xarray backend entrypoint for Planet Tanager HDF-EOS5 files.

    Registered as ``engine="tanager"`` via the ``xarray.backends`` entry point
    in pyproject.toml. Users don't instantiate this class directly; xarray
    calls it when you write::

        ds = xr.open_dataset("scene.h5", engine="tanager")

    Xarray's engine protocol layers chunking (``chunks=``), variable dropping
    (``drop_variables=``), and CF decoding on top of what this entrypoint
    returns. See https://docs.xarray.dev/en/latest/internals/how-to-add-new-backend.html
    """

    description = "Open Planet Tanager HDF-EOS5 hyperspectral data cubes (.h5)"
    open_dataset_parameters = ["filename_or_obj", "drop_variables"]

    def guess_can_open(self, filename_or_obj) -> bool:
        """Return True if the file looks like a Tanager HDF-EOS5 product.

        Checks for a .h5 extension and the presence of a
        ``HDFEOS/GRIDS/HYP`` or ``HDFEOS/SWATHS/HYP`` group. Called by
        xarray when no engine is specified, to auto-detect the right backend.
        This assumes no other satellites use the same structure.
        """
        try:
            p = str(filename_or_obj)
        except Exception:
            return False
        if not p.endswith(".h5"):
            return False
        try:
            with h5py.File(p, "r") as f:
                return "HDFEOS/GRIDS/HYP" in f or "HDFEOS/SWATHS/HYP" in f
        except Exception:
            return False

    def open_dataset(
        self,
        filename_or_obj,
        *,
        drop_variables=None,
    ) -> xr.Dataset:
        """Open a Tanager HDF-EOS5 file and return a lazy xarray Dataset.

        Called by xarray's engine protocol; delegates to :func:`open_tanager`.
        """
        return open_tanager(
            filename_or_obj,
            drop_variables=drop_variables,
        )
