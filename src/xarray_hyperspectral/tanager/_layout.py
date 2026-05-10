"""Detect Tanager HDF-EOS5 product type from file structure."""


from dataclasses import dataclass
from typing import Literal

import h5py


@dataclass(frozen=True)
class Layout:
    """Result of inspecting a Tanager HDF-EOS5 file to determine its product type."""

    # Is the product organized as a swath (with geolocation tied to each pixel)
    # or a grid (with geolocation defined by corner coordinates and pixel size)?
    geometry: Literal["swath", "grid"]

    # What kind of product is it: toa radiance or surface reflectance?
    product_kind: Literal["radiance", "reflectance"]

    # What is the specific product type?
    product_type: Literal[
        "basic_radiance_hdf5",
        "ortho_radiance_hdf5",
        "basic_sr_hdf5",
        "ortho_sr_hdf5"]

    # Where in the HDF5 file is the hyperspectral cube located?
    hyp_group: Literal["HDFEOS/GRIDS/HYP", "HDFEOS/SWATHS/HYP"]

    # What is the name of the dataset containing the hyperspectral cube?
    cube_name: Literal["toa_radiance", "surface_reflectance"]


def discover_layout(f: h5py.File) -> Layout:
    """Inspect an open Tanager HDF-EOS5 file and return its product layout.

    Raises ValueError if the file is not a recognized Tanager product.
    """
    if "HDFEOS/GRIDS/HYP" in f:
        geometry = "grid"
        prefix = "ortho"
        hyp = "HDFEOS/GRIDS/HYP"
    elif "HDFEOS/SWATHS/HYP" in f:
        geometry = "swath"
        prefix = "basic"
        hyp = "HDFEOS/SWATHS/HYP"
    else:
        raise ValueError("Not a Tanager HDF-EOS5 file: no GRIDS/HYP or SWATHS/HYP group")

    data_fields = f[f"{hyp}/Data Fields"]
    if "surface_reflectance" in data_fields:
        kind = "reflectance"
        cube_name = "surface_reflectance"
        suffix = "sr"
    elif "toa_radiance" in data_fields:
        kind = "radiance"
        cube_name = "toa_radiance"
        suffix = "radiance"
    else:
        raise ValueError("No toa_radiance or surface_reflectance dataset found")

    return Layout(
        geometry=geometry,
        product_kind=kind,
        product_type=f"{prefix}_{suffix}_hdf5",
        hyp_group=hyp,
        cube_name=cube_name,
    )
