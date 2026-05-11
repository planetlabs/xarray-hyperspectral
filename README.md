# 🌈🛰️ xarray-hyperspectral

***Open hyperspectral remote sensing data cubes as xarray Datasets.***

Xarray-hyperspectral is a lightweight [xarray](https://xarray.dev/) backend engine for [Planet Tanager](https://www.planet.com/products/hyperspectral/) hyperspectral data. It reads Tanager's HDF-EOS5 files and returns lazy, CF-compliant xarray Datasets with proper wavelength coordinates, units, and CRS metadata. This makes Tanager data interoperable with the xarray ecosystem of visualization, analysis, and machine learning tools.

![Demo](docs/xarray-hyperspectral-demo.gif)

## 📦 Installation

```bash
pip install xarray-hyperspectral
```

## 🚀 Quickstart

```python
import xarray as xr

# Open a local Tanager HDF-EOS5 file
ds = xr.open_dataset("scene.h5", engine="tanager")

# Plot a single band
ds["reflectance"].sel(wavelength=850, method="nearest").plot()

# Plot a spectrum at a single pixel
ds["reflectance"].isel(y=400, x=400).plot()
```

See the [tutorials](docs/tutorials/quickstart.ipynb) for more examples including RGB composites and interactive visualization.

## 📝 License

Apache-2.0
