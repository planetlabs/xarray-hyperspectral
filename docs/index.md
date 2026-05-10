# 🌈🛰️ xarray-hyperspectral

***Open hyperspectral remote sensing data cubes as xarray Datasets.***

This is a lightweight [xarray](https://xarray.dev/) backend engine for [Planet Tanager](https://www.planet.com/products/hyperspectral/) hyperspectral data. It reads Tanager's HDF-EOS5 files and returns lazy, CF-compliant xarray Datasets with proper wavelength coordinates, units, and CRS metadata. This makes Tanager data interoperable with the xarray ecosystem of visualization, analysis, and machine learning tools.
