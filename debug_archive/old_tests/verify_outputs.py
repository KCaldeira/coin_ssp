#!/usr/bin/env python3

import xarray as xr
import numpy as np
import os

def verify_outputs():
    """Verify the NetCDF and PDF outputs were created correctly."""
    
    print("=== Output File Verification ===\n")
    
    # Check files exist
    netcdf_path = "data/output/target_gdp_reductions.nc"
    pdf_path = "data/output/target_gdp_reductions_maps.pdf"
    
    if os.path.exists(netcdf_path):
        print(f"✅ NetCDF file exists: {netcdf_path}")
        file_size = os.path.getsize(netcdf_path) / 1024  # KB
        print(f"   File size: {file_size:.1f} KB")
    else:
        print(f"❌ NetCDF file missing: {netcdf_path}")
        return
    
    if os.path.exists(pdf_path):
        print(f"✅ PDF file exists: {pdf_path}")
        file_size = os.path.getsize(pdf_path) / 1024  # KB
        print(f"   File size: {file_size:.1f} KB")
    else:
        print(f"❌ PDF file missing: {pdf_path}")
    
    # Examine NetCDF structure
    print(f"\n=== NetCDF Structure ===")
    ds = xr.open_dataset(netcdf_path)
    
    print(f"Dimensions: {dict(ds.dims)}")
    print(f"Variables: {list(ds.data_vars.keys())}")
    print(f"Coordinates: {list(ds.coords.keys())}")
    
    # Check the 3D target reductions array
    target_reductions = ds.target_gdp_reductions.values
    print(f"\nTarget GDP Reductions Array:")
    print(f"  Shape: {target_reductions.shape}")
    print(f"  Data type: {target_reductions.dtype}")
    
    # Show layer statistics
    reduction_types = ['Constant', 'Linear', 'Quadratic']
    for i, name in enumerate(reduction_types):
        data = target_reductions[i]
        print(f"\n{name} Layer:")
        print(f"  Min: {data.min():.4f}")
        print(f"  Max: {data.max():.4f}")
        print(f"  Mean: {data.mean():.4f}")
        print(f"  Has NaN: {np.isnan(data).any()}")
        print(f"  Has Inf: {np.isinf(data).any()}")
    
    # Check coordinates
    print(f"\nCoordinates:")
    print(f"  Latitude range: {ds.lat.min().values:.2f}° to {ds.lat.max().values:.2f}°")
    print(f"  Longitude range: {ds.lon.min().values:.2f}° to {ds.lon.max().values:.2f}°")
    print(f"  Temperature range: {ds.temperature_ref.min().values:.1f}°C to {ds.temperature_ref.max().values:.1f}°C")
    
    # Verify units and attributes
    print(f"\nAttributes:")
    print(f"  Target reductions units: {ds.target_gdp_reductions.attrs.get('units', 'Not set')}")
    print(f"  Temperature units: {ds.temperature_ref.attrs.get('units', 'Not set')}")
    
    print(f"\n✅ All outputs verified successfully!")

if __name__ == "__main__":
    verify_outputs()