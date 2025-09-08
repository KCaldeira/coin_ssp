#!/usr/bin/env python3

import xarray as xr
import numpy as np

def examine_netcdf_files():
    """Examine the structure of NetCDF files in data/input/"""
    
    files = [
        'data/input/gridRaw_tas_CanESM5_ssp585.nc',
        'data/input/gridRaw_pr_CanESM5_ssp585.nc', 
        'data/input/gridded_gdp_regrid_CanESM5.nc',
        'data/input/gridded_pop_regrid_CanESM5.nc'
    ]
    
    for file_path in files:
        print(f"\n{'='*60}")
        print(f"File: {file_path}")
        print(f"{'='*60}")
        
        try:
            ds = xr.open_dataset(file_path, decode_times=False)
            print(f"Dimensions: {dict(ds.dims)}")
            print(f"Variables: {list(ds.data_vars.keys())}")
            print(f"Coordinates: {list(ds.coords.keys())}")
            
            # Show time range - use axis_0 which appears to be the time dimension
            if 'axis_0' in ds.dims:
                axis_vals = ds.axis_0.values
                print(f"Time axis (axis_0): {len(axis_vals)} steps, values: {axis_vals[0]} to {axis_vals[-1]}")
            
            if 'time' in ds.coords:
                time_vals = ds.time.values
                print(f"Time coordinate: {time_vals[0]} to {time_vals[-1]}")
            
            # Show spatial dimensions
            if 'lat' in ds.coords:
                lat_vals = ds.lat.values
                print(f"Latitude: {lat_vals.min():.2f} to {lat_vals.max():.2f} ({len(lat_vals)} points)")
                
            if 'lon' in ds.coords:
                lon_vals = ds.lon.values
                print(f"Longitude: {lon_vals.min():.2f} to {lon_vals.max():.2f} ({len(lon_vals)} points)")
            
            # Show first data variable details
            for var_name in list(ds.data_vars.keys())[:1]:
                var = ds[var_name]
                print(f"Variable '{var_name}': shape={var.shape}, dims={var.dims}")
                if hasattr(var, 'units'):
                    print(f"  Units: {var.units}")
                if hasattr(var, 'long_name'):
                    print(f"  Description: {var.long_name}")
                    
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
        
        print()

if __name__ == "__main__":
    examine_netcdf_files()