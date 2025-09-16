#!/usr/bin/env python3
"""
Debug script to examine NetCDF file time coordinates.
"""
import xarray as xr
import numpy as np

def examine_netcdf_files():
    """Examine the time coordinates in the NetCDF files"""

    files_to_check = [
        'data/input/gridRaw_tas_CanESM5_ssp245.nc',
        'data/input/gridRaw_pr_CanESM5_ssp245.nc',
        'data/input/gridded_gdp_regrid_CanESM5_ssp245.nc',
        'data/input/gridded_pop_regrid_CanESM5_ssp245.nc'
    ]

    for filepath in files_to_check:
        print(f"\n" + "="*60)
        print(f"FILE: {filepath}")
        print("="*60)

        try:
            ds = xr.open_dataset(filepath)
            print(f"Dataset shape: {dict(ds.dims)}")

            if 'time' in ds.dims:
                print(f"Time dimension length: {ds.dims['time']}")
                time_coord = ds.time
                print(f"Time coordinate type: {type(time_coord.values[0])}")
                print(f"First few time values: {time_coord.values[:5]}")
                print(f"Last few time values: {time_coord.values[-5:]}")

                if hasattr(time_coord.values[0], 'year'):
                    years = [t.year for t in time_coord.values]
                    print(f"Year range: {min(years)} to {max(years)}")
                elif str(time_coord.values.dtype).startswith('int'):
                    print(f"Integer time range: {time_coord.values.min()} to {time_coord.values.max()}")

            # Check for variables
            print(f"Variables: {list(ds.data_vars.keys())}")

            # Check coordinate names
            print(f"Coordinates: {list(ds.coords.keys())}")

            # Check for any obvious data issues
            for var_name, var in ds.data_vars.items():
                if len(var.dims) >= 3:  # Has time dimension
                    sample_slice = var.isel({d: 0 for d in var.dims if d != 'time'})
                    finite_count = np.isfinite(sample_slice.values).sum()
                    total_count = len(sample_slice.values)
                    print(f"{var_name}: {finite_count}/{total_count} finite values in sample slice")

            ds.close()

        except Exception as e:
            print(f"Error reading {filepath}: {e}")

if __name__ == "__main__":
    examine_netcdf_files()