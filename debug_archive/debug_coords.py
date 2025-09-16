#!/usr/bin/env python3
"""
Debug NetCDF coordinates to understand the structure.
"""
import xarray as xr

def debug_coordinates():
    """Debug coordinate names and structures"""

    files = [
        'data/input/gridRaw_tas_CanESM5_ssp245.nc',
        'data/input/gridded_gdp_regrid_CanESM5_ssp245.nc',
        'data/input/gridded_pop_regrid_CanESM5_ssp245.nc'
    ]

    for filepath in files:
        print(f"\n" + "="*60)
        print(f"FILE: {filepath}")
        print("="*60)

        try:
            ds = xr.open_dataset(filepath, decode_times=False)

            print(f"Dimensions: {dict(ds.dims)}")
            print(f"Coordinates: {list(ds.coords.keys())}")
            print(f"Data variables: {list(ds.data_vars.keys())}")

            # Check each coordinate
            for coord_name in ds.coords:
                coord = ds.coords[coord_name]
                print(f"\nCoordinate '{coord_name}':")
                print(f"  Shape: {coord.shape}")
                print(f"  Dtype: {coord.dtype}")
                if hasattr(coord, 'values'):
                    values = coord.values
                    if len(values) > 0:
                        print(f"  Values: {values[:3]}...{values[-3:]} (showing first/last 3)")
                        if hasattr(coord, 'units'):
                            print(f"  Units: {coord.units}")

            ds.close()

        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    debug_coordinates()