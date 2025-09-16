#!/usr/bin/env python3
"""
Script to fix GridRaw NetCDF files by renaming axis_0 to time
and changing values from 0-85 to 2015-2100.
"""

import xarray as xr
import numpy as np
import glob
import os

def fix_time_axis(file_path):
    """
    Fix the time axis in a GridRaw NetCDF file and save as gridRawAlt_ version.

    Parameters
    ----------
    file_path : str
        Path to the NetCDF file to modify
    """
    print(f"Processing: {os.path.basename(file_path)}")

    # Load the dataset
    ds = xr.open_dataset(file_path)

    # Check if axis_0 exists
    if 'axis_0' not in ds.dims:
        print(f"  Warning: axis_0 dimension not found in {file_path}")
        ds.close()
        return

    # Get the current axis_0 values
    current_values = ds.axis_0.values
    print(f"  Current axis_0 values: {current_values[0]} to {current_values[-1]} (length: {len(current_values)})")

    # Create new time values from 2015 to 2100
    new_time_values = np.arange(2015, 2015 + len(current_values))
    print(f"  New time values: {new_time_values[0]} to {new_time_values[-1]} (length: {len(new_time_values)})")

    # Rename the dimension and coordinate
    ds = ds.rename({'axis_0': 'time'})

    # Update the time coordinate values
    ds = ds.assign_coords(time=new_time_values)

    # Add proper time attributes
    ds.time.attrs = {
        'long_name': 'Time',
        'standard_name': 'time',
        'units': 'years',
        'axis': 'T',
        'calendar': 'standard'
    }

    # Update global attributes
    if 'history' in ds.attrs:
        ds.attrs['history'] += f"\n{np.datetime64('now')}: Renamed axis_0 to time, updated values to 2015-2100"
    else:
        ds.attrs['history'] = f"{np.datetime64('now')}: Renamed axis_0 to time, updated values to 2015-2100"

    # Create output path with gridRawAlt_ prefix
    directory = os.path.dirname(file_path)
    filename = os.path.basename(file_path)
    new_filename = filename.replace('gridRaw_', 'gridRawAlt_')
    output_path = os.path.join(directory, new_filename)

    # Save the modified dataset
    print(f"  Saving as: {new_filename}")
    ds.to_netcdf(output_path)
    ds.close()
    print(f"  ✅ Complete: {new_filename}")

def main():
    """Process all GridRaw NetCDF files in the data/input directory."""

    # Find all GridRaw files
    pattern = "/home/kcaldeira/coin_ssp/data/input/gridRaw_*.nc"
    files = glob.glob(pattern)

    print(f"Found {len(files)} GridRaw files to process:")
    for file_path in files:
        print(f"  - {os.path.basename(file_path)}")

    print("\nCreating gridRawAlt_ versions with corrected time axis...")
    print("=" * 60)

    for file_path in files:
        try:
            fix_time_axis(file_path)
        except Exception as e:
            print(f"  ❌ Error processing {os.path.basename(file_path)}: {e}")
        print()

    print("=" * 60)
    print("All files processed!")

    # Show what was created
    alt_pattern = "/home/kcaldeira/coin_ssp/data/input/gridRawAlt_*.nc"
    alt_files = glob.glob(alt_pattern)
    print(f"\nCreated {len(alt_files)} gridRawAlt_ files:")
    for file_path in alt_files:
        print(f"  - {os.path.basename(file_path)}")

if __name__ == "__main__":
    main()