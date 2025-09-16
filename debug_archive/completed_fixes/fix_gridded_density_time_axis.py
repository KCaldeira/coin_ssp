#!/usr/bin/env python3
"""
Fix time axis in Gridded density NetCDF files.

This script converts the problematic "years since 1661-1-1" time encoding
to simple integer years from 2006 to 2100, creating GriddedAlt_ versions.
"""

import xarray as xr
import numpy as np
import glob
import os
from datetime import datetime

def fix_gridded_density_time_axis(input_file, output_file):
    """
    Fix the time axis in a Gridded density NetCDF file and save as GriddedAlt_ version.

    Parameters
    ----------
    input_file : str
        Path to input NetCDF file
    output_file : str
        Path to output NetCDF file
    """
    print(f"Processing: {os.path.basename(input_file)}")

    # Open with decode_times=False to avoid the "years since" parsing error
    ds = xr.open_dataset(input_file, decode_times=False)

    # Check original time coordinate
    original_time = ds.time.values
    print(f"  Original time shape: {original_time.shape}")
    print(f"  Original time range: {original_time.min()} to {original_time.max()}")

    # Convert from "years since 1661" to actual years
    # The time values are offsets from 1661, so add 1661 to get actual years
    actual_years = original_time + 1661
    print(f"  Converted year range: {actual_years.min()} to {actual_years.max()}")

    # Create new time coordinate as integer years from 2006 to 2100
    # Assuming the data spans this range and we want annual resolution
    start_year = int(actual_years.min())
    end_year = int(actual_years.max())
    new_time = np.arange(start_year, end_year + 1, 1, dtype=np.int64)

    print(f"  New time coordinate: {new_time[0]} to {new_time[-1]} (length: {len(new_time)})")

    # Create new dataset with corrected time coordinate
    new_ds = ds.copy()

    # Update time coordinate
    new_ds = new_ds.assign_coords(time=('time', new_time))

    # Update time coordinate attributes
    new_ds.time.attrs.update({
        'long_name': 'Time',
        'standard_name': 'time',
        'units': 'years',
        'axis': 'T',
        'calendar': 'standard'
    })

    # Add processing history
    timestamp = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
    history_entry = f"{timestamp}: Converted time coordinate from 'years since 1661' to integer years {start_year}-{end_year}"

    if 'history' in new_ds.attrs:
        new_ds.attrs['history'] = history_entry + '; ' + new_ds.attrs['history']
    else:
        new_ds.attrs['history'] = history_entry

    # Save the corrected file
    print(f"  Saving to: {os.path.basename(output_file)}")
    new_ds.to_netcdf(output_file)
    new_ds.close()
    ds.close()

    print(f"  ✓ Successfully created {os.path.basename(output_file)}")

def main():
    """Process all Gridded density files."""

    # Pattern to match Gridded density files
    pattern = "/home/kcaldeira/coin_ssp/data/input/Gridded_*density_*.nc"
    input_files = glob.glob(pattern)

    if not input_files:
        print("No Gridded density files found!")
        return

    print(f"Found {len(input_files)} Gridded density files:")
    for f in input_files:
        print(f"  {os.path.basename(f)}")

    print("\nCreating GriddedAlt_ versions with corrected time axis...")

    success_count = 0
    for input_file in input_files:
        try:
            # Create output path with GriddedAlt_ prefix
            input_dir = os.path.dirname(input_file)
            filename = os.path.basename(input_file)
            new_filename = filename.replace('Gridded_', 'GriddedAlt_')
            output_file = os.path.join(input_dir, new_filename)

            fix_gridded_density_time_axis(input_file, output_file)
            success_count += 1

        except Exception as e:
            print(f"  ❌ Error processing {os.path.basename(input_file)}: {e}")

    print(f"\n✓ Successfully processed {success_count}/{len(input_files)} files")

    # List the created files
    alt_pattern = "/home/kcaldeira/coin_ssp/data/input/GriddedAlt_*density_*.nc"
    alt_files = glob.glob(alt_pattern)
    print(f"\nCreated {len(alt_files)} GriddedAlt_ files:")
    for f in sorted(alt_files):
        print(f"  {os.path.basename(f)}")

if __name__ == "__main__":
    main()