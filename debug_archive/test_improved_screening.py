#!/usr/bin/env python3
"""
Test the improved grid cell screening that checks all time points.
"""
import numpy as np
import json
from coin_ssp_utils import load_all_netcdf_data, get_ssp_data

def test_improved_screening():
    """Test the improved screening that checks all time points"""

    # Load the configuration
    with open('coin_ssp_integrated_config_example.json', 'r') as f:
        config = json.load(f)

    # Load NetCDF data
    print("Loading NetCDF data...")
    all_netcdf_data = load_all_netcdf_data(config)

    # Get SSP245 data
    gdp_data = get_ssp_data(all_netcdf_data, 'ssp245', 'gdp')
    pop_data = get_ssp_data(all_netcdf_data, 'ssp245', 'population')

    print(f"GDP data shape: {gdp_data.shape}")
    print(f"Population data shape: {pop_data.shape}")

    nlat, nlon, ntime = gdp_data.shape

    # Test both screening methods
    print(f"\nComparing screening methods:")

    # Old screening (first time point only)
    old_valid_count = 0
    new_valid_count = 0

    print("Analyzing grid cells...")

    for lat_idx in range(nlat):
        for lon_idx in range(nlon):
            # Old method (first time point only)
            old_valid = (gdp_data[lat_idx, lon_idx, 0] > 0 and
                        pop_data[lat_idx, lon_idx, 0] > 0)

            # New method (all time points)
            gdp_timeseries = gdp_data[lat_idx, lon_idx, :]
            pop_timeseries = pop_data[lat_idx, lon_idx, :]
            new_valid = (np.all(gdp_timeseries > 0) and
                        np.all(pop_timeseries > 0))

            if old_valid:
                old_valid_count += 1
            if new_valid:
                new_valid_count += 1

            # Show examples of cells that would be filtered out by new method
            if old_valid and not new_valid and new_valid_count < 5:
                print(f"\n  Example cell ({lat_idx}, {lon_idx}) - Old: Valid, New: Invalid")
                print(f"    GDP zeros: {np.sum(gdp_timeseries == 0)}/{len(gdp_timeseries)}")
                print(f"    Pop zeros: {np.sum(pop_timeseries == 0)}/{len(pop_timeseries)}")
                print(f"    GDP range: {np.min(gdp_timeseries):.2e} to {np.max(gdp_timeseries):.2e}")
                print(f"    Pop range: {np.min(pop_timeseries):.2e} to {np.max(pop_timeseries):.2e}")

    print(f"\nScreening Results:")
    print(f"  Old method (first time point only): {old_valid_count} valid cells")
    print(f"  New method (all time points): {new_valid_count} valid cells")
    print(f"  Cells filtered out by new method: {old_valid_count - new_valid_count}")
    print(f"  Filtering rate: {100*(old_valid_count - new_valid_count)/old_valid_count:.1f}%")

if __name__ == "__main__":
    test_improved_screening()