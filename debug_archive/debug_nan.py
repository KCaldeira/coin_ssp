#!/usr/bin/env python3
"""
Debug script to diagnose NaN issues in the integrated pipeline.
"""
import numpy as np
import json
from coin_ssp_utils import load_all_netcdf_data, get_ssp_data, calculate_tfp_coin_ssp
from coin_ssp_core import ModelParams

def debug_grid_cell_data():
    """Debug a specific grid cell to understand NaN generation"""

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

    # Find a valid grid cell
    nlat, nlon, ntime = gdp_data.shape
    valid_cells = []

    for lat_idx in range(min(5, nlat)):  # Check first 5 rows
        for lon_idx in range(min(10, nlon)):  # Check first 10 columns
            if gdp_data[lat_idx, lon_idx, 0] > 0 and pop_data[lat_idx, lon_idx, 0] > 0:
                valid_cells.append((lat_idx, lon_idx))
                print(f"Valid cell found at ({lat_idx}, {lon_idx})")

                # Extract time series
                gdp_series = gdp_data[lat_idx, lon_idx, :]
                pop_series = pop_data[lat_idx, lon_idx, :]

                print(f"  GDP range: {np.min(gdp_series):.2e} to {np.max(gdp_series):.2e}")
                print(f"  Pop range: {np.min(pop_series):.2e} to {np.max(pop_series):.2e}")

                # Check for NaN or zero values
                print(f"  GDP has NaN: {np.any(np.isnan(gdp_series))}")
                print(f"  Pop has NaN: {np.any(np.isnan(pop_series))}")
                print(f"  GDP has zeros: {np.any(gdp_series == 0)}")
                print(f"  Pop has zeros: {np.any(pop_series == 0)}")

                # Try TFP calculation
                params = ModelParams()
                print(f"  Attempting TFP calculation...")

                try:
                    tfp_result, k_result = calculate_tfp_coin_ssp(pop_series, gdp_series, params)
                    print(f"  TFP calculation successful!")
                    print(f"  TFP range: {np.min(tfp_result):.6f} to {np.max(tfp_result):.6f}")
                    print(f"  K range: {np.min(k_result):.6f} to {np.max(k_result):.6f}")
                    print(f"  TFP has NaN: {np.any(np.isnan(tfp_result))}")
                    print(f"  K has NaN: {np.any(np.isnan(k_result))}")

                    # Show last few values where the error occurs
                    print(f"  Last 5 TFP values: {tfp_result[-5:]}")
                    print(f"  Last 5 K values: {k_result[-5:]}")

                except Exception as e:
                    print(f"  TFP calculation failed: {e}")

                break
        if valid_cells:
            break

    if not valid_cells:
        print("No valid cells found in sample area!")

if __name__ == "__main__":
    debug_grid_cell_data()