#!/usr/bin/env python3
"""
Debug TFP calculation to understand why capital stock becomes zero/negative
"""
import numpy as np
import json
from coin_ssp_utils import load_all_netcdf_data, get_ssp_data, calculate_tfp_coin_ssp
from coin_ssp_core import ModelParams

def debug_tfp_on_sample_cells():
    """Debug TFP calculation on a few sample grid cells"""

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

    # Find first few valid grid cells
    nlat, nlon, ntime = gdp_data.shape
    valid_cells = []

    # Check overall data statistics first
    print(f"\nData statistics:")
    print(f"GDP non-zero cells: {np.sum(gdp_data[:, :, 0] > 0)} / {nlat*nlon}")
    print(f"Pop non-zero cells: {np.sum(pop_data[:, :, 0] > 0)} / {nlat*nlon}")
    print(f"Both non-zero cells: {np.sum((gdp_data[:, :, 0] > 0) & (pop_data[:, :, 0] > 0))}")

    print(f"GDP range: {np.min(gdp_data):.2e} to {np.max(gdp_data):.2e}")
    print(f"Pop range: {np.min(pop_data):.2e} to {np.max(pop_data):.2e}")

    print("\nSearching for valid grid cells...")
    for lat_idx in range(nlat):
        for lon_idx in range(nlon):
            if gdp_data[lat_idx, lon_idx, 0] > 0 and pop_data[lat_idx, lon_idx, 0] > 0:
                valid_cells.append((lat_idx, lon_idx))
                print(f"Valid cell found at ({lat_idx}, {lon_idx})")
                if len(valid_cells) >= 3:  # Only test first 3 valid cells
                    break
        if len(valid_cells) >= 3:
            break

    if not valid_cells:
        print("No valid cells found!")
        return

    # Test TFP calculation on each valid cell
    params = ModelParams()
    print(f"\nModel parameters: s={params.s}, alpha={params.alpha}, delta={params.delta}")

    for i, (lat_idx, lon_idx) in enumerate(valid_cells):
        print(f"\n" + "="*60)
        print(f"TESTING GRID CELL {i+1}: ({lat_idx}, {lon_idx})")
        print("="*60)

        # Extract time series
        gdp_series = gdp_data[lat_idx, lon_idx, :]
        pop_series = pop_data[lat_idx, lon_idx, :]

        print(f"Time series length: {len(gdp_series)}")
        print(f"GDP first/last: {gdp_series[0]:.2e} / {gdp_series[-1]:.2e}")
        print(f"Pop first/last: {pop_series[0]:.2e} / {pop_series[-1]:.2e}")

        # Check for any suspicious patterns
        gdp_growth = gdp_series[1:] / gdp_series[:-1]
        pop_growth = pop_series[1:] / pop_series[:-1]

        print(f"GDP growth range: {np.min(gdp_growth):.4f} to {np.max(gdp_growth):.4f}")
        print(f"Pop growth range: {np.min(pop_growth):.4f} to {np.max(pop_growth):.4f}")

        # Look for extreme values
        if np.any(gdp_growth < 0.5) or np.any(gdp_growth > 2.0):
            extreme_indices = np.where((gdp_growth < 0.5) | (gdp_growth > 2.0))[0]
            print(f"WARNING: Extreme GDP growth at indices: {extreme_indices}")

        try:
            # Attempt TFP calculation
            print("\nAttempting TFP calculation...")
            tfp_result, k_result = calculate_tfp_coin_ssp(pop_series, gdp_series, params)
            print("TFP calculation completed successfully!")

        except Exception as e:
            print(f"TFP calculation failed: {e}")
            continue

if __name__ == "__main__":
    debug_tfp_on_sample_cells()