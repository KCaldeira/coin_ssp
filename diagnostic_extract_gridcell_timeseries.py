#!/usr/bin/env python3
"""
Diagnostic tool to extract time series data for specific grid cells.

This script identifies grid cells with maximum and minimum regression slopes
from Step 3 results, then extracts all time-varying data from all_data
for those locations and writes to CSV files.

Usage:
    python diagnostic_extract_gridcell_timeseries.py <config_file> <step3_netcdf_file>

Example:
    python diagnostic_extract_gridcell_timeseries.py \
        coin_ssp_config_linear_parameter_sensitivity.json \
        data/output/output_CanESM5_20251002_095048/linear-sensitivity/step3_linear-sensitivity_CanESM5_ssp245_scaling_factors.nc
"""

import sys
import json
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import Dict, Any, Tuple

from coin_ssp_utils import load_all_data


def find_extreme_slope_locations(step3_nc_path: str, response_idx: int = 0, target_idx: int = 0) -> Tuple[Dict, Dict]:
    """
    Find grid cell locations with maximum and minimum regression slopes.

    Parameters
    ----------
    step3_nc_path : str
        Path to Step 3 NetCDF file containing regression slopes
    response_idx : int
        Response function index to analyze (default: 0)
    target_idx : int
        Target index to analyze (default: 0)

    Returns
    -------
    Tuple[Dict, Dict]
        (max_info, min_info) where each dict contains:
        - lat_idx, lon_idx: array indices
        - lat, lon: coordinate values
        - slope: regression slope value
    """
    print(f"\nAnalyzing Step 3 results: {step3_nc_path}")

    # Load Step 3 NetCDF file
    ds = xr.open_dataset(step3_nc_path)

    # Extract regression slopes and success mask
    regression_slopes = ds['regression_slopes'].values  # [response_func, target, lat, lon]
    success_mask = ds['regression_success_mask'].values  # [response_func, target, lat, lon]
    lat = ds['lat'].values
    lon = ds['lon'].values

    # Get response function and target names
    response_names = ds['response_function_names'].values
    target_names = ds['target_names'].values

    print(f"Response function: {response_names[response_idx]}")
    print(f"Target: {target_names[target_idx]}")

    # Extract slopes for selected response function and target
    slopes = regression_slopes[response_idx, target_idx, :, :]
    success = success_mask[response_idx, target_idx, :, :]

    # Find valid slopes
    valid_slopes = slopes[success & np.isfinite(slopes)]

    if len(valid_slopes) == 0:
        raise ValueError("No valid regression slopes found!")

    print(f"Valid regression slopes: {len(valid_slopes)} / {success.size}")
    print(f"Slope range: {np.min(valid_slopes):.6f} to {np.max(valid_slopes):.6f}")

    # Find maximum slope location
    max_val = np.max(valid_slopes)
    max_indices = np.where((success & np.isfinite(slopes)) & (slopes == max_val))
    max_lat_idx, max_lon_idx = max_indices[0][0], max_indices[1][0]

    max_info = {
        'lat_idx': max_lat_idx,
        'lon_idx': max_lon_idx,
        'lat': lat[max_lat_idx],
        'lon': lon[max_lon_idx],
        'slope': max_val
    }

    # Find minimum slope location
    min_val = np.min(valid_slopes)
    min_indices = np.where((success & np.isfinite(slopes)) & (slopes == min_val))
    min_lat_idx, min_lon_idx = min_indices[0][0], min_indices[1][0]

    min_info = {
        'lat_idx': min_lat_idx,
        'lon_idx': min_lon_idx,
        'lat': lat[min_lat_idx],
        'lon': lon[min_lon_idx],
        'slope': min_val
    }

    print(f"\nMaximum slope: {max_val:.6f} at lat={max_info['lat']:.2f}, lon={max_info['lon']:.2f} (indices: [{max_lat_idx},{max_lon_idx}])")
    print(f"Minimum slope: {min_val:.6f} at lat={min_info['lat']:.2f}, lon={min_info['lon']:.2f} (indices: [{min_lat_idx},{min_lon_idx}])")

    ds.close()
    return max_info, min_info


def extract_timeseries_for_location(all_data: Dict[str, Any], lat_idx: int, lon_idx: int) -> pd.DataFrame:
    """
    Extract all time-varying data for a specific grid cell location.

    Parameters
    ----------
    all_data : Dict[str, Any]
        Full loaded data structure from load_all_data()
    lat_idx : int
        Latitude index
    lon_idx : int
        Longitude index

    Returns
    -------
    pd.DataFrame
        Time series data with years as index and variables as columns
    """
    print(f"\nExtracting time series for grid cell [{lat_idx},{lon_idx}]...")

    # Get years array
    years = all_data['years']

    # Initialize data dictionary
    data_dict = {}

    # Extract data from each SSP
    for ssp_key in all_data.keys():
        if ssp_key.startswith('_') or ssp_key == 'years' or ssp_key in ['tas0_2d', 'pr0_2d']:
            continue  # Skip metadata and reference fields

        ssp_data = all_data[ssp_key]

        # Extract all variables with time dimension
        for var_name, var_data in ssp_data.items():
            if var_name.startswith('_'):
                continue  # Skip metadata

            # Check if this is a time-varying array [time, lat, lon]
            if isinstance(var_data, np.ndarray) and var_data.ndim == 3:
                if var_data.shape[0] == len(years):
                    # Extract time series for this grid cell
                    timeseries = var_data[:, lat_idx, lon_idx]
                    column_name = f"{ssp_key}_{var_name}"
                    data_dict[column_name] = timeseries

    # Create DataFrame with years as index
    df = pd.DataFrame(data_dict, index=years)
    df.index.name = 'year'

    print(f"Extracted {len(df.columns)} time series variables across {len(df)} years")

    return df


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    config_file = sys.argv[1]
    step3_nc_file = sys.argv[2]

    # Optional: response and target indices
    response_idx = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    target_idx = int(sys.argv[4]) if len(sys.argv) > 4 else 0

    print("="*80)
    print("DIAGNOSTIC: Grid Cell Time Series Extraction")
    print("="*80)
    print(f"Config file: {config_file}")
    print(f"Step 3 NetCDF: {step3_nc_file}")
    print(f"Response function index: {response_idx}")
    print(f"Target index: {target_idx}")

    # Load configuration
    with open(config_file, 'r') as f:
        config = json.load(f)

    # Find extreme slope locations
    max_info, min_info = find_extreme_slope_locations(step3_nc_file, response_idx, target_idx)

    # Load all data
    print("\nLoading all data...")
    all_data = load_all_data(config)

    # Extract time series for maximum slope location
    print("\n" + "="*80)
    print("MAXIMUM SLOPE LOCATION")
    print("="*80)
    max_df = extract_timeseries_for_location(all_data, max_info['lat_idx'], max_info['lon_idx'])

    # Add metadata columns
    max_df.insert(0, 'lon', max_info['lon'])
    max_df.insert(0, 'lat', max_info['lat'])
    max_df.insert(0, 'lon_idx', max_info['lon_idx'])
    max_df.insert(0, 'lat_idx', max_info['lat_idx'])
    max_df.insert(0, 'slope', max_info['slope'])

    # Extract time series for minimum slope location
    print("\n" + "="*80)
    print("MINIMUM SLOPE LOCATION")
    print("="*80)
    min_df = extract_timeseries_for_location(all_data, min_info['lat_idx'], min_info['lon_idx'])

    # Add metadata columns
    min_df.insert(0, 'lon', min_info['lon'])
    min_df.insert(0, 'lat', min_info['lat'])
    min_df.insert(0, 'lon_idx', min_info['lon_idx'])
    min_df.insert(0, 'lat_idx', min_info['lat_idx'])
    min_df.insert(0, 'slope', min_info['slope'])

    # Determine output directory from step3 file path
    step3_path = Path(step3_nc_file)
    output_dir = step3_path.parent

    # Generate output filenames
    json_id = config['run_metadata']['json_id']
    model_name = config['climate_model']['model_name']
    reference_ssp = config['ssp_scenarios']['reference_ssp']

    max_csv_path = output_dir / f"diagnostic_{json_id}_{model_name}_{reference_ssp}_max_slope_timeseries.csv"
    min_csv_path = output_dir / f"diagnostic_{json_id}_{model_name}_{reference_ssp}_min_slope_timeseries.csv"

    # Write CSV files
    max_df.to_csv(max_csv_path, float_format='%.6f')
    min_df.to_csv(min_csv_path, float_format='%.6f')

    print("\n" + "="*80)
    print("OUTPUT FILES")
    print("="*80)
    print(f"Maximum slope timeseries: {max_csv_path}")
    print(f"  Location: lat={max_info['lat']:.2f}, lon={max_info['lon']:.2f}")
    print(f"  Slope: {max_info['slope']:.6f}")
    print(f"  Variables: {len(max_df.columns) - 5}")  # Subtract metadata columns
    print()
    print(f"Minimum slope timeseries: {min_csv_path}")
    print(f"  Location: lat={min_info['lat']:.2f}, lon={min_info['lon']:.2f}")
    print(f"  Slope: {min_info['slope']:.6f}")
    print(f"  Variables: {len(min_df.columns) - 5}")  # Subtract metadata columns
    print()
    print("Done!")


if __name__ == "__main__":
    main()
