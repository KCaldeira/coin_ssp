#!/usr/bin/env python3
"""
Utility to compute global means of scaling factors from Step 3 NetCDF output.

This script analyzes the scaling factors computed during Step 3 optimization
and calculates area-weighted global means for each target/response function combination.
"""

import xarray as xr
import numpy as np
import sys
from pathlib import Path


def calculate_area_weighted_global_mean(data, lat, valid_mask):
    """Calculate area-weighted global mean using only valid grid cells."""
    # Convert to numpy arrays if needed
    if hasattr(data, 'values'):
        data = data.values
    if hasattr(lat, 'values'):
        lat = lat.values
    if hasattr(valid_mask, 'values'):
        valid_mask = valid_mask.values

    # Area weights (cosine of latitude)
    lat_rad = np.deg2rad(lat)
    weights = np.cos(lat_rad)

    # Create 2D weight grid
    weight_grid = np.broadcast_to(weights[:, np.newaxis], data.shape)

    # Apply valid mask
    masked_data = np.where(valid_mask, data, 0.0)
    masked_weights = np.where(valid_mask, weight_grid, 0.0)

    # Calculate weighted mean
    total_weight = np.sum(masked_weights)
    if total_weight > 0:
        return np.sum(masked_data * masked_weights) / total_weight
    else:
        return np.nan


def analyze_scaling_factors(netcdf_path):
    """
    Analyze scaling factors from Step 3 NetCDF file.

    Parameters
    ----------
    netcdf_path : str
        Path to step3_scaling_factors_*.nc file
    """
    print(f"Loading Step 3 scaling factors from: {netcdf_path}")

    # Load NetCDF file
    ds = xr.open_dataset(netcdf_path)

    # Extract coordinates and metadata
    lat = ds.lat.values
    lon = ds.lon.values
    valid_mask = ds.valid_mask.values

    # Extract dimension labels with backward compatibility
    if 'response_func' in ds.coords:
        response_function_names = [str(name) for name in ds.response_func.values]
        response_dim = 'response_func'
    elif 'damage_func' in ds.coords:  # Backward compatibility
        response_function_names = [str(name) for name in ds.damage_func.values]
        response_dim = 'damage_func'
    else:
        raise ValueError("Could not find response function dimension in NetCDF file")

    target_names = [str(t) for t in ds.target.values]

    print(f"Found {len(response_function_names)} response functions, {len(target_names)} targets")
    print(f"Valid grid cells: {np.sum(valid_mask)} of {valid_mask.size}")
    print()

    # Extract scaling factors array
    scaling_factors = ds.scaling_factors  # [lat, lon, response_func/damage_func, target]

    print("=" * 80)
    print("GLOBAL MEAN SCALING FACTORS")
    print("=" * 80)
    print(f"{'Target':<30} {'Response Function':<20} {'Global Mean':<12}")
    print("-" * 80)

    # Loop through all combinations
    results = {}
    for target_idx, target_name in enumerate(target_names):
        results[target_name] = {}

        for resp_idx, resp_name in enumerate(response_function_names):
            # Extract scaling factor for this combination
            if response_dim == 'response_func':
                scale_data = scaling_factors.isel(response_func=resp_idx, target=target_idx)
            else:  # damage_func
                scale_data = scaling_factors.isel(damage_func=resp_idx, target=target_idx)

            # Calculate global mean
            global_mean = calculate_area_weighted_global_mean(scale_data, lat, valid_mask)
            results[target_name][resp_name] = global_mean

            print(f"{target_name:<30} {resp_name:<20} {global_mean:<12.6f}")

    print("-" * 80)
    print()

    # Summary statistics
    print("SUMMARY STATISTICS")
    print("=" * 50)

    all_scaling_factors = []
    for target_name in results:
        for resp_name in results[target_name]:
            value = results[target_name][resp_name]
            if np.isfinite(value):
                all_scaling_factors.append(value)

    if all_scaling_factors:
        all_scaling_factors = np.array(all_scaling_factors)
        print(f"Overall mean scaling factor:    {np.mean(all_scaling_factors):.6f}")
        print(f"Standard deviation:             {np.std(all_scaling_factors):.6f}")
        print(f"Minimum scaling factor:         {np.min(all_scaling_factors):.6f}")
        print(f"Maximum scaling factor:         {np.max(all_scaling_factors):.6f}")
        print(f"Range:                          {np.max(all_scaling_factors) - np.min(all_scaling_factors):.6f}")

    print()
    print("Target-specific statistics:")
    print("-" * 30)

    for target_name in results:
        target_values = []
        for resp_name in results[target_name]:
            value = results[target_name][resp_name]
            if np.isfinite(value):
                target_values.append(value)

        if target_values:
            target_values = np.array(target_values)
            print(f"{target_name}:")
            print(f"  Mean: {np.mean(target_values):.6f}")
            print(f"  Std:  {np.std(target_values):.6f}")
            print(f"  Range: [{np.min(target_values):.6f}, {np.max(target_values):.6f}]")

    # Close dataset
    ds.close()

    return results


def main():
    """Main function to run scaling factor analysis."""
    if len(sys.argv) > 1:
        netcdf_path = sys.argv[1]
    else:
        # Use default path
        netcdf_path = "./data/output/step3_scaling_factors_CanESM5_ssp245.nc"

    if not Path(netcdf_path).exists():
        print(f"Error: NetCDF file not found: {netcdf_path}")
        return

    # Analyze scaling factors
    results = analyze_scaling_factors(netcdf_path)

    print(f"\nAnalysis complete for: {netcdf_path}")


if __name__ == "__main__":
    main()