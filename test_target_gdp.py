#!/usr/bin/env python3

import xarray as xr
import numpy as np
import json
from coin_ssp_utils import calculate_global_mean

def test_target_gdp_results():
    """Test and verify the target GDP reduction results."""
    
    # Load config to get reference temperatures
    with open('target_gdp_config_example.json', 'r') as f:
        config = json.load(f)
    
    linear_ref_temp = config['linear_target']['reference_temperature']
    quad_ref_temp = config['quadratic_target']['reference_temperature']
    quad_zero_temp = config['quadratic_target']['zero_reduction_temperature']
    
    # Load results
    results = xr.open_dataset('data/output/target_gdp_reductions.nc')
    
    print("=== Target GDP Reduction Results ===\n")
    
    # Print dataset info
    print("Dataset variables:", list(results.data_vars.keys()))
    print("Dataset dimensions:", dict(results.dims))
    print()
    
    # Extract arrays
    reductions = results.target_gdp_reductions.values  # (3, 64, 128)
    temp_ref = results.temperature_ref.values          # (64, 128)
    gdp_target = results.gdp_target.values             # (64, 128) 
    lat = results.lat.values
    lon = results.lon.values
    
    # Calculate global means for verification
    constant_global = calculate_global_mean(reductions[0], lat)
    linear_global = calculate_global_mean(reductions[1], lat) 
    quadratic_global = calculate_global_mean(reductions[2], lat)
    temp_global = calculate_global_mean(temp_ref, lat)
    gdp_global = calculate_global_mean(gdp_target, lat)
    
    # Test GDP-weighted global means to verify constraints
    gdp_weighted_constant = calculate_global_mean(gdp_target * (1 + reductions[0]), lat) / gdp_global - 1
    gdp_weighted_linear = calculate_global_mean(gdp_target * (1 + reductions[1]), lat) / gdp_global - 1
    gdp_weighted_quadratic = calculate_global_mean(gdp_target * (1 + reductions[2]), lat) / gdp_global - 1
    
    print(f"Global mean reference temperature: {temp_global:.2f}°C")
    print(f"Global mean constant reduction: {constant_global:.4f}")
    print(f"Global mean linear reduction: {linear_global:.4f}")  
    print(f"Global mean quadratic reduction: {quadratic_global:.4f}")
    print(f"GDP-weighted constant reduction: {gdp_weighted_constant:.4f} (target: {config['constant_target']['gdp_reduction']})")
    print(f"GDP-weighted linear reduction: {gdp_weighted_linear:.4f} (target: {config['linear_target']['global_mean_reduction']})")
    print(f"GDP-weighted quadratic reduction: {gdp_weighted_quadratic:.4f} (target: {config['quadratic_target']['global_mean_reduction']})")
    print()
    
    # Test specific temperature points
    print("=== Verification at specific temperatures ===")
    
    # Find grid cells closest to specific temperatures
    temp_flatten = temp_ref.flatten()
    
    # Test at reference temperature
    idx_ref = np.argmin(np.abs(temp_flatten - linear_ref_temp))
    temp_ref_actual = temp_flatten[idx_ref]
    lat_idx, lon_idx = np.unravel_index(idx_ref, temp_ref.shape)
    
    print(f"At T ≈ {linear_ref_temp}°C (actual: {temp_ref_actual:.1f}°C):")
    print(f"  Constant: {reductions[0, lat_idx, lon_idx]:.4f}")
    print(f"  Linear: {reductions[1, lat_idx, lon_idx]:.4f} (config target: {config['linear_target']['reduction_at_reference_temp']})")
    print(f"  Quadratic: {reductions[2, lat_idx, lon_idx]:.4f} (config target: {config['quadratic_target']['reduction_at_reference_temp']})")
    print()
    
    # Test at zero reduction temperature for quadratic
    idx_zero = np.argmin(np.abs(temp_flatten - quad_zero_temp))
    temp_zero = temp_flatten[idx_zero]
    lat_idx_zero, lon_idx_zero = np.unravel_index(idx_zero, temp_ref.shape)
    
    print(f"At T ≈ {quad_zero_temp}°C (actual: {temp_zero:.1f}°C):")
    print(f"  Constant: {reductions[0, lat_idx_zero, lon_idx_zero]:.4f}")
    print(f"  Linear: {reductions[1, lat_idx_zero, lon_idx_zero]:.4f}")
    print(f"  Quadratic: {reductions[2, lat_idx_zero, lon_idx_zero]:.4f} (should be ≈ 0)")
    print()
    
    # Print range statistics
    print("=== Range Statistics ===")
    for i, name in enumerate(['Constant', 'Linear', 'Quadratic']):
        data = reductions[i]
        print(f"{name}:")
        print(f"  Min: {data.min():.4f}")
        print(f"  Max: {data.max():.4f}")
        print(f"  Mean: {data.mean():.4f}")
        print(f"  Std: {data.std():.4f}")
        print()
    
    print("=== Temperature Range ===")
    print(f"Temperature min: {temp_ref.min():.1f}°C")
    print(f"Temperature max: {temp_ref.max():.1f}°C")
    print(f"Temperature mean: {temp_ref.mean():.1f}°C")

if __name__ == "__main__":
    test_target_gdp_results()