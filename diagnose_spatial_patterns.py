#!/usr/bin/env python3

import xarray as xr
import numpy as np
from coin_ssp_netcdf import calculate_global_mean

def diagnose_spatial_patterns():
    """Diagnose the spatial patterns to see if they make sense."""
    
    # Load results
    ds = xr.open_dataset('data/output/target_gdp_reductions.nc')
    
    temp_ref = ds.temperature_ref.values
    gdp_target = ds.gdp_target.values  
    linear_reduction = ds.target_gdp_reductions.values[1]  # Linear layer
    lat = ds.lat.values
    
    print("=== Spatial Pattern Diagnosis ===\n")
    
    # Find some representative points
    temp_flat = temp_ref.flatten()
    linear_flat = linear_reduction.flatten()
    
    # Sort by temperature to see the relationship
    sort_idx = np.argsort(temp_flat)
    temps_sorted = temp_flat[sort_idx]
    reductions_sorted = linear_flat[sort_idx]
    
    # Show temperature vs reduction relationship
    print("Temperature vs Linear Reduction (sample points):")
    print("Temp(°C)  Reduction   Expected?")
    print("-" * 35)
    
    # Sample every 1000 points to see the pattern
    sample_indices = np.arange(0, len(temps_sorted), len(temps_sorted)//20)
    for i in sample_indices:
        temp = temps_sorted[i]
        reduction = reductions_sorted[i]
        print(f"{temp:8.1f}  {reduction:9.4f}")
    
    print(f"\nLinear function coefficients:")
    print(f"a0 = {-3.337902:.6f}")
    print(f"a1 = {0.102930:.6f}")
    print(f"So: reduction(T) = -3.337902 + 0.102930 * T")
    
    print(f"\nExpected values:")
    for test_temp in [-60, -20, 0, 20, 30]:
        expected = -3.337902 + 0.102930 * test_temp
        print(f"At {test_temp:3.0f}°C: {expected:.4f}")
    
    # Check if this matches what we see
    print(f"\nActual temperature range: {temp_ref.min():.1f}°C to {temp_ref.max():.1f}°C")
    print(f"Actual reduction range: {linear_reduction.min():.4f} to {linear_reduction.max():.4f}")
    
    # Find specific temperature examples from the grid
    print(f"\nSpecific grid cell examples:")
    
    # Find coldest points
    cold_mask = temp_ref < -50
    if np.any(cold_mask):
        cold_temps = temp_ref[cold_mask]
        cold_reductions = linear_reduction[cold_mask]
        print(f"Cold regions (<-50°C): {len(cold_temps)} points")
        print(f"  Temp range: {cold_temps.min():.1f}°C to {cold_temps.max():.1f}°C") 
        print(f"  Reduction range: {cold_reductions.min():.4f} to {cold_reductions.max():.4f}")
    
    # Find warm points
    warm_mask = temp_ref > 25
    if np.any(warm_mask):
        warm_temps = temp_ref[warm_mask]
        warm_reductions = linear_reduction[warm_mask]
        print(f"Warm regions (>25°C): {len(warm_temps)} points")
        print(f"  Temp range: {warm_temps.min():.1f}°C to {warm_temps.max():.1f}°C")
        print(f"  Reduction range: {warm_reductions.min():.4f} to {warm_reductions.max():.4f}")

if __name__ == "__main__":
    diagnose_spatial_patterns()