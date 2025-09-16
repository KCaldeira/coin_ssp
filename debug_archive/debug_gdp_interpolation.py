#!/usr/bin/env python3
"""
Debug GDP interpolation to understand zero value patterns.
"""
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from coin_ssp_utils import extract_year_coordinate, interpolate_to_annual_grid
import json

def analyze_gdp_interpolation():
    """Analyze GDP data before and after interpolation"""

    print("="*60)
    print("ANALYZING GDP INTERPOLATION")
    print("="*60)

    # Load GDP data directly
    print("Loading GDP data...")
    gdp_ds = xr.open_dataset('data/input/gridded_gdp_regrid_CanESM5_ssp245.nc', decode_times=False)
    gdp_raw_all = gdp_ds.gdp_20202100ssp5.values
    gdp_years_raw, gdp_valid_mask = extract_year_coordinate(gdp_ds)
    gdp_raw = gdp_raw_all[gdp_valid_mask]

    lat = gdp_ds.lat.values
    lon = gdp_ds.lon.values

    print(f"Original GDP data shape: {gdp_raw.shape}")
    print(f"GDP years: {gdp_years_raw}")
    print(f"Year range: {gdp_years_raw.min()} to {gdp_years_raw.max()}")

    # Create interpolated version
    print("Creating interpolated data...")
    gdp_years_annual = np.arange(gdp_years_raw.min(), gdp_years_raw.max() + 1)
    gdp_interpolated = interpolate_to_annual_grid(gdp_years_raw, gdp_raw, gdp_years_annual)

    print(f"Interpolated GDP data shape: {gdp_interpolated.shape}")
    print(f"Interpolated years: {len(gdp_years_annual)} points from {gdp_years_annual.min()} to {gdp_years_annual.max()}")

    # Count zeros in each grid cell
    print("Counting zeros...")
    nlat, nlon = gdp_raw.shape[1], gdp_raw.shape[2]

    zeros_original = np.zeros((nlat, nlon))
    zeros_interpolated = np.zeros((nlat, nlon))

    for lat_idx in range(nlat):
        for lon_idx in range(nlon):
            # Count zeros in original data
            original_series = gdp_raw[:, lat_idx, lon_idx]
            zeros_original[lat_idx, lon_idx] = np.sum(original_series == 0)

            # Count zeros in interpolated data
            interp_series = gdp_interpolated[:, lat_idx, lon_idx]
            zeros_interpolated[lat_idx, lon_idx] = np.sum(interp_series == 0)

    print(f"Original data - Total zero values: {np.sum(zeros_original)}")
    print(f"Interpolated data - Total zero values: {np.sum(zeros_interpolated)}")

    # Save data to NetCDF files for examination
    print("Saving data files...")

    # Save original data
    original_ds = xr.Dataset({
        'gdp': (['time', 'lat', 'lon'], gdp_raw),
        'zero_count': (['lat', 'lon'], zeros_original)
    }, coords={
        'time': gdp_years_raw,
        'lat': lat,
        'lon': lon
    })
    original_ds.to_netcdf('debug_gdp_original.nc')

    # Save interpolated data
    interp_ds = xr.Dataset({
        'gdp': (['time', 'lat', 'lon'], gdp_interpolated),
        'zero_count': (['lat', 'lon'], zeros_interpolated)
    }, coords={
        'time': gdp_years_annual,
        'lat': lat,
        'lon': lon
    })
    interp_ds.to_netcdf('debug_gdp_interpolated.nc')

    print("Files saved:")
    print("  debug_gdp_original.nc - Original GDP data and zero counts")
    print("  debug_gdp_interpolated.nc - Interpolated GDP data and zero counts")

    # Create maps
    create_zero_count_maps(lat, lon, zeros_original, zeros_interpolated)

    # Sample a few problematic grid cells
    analyze_sample_cells(gdp_raw, gdp_interpolated, gdp_years_raw, gdp_years_annual,
                        zeros_original, zeros_interpolated, lat, lon)

    gdp_ds.close()
    print("Analysis complete!")

def create_zero_count_maps(lat, lon, zeros_original, zeros_interpolated):
    """Create maps showing zero counts"""

    print("Creating zero count maps...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Original data map
    im1 = ax1.imshow(zeros_original, cmap='viridis', aspect='auto')
    ax1.set_title(f'Original GDP Data - Zero Counts\nMax zeros: {np.max(zeros_original):.0f}')
    ax1.set_xlabel('Longitude index')
    ax1.set_ylabel('Latitude index')
    plt.colorbar(im1, ax=ax1)

    # Interpolated data map
    im2 = ax2.imshow(zeros_interpolated, cmap='viridis', aspect='auto')
    ax2.set_title(f'Interpolated GDP Data - Zero Counts\nMax zeros: {np.max(zeros_interpolated):.0f}')
    ax2.set_xlabel('Longitude index')
    ax2.set_ylabel('Latitude index')
    plt.colorbar(im2, ax=ax2)

    plt.tight_layout()
    plt.savefig('debug_gdp_zero_maps.pdf', dpi=150, bbox_inches='tight')
    print("  Saved: debug_gdp_zero_maps.pdf")

    # Statistics
    print(f"Zero count statistics:")
    print(f"  Original - Mean: {np.mean(zeros_original):.1f}, Max: {np.max(zeros_original):.0f}")
    print(f"  Interpolated - Mean: {np.mean(zeros_interpolated):.1f}, Max: {np.max(zeros_interpolated):.0f}")

def analyze_sample_cells(gdp_raw, gdp_interpolated, years_raw, years_annual,
                        zeros_original, zeros_interpolated, lat, lon):
    """Analyze a few sample grid cells in detail"""

    print("\nAnalyzing sample grid cells...")

    # Find cells with different zero patterns
    zero_diff = zeros_interpolated - zeros_original

    # Find cells where interpolation created more zeros
    increased_zeros = np.where(zero_diff > 0)
    if len(increased_zeros[0]) > 0:
        print(f"Found {len(increased_zeros[0])} cells where interpolation increased zeros")

        # Sample first few
        for i in range(min(3, len(increased_zeros[0]))):
            lat_idx, lon_idx = increased_zeros[0][i], increased_zeros[1][i]

            print(f"\nCell ({lat_idx}, {lon_idx}) - Lat: {lat[lat_idx]:.2f}, Lon: {lon[lon_idx]:.2f}")

            original_series = gdp_raw[:, lat_idx, lon_idx]
            interp_series = gdp_interpolated[:, lat_idx, lon_idx]

            print(f"  Original: {len(original_series)} points, {np.sum(original_series == 0)} zeros")
            print(f"  Interpolated: {len(interp_series)} points, {np.sum(interp_series == 0)} zeros")
            print(f"  Original data: {original_series}")
            print(f"  Original years: {years_raw}")
            print(f"  Non-zero original values: {original_series[original_series > 0]}")

    # Also check some cells that should have valid data
    print(f"\nSample of non-zero original cells:")
    nonzero_cells = np.where((zeros_original == 0) & (zeros_interpolated > 0))
    if len(nonzero_cells[0]) > 0:
        print(f"Found {len(nonzero_cells[0])} cells that were non-zero originally but have zeros after interpolation")

        for i in range(min(2, len(nonzero_cells[0]))):
            lat_idx, lon_idx = nonzero_cells[0][i], nonzero_cells[1][i]

            print(f"\nCell ({lat_idx}, {lon_idx}) - Lat: {lat[lat_idx]:.2f}, Lon: {lon[lon_idx]:.2f}")

            original_series = gdp_raw[:, lat_idx, lon_idx]
            interp_series = gdp_interpolated[:, lat_idx, lon_idx]

            print(f"  Original ({len(years_raw)} points): {original_series}")
            print(f"  Interpolated ({len(years_annual)} points): {interp_series[:10]}..{interp_series[-5:]}")

if __name__ == "__main__":
    analyze_gdp_interpolation()