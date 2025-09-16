#!/usr/bin/env python3
"""
Standalone test script for Step 4 visualization using NetCDF output data.

This script reads Step 4 results from NetCDF file and creates the PDF visualization
without needing to run the full pipeline or access SSP input data.
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
from pathlib import Path


def calculate_global_mean_from_netcdf(data, lat, valid_mask):
    """Calculate area-weighted global mean using only valid grid cells."""
    # Convert to numpy if xarray
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


def create_step4_test_visualization(netcdf_path, output_dir=None):
    """
    Create PDF visualization from Step 4 NetCDF output file.

    Parameters
    ----------
    netcdf_path : str
        Path to step4_forward_results_{model}.nc file
    output_dir : str, optional
        Output directory (defaults to same directory as NetCDF file)

    Returns
    -------
    str
        Path to generated PDF file
    """
    print(f"Loading Step 4 results from: {netcdf_path}")

    # Load NetCDF file
    ds = xr.open_dataset(netcdf_path)

    # Extract coordinates and metadata
    lat = ds.lat.values
    lon = ds.lon.values
    time = ds.time.values

    # Convert time index to actual years (projection period 2015-2100)
    start_year = 2015
    years = start_year + time
    valid_mask = ds.valid_mask.values

    # Extract dimension labels
    ssp_names = [str(s) for s in ds.ssp.values]
    response_function_names = [str(d) for d in ds.damage_func.values]
    target_names = [str(t) for t in ds.target.values]

    print(f"Found {len(ssp_names)} SSPs, {len(response_function_names)} response functions, {len(target_names)} targets")
    print(f"Time series: {len(years)} years ({years[0]:.0f}-{years[-1]:.0f})")
    print(f"Valid grid cells: {np.sum(valid_mask)} of {valid_mask.size}")

    # Set output directory and filename
    if output_dir is None:
        output_dir = Path(netcdf_path).parent

    pdf_filename = f"test_step4_visualization_{Path(netcdf_path).stem}.pdf"
    pdf_path = os.path.join(output_dir, pdf_filename)

    # Create PDF with multi-page layout
    with PdfPages(pdf_path) as pdf:
        page_count = 0

        # Loop through all combinations
        for target_idx, target_name in enumerate(target_names):
            for damage_idx, damage_name in enumerate(response_function_names):
                for ssp_idx, ssp in enumerate(ssp_names):
                    page_count += 1

                    # Create new page
                    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

                    # Extract data for this combination
                    gdp_climate = ds.gdp_climate.isel(ssp=ssp_idx, damage_func=damage_idx, target=target_idx)  # [lat, lon, time]
                    gdp_weather = ds.gdp_weather.isel(ssp=ssp_idx, damage_func=damage_idx, target=target_idx)  # [lat, lon, time]

                    # Calculate global mean time series
                    ntime = len(years)
                    y_climate_series = np.zeros(ntime)
                    y_weather_series = np.zeros(ntime)

                    for t in range(ntime):
                        # Extract spatial slice for this time
                        gdp_climate_t = gdp_climate.isel(time=t).values  # [lat, lon]
                        gdp_weather_t = gdp_weather.isel(time=t).values  # [lat, lon]

                        # Calculate global means
                        y_climate_series[t] = calculate_global_mean_from_netcdf(gdp_climate_t, lat, valid_mask)
                        y_weather_series[t] = calculate_global_mean_from_netcdf(gdp_weather_t, lat, valid_mask)

                    # Plot the time series (no baseline - using Step 4 output only)
                    ax.plot(years, y_climate_series, 'r-', linewidth=2, label='GDP with Climate Effects')
                    ax.plot(years, y_weather_series, 'b--', linewidth=2, label='GDP with Weather Only')

                    # Calculate and show climate impact
                    if np.any(np.isfinite(y_climate_series)) and np.any(np.isfinite(y_weather_series)):
                        # Find target period (last 20 years)
                        target_years = years >= (years[-1] - 20)
                        if np.any(target_years):
                            climate_mean = np.nanmean(y_climate_series[target_years])
                            weather_mean = np.nanmean(y_weather_series[target_years])
                            if weather_mean > 0:
                                impact_pct = ((climate_mean / weather_mean) - 1) * 100
                                impact_text = f"Climate Impact (target period): {impact_pct:.1f}%"
                                ax.text(0.02, 0.02, impact_text, transform=ax.transAxes,
                                       fontsize=10, verticalalignment='bottom',
                                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

                    # Formatting
                    ax.set_xlabel('Year', fontsize=12)
                    ax.set_ylabel('Global Mean GDP (Area-Weighted)', fontsize=12)
                    ax.set_title(f'{target_name} × {damage_name} × {ssp.upper()}\n'
                                f'Step 4 Forward Model Results',
                                fontsize=14, fontweight='bold')
                    ax.legend(fontsize=10)
                    ax.grid(True, alpha=0.3)

                    # Set reasonable y-axis limits
                    all_values = np.concatenate([y_climate_series, y_weather_series])
                    valid_values = all_values[np.isfinite(all_values)]
                    if len(valid_values) > 0:
                        y_min, y_max = np.percentile(valid_values, [1, 99])
                        y_range = y_max - y_min
                        if y_range > 0:
                            ax.set_ylim(y_min - 0.05*y_range, y_max + 0.05*y_range)

                    plt.tight_layout()
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)

        print(f"Generated {page_count} pages in test visualization")

    # Close dataset
    ds.close()

    print(f"Test visualization saved to: {pdf_path}")
    return pdf_path


def main():
    """Main function to run test visualization."""
    import sys

    if len(sys.argv) > 1:
        netcdf_path = sys.argv[1]
    else:
        # Find most recent Step 4 output
        import glob
        pattern = "data/output/*/step4_forward_results_*.nc"
        files = glob.glob(pattern)
        if not files:
            print("No Step 4 NetCDF files found. Please run integrated pipeline first.")
            return

        # Use most recent file
        netcdf_path = max(files, key=os.path.getmtime)
        print(f"Using most recent Step 4 output: {netcdf_path}")

    # Create visualization
    pdf_path = create_step4_test_visualization(netcdf_path)
    print(f"\nVisualization complete: {pdf_path}")


if __name__ == "__main__":
    main()