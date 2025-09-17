#!/usr/bin/env python3
"""
Create spatial maps of climate/weather GDP ratios from Step 4 results.

This script calculates the ratio y_climate/(y_weather + epsilon) averaged over
the target period (2080-2100) and creates maps for all parameter combinations.
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.colors as mcolors
import os
from pathlib import Path


def create_ratio_maps_from_netcdf(netcdf_path, output_dir=None, epsilon=1e-20):
    """
    Create spatial maps of climate/weather GDP ratios from Step 4 NetCDF output.

    Parameters
    ----------
    netcdf_path : str
        Path to step4_forward_results_{model}.nc file
    output_dir : str, optional
        Output directory (defaults to same directory as NetCDF file)
    epsilon : float, optional
        Small value to prevent division by zero (default: 1e-20)

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

    # Define target period (2080-2100)
    target_start = 2080
    target_end = 2100
    target_indices = (years >= target_start) & (years <= target_end)
    n_target_years = np.sum(target_indices)
    print(f"Target period: {target_start}-{target_end} ({n_target_years} years)")

    # Set output directory and filename
    if output_dir is None:
        output_dir = Path(netcdf_path).parent

    pdf_filename = f"step4_ratio_maps_{Path(netcdf_path).stem}.pdf"
    pdf_path = os.path.join(output_dir, pdf_filename)

    # Total number of maps
    total_maps = len(ssp_names) * len(response_function_names) * len(target_names)
    print(f"Creating {total_maps} ratio maps...")

    # Create PDF with multi-page layout
    with PdfPages(pdf_path) as pdf:
        map_count = 0

        # Loop through all combinations
        for ssp_idx, ssp in enumerate(ssp_names):
            for target_idx, target_name in enumerate(target_names):
                for damage_idx, damage_name in enumerate(response_function_names):
                    map_count += 1

                    # Create new page
                    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

                    # Extract data for this combination
                    gdp_climate = ds.gdp_climate.isel(ssp=ssp_idx, damage_func=damage_idx, target=target_idx)  # [lat, lon, time]
                    gdp_weather = ds.gdp_weather.isel(ssp=ssp_idx, damage_func=damage_idx, target=target_idx)  # [lat, lon, time]

                    # Calculate ratio for target period at each grid cell
                    # Shape: [lat, lon, time] -> [lat, lon] (averaged over target period)
                    ratio_map = np.full((len(lat), len(lon)), np.nan)

                    for lat_idx in range(len(lat)):
                        for lon_idx in range(len(lon)):
                            if valid_mask[lat_idx, lon_idx]:
                                # Extract time series for this grid cell
                                climate_series = gdp_climate.isel(lat=lat_idx, lon=lon_idx).values[target_indices]
                                weather_series = gdp_weather.isel(lat=lat_idx, lon=lon_idx).values[target_indices]

                                # Calculate ratios for target period
                                ratios = climate_series / (weather_series + epsilon)

                                # Average over target period
                                if np.any(np.isfinite(ratios)):
                                    ratio_map[lat_idx, lon_idx] = np.nanmean(ratios)

                    # Create map
                    # Set up coordinate grids for plotting
                    lon_grid, lat_grid = np.meshgrid(lon, lat)

                    # Determine color scale
                    valid_ratios = ratio_map[valid_mask & np.isfinite(ratio_map)]
                    if len(valid_ratios) > 0:
                        # Center around 1.0 (no climate impact)
                        vmin = np.percentile(valid_ratios, 5)
                        vmax = np.percentile(valid_ratios, 95)

                        # Ensure colorbar is centered around 1.0
                        abs_max = max(abs(vmin - 1.0), abs(vmax - 1.0))
                        vmin = 1.0 - abs_max
                        vmax = 1.0 + abs_max

                        # Use diverging colormap centered at 1.0
                        cmap = plt.cm.RdBu_r  # Red for high ratios (less damage), blue for low ratios (more damage)
                        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)
                    else:
                        vmin, vmax = 0.9, 1.1
                        cmap = plt.cm.RdBu_r
                        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)

                    # Plot the map
                    masked_ratio = np.where(valid_mask, ratio_map, np.nan)
                    im = ax.pcolormesh(lon_grid, lat_grid, masked_ratio, cmap=cmap, norm=norm, shading='auto')

                    # Add coastlines (simplified)
                    ax.contour(lon_grid, lat_grid, valid_mask, levels=[0.5], colors='black', linewidths=0.5, alpha=0.7)

                    # Formatting
                    ax.set_xlabel('Longitude', fontsize=12)
                    ax.set_ylabel('Latitude', fontsize=12)
                    ax.set_title(f'{ssp.upper()} × {target_name} × {damage_name}\n'
                                f'Climate/Weather GDP Ratio (Target Period Mean: {target_start}-{target_end})',
                                fontsize=14, fontweight='bold')

                    # Add colorbar
                    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
                    cbar.set_label('GDP Climate/Weather Ratio', fontsize=11)

                    # Add reference line at 1.0
                    if hasattr(cbar, 'ax'):
                        cbar.ax.axhline(y=1.0, color='black', linestyle='-', linewidth=1, alpha=0.8)

                    # Add statistics text
                    if len(valid_ratios) > 0:
                        mean_ratio = np.nanmean(valid_ratios)
                        median_ratio = np.nanmedian(valid_ratios)
                        min_ratio = np.nanmin(valid_ratios)
                        max_ratio = np.nanmax(valid_ratios)

                        stats_text = f"Mean: {mean_ratio:.3f}\nMedian: {median_ratio:.3f}\nRange: [{min_ratio:.3f}, {max_ratio:.3f}]"
                        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                               fontsize=9, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                    # Add map count
                    ax.text(0.98, 0.98, f"Map {map_count}/{total_maps}", transform=ax.transAxes,
                           fontsize=9, verticalalignment='top', horizontalalignment='right',
                           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

                    plt.tight_layout()
                    pdf.savefig(fig, bbox_inches='tight', dpi=150)
                    plt.close(fig)

        print(f"Generated {map_count} ratio maps")

    # Close dataset
    ds.close()

    print(f"Ratio maps saved to: {pdf_path}")
    return pdf_path


def main():
    """Main function to run ratio map creation."""
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

    # Create ratio maps
    pdf_path = create_ratio_maps_from_netcdf(netcdf_path)
    print(f"\nRatio maps complete: {pdf_path}")


if __name__ == "__main__":
    main()