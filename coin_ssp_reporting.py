"""
Visualization and Reporting Module for COIN_SSP

This module contains all visualization and reporting functions including:
- Forward model visualization functions
- Target GDP visualization
- Scaling factors visualization
- Objective function visualization
- TFP baseline visualization
- GDP weighted scaling summaries

Extracted from coin_ssp_utils.py and main.py for better organization.
"""

import copy
import json
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
import xarray as xr
from datetime import datetime
from typing import Dict, Any, List
from coin_ssp_models import ScalingParams

# Import functions from other coin_ssp modules
from coin_ssp_netcdf import (
    create_serializable_config, extract_year_coordinate
)
from coin_ssp_math_utils import (
    apply_time_series_filter, calculate_zero_biased_range, calculate_zero_biased_axis_range,
    calculate_area_weights, calculate_time_means, calculate_global_mean
)

# Import get_grid_metadata function
def get_grid_metadata(all_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract grid metadata from loaded NetCDF data structure.

    Parameters
    ----------
    all_data : Dict[str, Any]
        Result from load_all_data()

    Returns
    -------
    Dict[str, Any]
        Metadata dictionary containing coordinates and dimensions
    """
    return {
        'lat': all_data['_metadata']['lat'],
        'lon': all_data['_metadata']['lon'],
        'nlat': len(all_data['_metadata']['lat']),
        'nlon': len(all_data['_metadata']['lon']),
    }


def get_adaptive_subplot_layout(n_targets):
    """
    Calculate optimal subplot layout based on number of targets.

    For 3 or fewer targets: single column layout
    For 4+ targets: two-column layout to maintain reasonable aspect ratios

    Parameters
    ----------
    n_targets : int
        Number of targets/plots per page

    Returns
    -------
    tuple
        (rows, cols, figsize) for matplotlib subplot layout
    """
    if n_targets <= 3:
        # Single column layout for 3 or fewer
        return (n_targets, 1, (12, 16))
    else:
        # Two column layout for 4+
        rows = (n_targets + 1) // 2  # Ceiling division
        cols = 2
        height = 4 * rows + 4  # Scale height with number of rows
        return (rows, cols, (16, height))


def get_ssp_data(all_data: Dict[str, Any], ssp_name: str, data_type: str) -> np.ndarray:
    """
    Extract specific data array from loaded NetCDF data structure.

    Parameters
    ----------
    all_data : Dict[str, Any]
        Result from load_all_data()
    ssp_name : str
        SSP scenario name (e.g., 'ssp245')
    data_type : str
        Data type ('tas', 'pr', 'gdp', 'pop')

    Returns
    -------
    np.ndarray
        Data array with shape [lat, lon, time]
    """
    if ssp_name not in all_data:
        raise KeyError(f"SSP scenario '{ssp_name}' not found in loaded data. Available: {all_data['_metadata']['ssp_list']}")

    if data_type not in all_data[ssp_name]:
        raise KeyError(f"Data type '{data_type}' not found for {ssp_name}. Available: {list(all_data[ssp_name].keys())}")

    return all_data[ssp_name][data_type]


def create_forward_model_visualization(forward_results, config, output_dir, all_data):
    """
    Create comprehensive PDF visualization for Step 4 forward model results.

    Generates a multi-page PDF with one page per (target, response_function, SSP) combination.
    Each page shows global mean time series with three lines:
    - y_climate: GDP with full climate change effects
    - y_weather: GDP with weather variability only
    - baseline: Original SSP GDP projections

    Parameters
    ----------
    forward_results : dict
        Results from Step 4 forward integration containing SSP-specific data
    config : dict
        Configuration dictionary with scenarios and response functions
    output_dir : str
        Directory for output files
    model_name : str
        Climate model name for labeling
    all_data : dict
        All loaded NetCDF data for baseline GDP access

    Returns
    -------
    str
        Path to generated PDF file
    """

    # Extract configuration values for standardized naming
    json_id = config['run_metadata']['json_id']
    model_name = config['climate_model']['model_name']

    # Generate output filename using standardized pattern
    pdf_filename = f"step4_{json_id}_{model_name}_forward_model_lineplots.pdf"
    pdf_path = os.path.join(output_dir, pdf_filename)

    # Extract metadata
    valid_mask = forward_results['valid_mask']
    lat = forward_results['_coordinates']['lat']
    response_function_names = forward_results['response_function_names']
    target_names = forward_results['target_names']
    forward_ssps = config['ssp_scenarios']['forward_simulation_ssps']

    # Calculate adaptive layout based on number of targets
    n_targets = len(target_names)
    subplot_rows, subplot_cols, fig_size = get_adaptive_subplot_layout(n_targets)

    # Calculate total pages (one page per response function × SSP combination)
    total_pages = len(response_function_names) * len(forward_ssps)

    print(f"Creating Step 4 line charts: {n_targets} targets per page across {total_pages} pages")
    print(f"  {len(response_function_names)} response functions × {len(forward_ssps)} SSPs")
    print(f"  Layout: {subplot_rows} rows × {subplot_cols} cols per page")

    # Create PDF with multi-page layout
    with PdfPages(pdf_path) as pdf:
        page_num = 0

        # Loop through response functions and SSPs (each gets its own page)
        for response_idx, response_name in enumerate(response_function_names):
            response_config = config['response_function_scalings'][response_idx]

            for ssp in forward_ssps:

                # Create new page for this response function × SSP combination
                page_num += 1
                fig = plt.figure(figsize=fig_size)
                fig.suptitle(f'Step 4: Forward Model Time Series - {model_name}\n'
                            f'SSP: {ssp.upper()} | Response Function: {response_config["scaling_name"]} - Page {page_num}/{total_pages}',
                            fontsize=16, fontweight='bold', y=0.96)

                # Plot all targets on this page
                for target_idx, target_name in enumerate(target_names):
                    target_config = config['gdp_targets'][target_idx]

                    # Calculate subplot position (1-indexed)
                    if subplot_cols == 1:
                        # Single column layout
                        subplot_idx = target_idx + 1
                        ax = plt.subplot(subplot_rows, subplot_cols, subplot_idx)
                    else:
                        # Two column layout
                        row = target_idx // subplot_cols
                        col = target_idx % subplot_cols
                        subplot_idx = row * subplot_cols + col + 1
                        ax = plt.subplot(subplot_rows, subplot_cols, subplot_idx)

                    # Get SSP-specific data
                    ssp_results = forward_results['forward_results'][ssp]
                    gdp_climate = ssp_results['gdp_climate']  # [lat, lon, response_func, target, time]
                    gdp_weather = ssp_results['gdp_weather']  # [lat, lon, response_func, target, time]

                    # Get baseline GDP data from all_data
                    baseline_gdp = get_ssp_data(all_data, ssp, 'gdp')  # [time, lat, lon]

                    # Get years array from pre-computed metadata
                    years = all_data['years']


                    # Extract time series for this combination
                    ntime = gdp_climate.shape[4]
                    y_climate_series = np.zeros(ntime)
                    y_weather_series = np.zeros(ntime)
                    baseline_series = np.zeros(ntime)

                    for t in range(ntime):
                        # Extract spatial slice for this time
                        gdp_climate_t = gdp_climate[:, :, response_idx, target_idx, t]
                        gdp_weather_t = gdp_weather[:, :, response_idx, target_idx, t]
                        baseline_t = baseline_gdp[t, :, :]  # Note: baseline is [time, lat, lon]

                        # Calculate global means
                        y_climate_series[t] = calculate_global_mean(gdp_climate_t, lat, valid_mask)
                        y_weather_series[t] = calculate_global_mean(gdp_weather_t, lat, valid_mask)
                        baseline_series[t] = calculate_global_mean(baseline_t, lat, valid_mask)

                    # Plot the time series
                    ax.plot(years, y_climate_series, 'r-', linewidth=2, label='GDP with Climate Effects')
                    ax.plot(years, y_weather_series, 'b--', linewidth=2, label='GDP with Weather Only')
                    ax.plot(years, baseline_series, 'k:', linewidth=2, label=f'Baseline {ssp.upper()} GDP')

                    # Formatting
                    ax.set_xlabel('Year', fontsize=12)
                    ax.set_ylabel('Global Mean GDP', fontsize=12)
                    ax.set_title(f'{target_config["target_name"]} × {response_config["scaling_name"]} × {ssp.upper()}\n'
                                f'({target_config.get("description", "")[:60]}...)',
                                fontsize=14, fontweight='bold')
                    ax.legend(fontsize=10)
                    ax.grid(True, alpha=0.3)

                    # Set reasonable y-axis limits using zero-biased range
                    all_values = np.concatenate([y_climate_series, y_weather_series, baseline_series])
                    valid_values = all_values[np.isfinite(all_values)]
                    if len(valid_values) > 0:
                        # Use 1-99 percentiles to exclude extreme outliers
                        percentile_values = valid_values[(valid_values >= np.percentile(valid_values, 1)) &
                                                        (valid_values <= np.percentile(valid_values, 99))]
                        y_min, y_max = calculate_zero_biased_axis_range(percentile_values, padding_factor=0.15)
                        ax.set_ylim(y_min, y_max)

                    # Add target reduction info as text
                    if 'target_amount' in target_config:
                        target_text = f"Target Reduction: {target_config['target_amount']*100:.1f}%"
                        ax.text(0.02, 0.98, target_text, transform=ax.transAxes,
                               fontsize=10, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

                # Save this page after plotting all targets
                plt.tight_layout()
                plt.subplots_adjust(top=0.93, bottom=0.05)
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

        print(f"Generated {total_pages} pages in Step 4 visualization")

    print(f"Forward model visualization saved to {pdf_path}")
    return pdf_path


def create_forward_model_ratio_visualization(forward_results, config, output_dir, all_data):
    """
    Create PDF visualization for Step 4 forward model results showing ratios relative to baseline.
    Generates a multi-page PDF with one page per (target, response_function, SSP) combination.
    Each page shows global mean time series with two lines:
    - (GDP weather / baseline GDP) - 1: Weather effects only (dashed blue)
    - (GDP climate / baseline GDP) - 1: Full climate effects (solid red)

    Parameters
    ----------
    forward_results : dict
        Results from Step 4 forward integration containing SSP-specific data
    config : dict
        Configuration dictionary containing scenarios, response functions, climate_model.model_name, and run_metadata.json_id
    output_dir : str
        Directory for output files
    all_data : dict
        All loaded NetCDF data for baseline GDP access

    Returns
    -------
    str
        Path to generated PDF file
    """

    # Extract configuration values for standardized naming
    json_id = config['run_metadata']['json_id']
    model_name = config['climate_model']['model_name']

    # Generate output filename using standardized pattern
    pdf_filename = f"step4_{json_id}_{model_name}_forward_model_ratios.pdf"
    pdf_path = os.path.join(output_dir, pdf_filename)

    # Extract metadata
    valid_mask = forward_results['valid_mask']
    lat = forward_results['_coordinates']['lat']
    lon = forward_results['_coordinates']['lon']
    years = forward_results['_coordinates']['years']
    response_function_names = forward_results['response_function_names']
    target_names = forward_results['target_names']
    forward_ssps = config['ssp_scenarios']['forward_simulation_ssps']

    # Calculate total pages and subplot layout
    n_targets = len(target_names)
    total_pages = len(forward_ssps) * len(response_function_names)
    print(f"Creating Step 4 ratio visualization: {total_pages} pages ({n_targets} targets per page)")

    # Create PDF with multi-page layout
    with PdfPages(pdf_path) as pdf:
        page_num = 0

        # Loop through combinations (target innermost for 3-per-page grouping)
        for ssp in forward_ssps:
            for response_idx, response_name in enumerate(response_function_names):
                page_num += 1

                # Create new page with 3 subplots (one per target)
                fig = plt.figure(figsize=(12, 16))  # Taller figure for vertical arrangement
                fig.suptitle(f'Step 4: GDP Ratios to Baseline - {model_name}\n'
                           f'SSP: {ssp.upper()} | Response Function: {response_name}',
                           fontsize=16, fontweight='bold', y=0.98)

                # Get SSP-specific data
                ssp_results = forward_results['forward_results'][ssp]
                gdp_climate = ssp_results['gdp_climate']  # [lat, lon, response_func, target, time]
                gdp_weather = ssp_results['gdp_weather']  # [lat, lon, response_func, target, time]

                # Get baseline GDP for this SSP
                baseline_gdp = get_ssp_data(all_data, ssp, 'gdp')  # [time, lat, lon]

                # Calculate global means for baseline (area-weighted using valid cells only)
                area_weights = calculate_area_weights(lat)
                baseline_global = []
                for t_idx in range(len(years)):
                    baseline_slice = baseline_gdp[t_idx, :, :]  # [lat, lon]
                    baseline_global.append(calculate_global_mean(baseline_slice, lat, valid_mask))
                baseline_global = np.array(baseline_global)

                # Plot each target on this page
                for target_idx, target_name in enumerate(target_names):
                    ax = plt.subplot(n_targets, 1, target_idx + 1)  # Dynamic rows, 1 column

                    # Extract data for this combination [lat, lon, time]
                    gdp_climate_combo = gdp_climate[:, :, response_idx, target_idx, :]
                    gdp_weather_combo = gdp_weather[:, :, response_idx, target_idx, :]

                    # Calculate global means for this combination
                    climate_global = []
                    weather_global = []
                    for t_idx in range(len(years)):
                        climate_slice = gdp_climate_combo[:, :, t_idx]  # [lat, lon]
                        weather_slice = gdp_weather_combo[:, :, t_idx]  # [lat, lon]

                        climate_global.append(calculate_global_mean(climate_slice, lat, valid_mask))
                        weather_global.append(calculate_global_mean(weather_slice, lat, valid_mask))

                    climate_global = np.array(climate_global)
                    weather_global = np.array(weather_global)

                    # Calculate ratios minus 1 (fractional change from baseline)
                    weather_ratio = weather_global / baseline_global - 1.0
                    climate_ratio = climate_global / baseline_global - 1.0

                    # Plot the ratio lines
                    ax.plot(years, weather_ratio, 'b--', linewidth=2, label='Weather Effects Only', alpha=0.8)
                    ax.plot(years, climate_ratio, 'r-', linewidth=2, label='Full Climate Effects', alpha=0.8)

                    # Add horizontal line at zero for reference
                    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)

                    # Formatting
                    ax.set_xlabel('Year', fontsize=12)
                    ax.set_ylabel('Fractional Change from Baseline', fontsize=12)
                    ax.set_title(f'Target: {target_name}', fontsize=14, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    ax.legend(fontsize=10, loc='best')

                    # Set reasonable y-axis limits using zero-biased range with 20% padding
                    all_values = np.concatenate([weather_ratio, climate_ratio])
                    valid_values = all_values[np.isfinite(all_values)]
                    if len(valid_values) > 0:
                        vmin, vmax = calculate_zero_biased_axis_range(valid_values, padding_factor=0.20)
                        ax.set_ylim(vmin, vmax)

                    # Add info box with final values
                    final_weather = weather_ratio[-1]
                    final_climate = climate_ratio[-1]
                    info_text = f'2100 Values:\nWeather: {final_weather:+.3f}\nClimate: {final_climate:+.3f}'
                    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                # Save this page
                plt.tight_layout()
                plt.subplots_adjust(top=0.93, bottom=0.05)
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

    print(f"Forward model ratio visualization saved to {pdf_path} ({total_pages} pages)")
    return pdf_path


def create_forward_model_maps_visualization(forward_results, config, output_dir, all_data):
    """
    Create spatial maps visualization for Step 4 forward model results.

    Generates two multi-page PDFs:
    1. Linear scale: (y_climate/y_weather) - 1 (original maps)
    2. Log10 scale: log10(y_climate/y_weather) showing extreme values and off-scale points

    Each PDF has one map per (target, response_function, SSP) combination,
    with data averaged over the configured target period.

    Parameters
    ----------
    forward_results : dict
        Results from Step 4 forward integration containing SSP-specific data
    config : dict
        Configuration dictionary with scenarios and response functions
    output_dir : str
        Directory for output files
    model_name : str
        Climate model name for labeling
    all_data : dict
        All loaded NetCDF data (not used but kept for consistency)

    Returns
    -------
    tuple
        (linear_pdf_path, log10_pdf_path) - Paths to both generated PDF files
    """

    # Extract configuration values for standardized naming
    json_id = config['run_metadata']['json_id']
    model_name = config['climate_model']['model_name']

    # Generate output filenames using standardized pattern
    linear_pdf_filename = f"step4_{json_id}_{model_name}_forward_model_maps.pdf"
    log10_pdf_filename = f"step4_{json_id}_{model_name}_forward_model_maps_log10.pdf"
    linear_pdf_path = os.path.join(output_dir, linear_pdf_filename)
    log10_pdf_path = os.path.join(output_dir, log10_pdf_filename)

    # Extract metadata
    valid_mask = forward_results['valid_mask']
    lat = forward_results['_coordinates']['lat']
    lon = forward_results['_coordinates']['lon']
    response_function_names = forward_results['response_function_names']
    target_names = forward_results['target_names']
    forward_ssps = config['ssp_scenarios']['forward_simulation_ssps']

    # Create coordinate grids for plotting
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    # Get target period from config
    target_start = config['time_periods']['target_period']['start_year']
    target_end = config['time_periods']['target_period']['end_year']

    # Calculate adaptive layout based on number of targets
    n_targets = len(target_names)
    subplot_rows, subplot_cols, fig_size = get_adaptive_subplot_layout(n_targets)

    # Calculate total pages (one page per response function × SSP combination)
    total_pages = len(response_function_names) * len(forward_ssps)

    print(f"Creating Step 4 maps: {n_targets} targets per page across {total_pages} pages")
    print(f"  {len(response_function_names)} response functions × {len(forward_ssps)} SSPs")
    print(f"  Layout: {subplot_rows} rows × {subplot_cols} cols per page")
    print(f"  Generating both linear and log10 scale PDFs in parallel")

    # Create both PDFs with multi-page layout
    with PdfPages(linear_pdf_path) as linear_pdf, PdfPages(log10_pdf_path) as log10_pdf:
        page_num = 0

        # Loop through SSPs and response functions (each gets its own page)
        for ssp in forward_ssps:

            for response_idx, response_name in enumerate(response_function_names):
                response_config = config['response_function_scalings'][response_idx]

                # Create new pages for both PDFs for this SSP × damage combination
                page_num += 1
                linear_fig = plt.figure(figsize=fig_size)
                linear_fig.suptitle(f'Step 4: Forward Model Results (Linear Scale) - {model_name}\n'
                                   f'SSP: {ssp.upper()} | Response Function: {response_config["scaling_name"]} - Page {page_num}/{total_pages}',
                                   fontsize=16, fontweight='bold', y=0.96)

                log10_fig = plt.figure(figsize=fig_size)
                log10_fig.suptitle(f'Step 4: Forward Model Results (Log10 Scale) - {model_name}\n'
                                  f'SSP: {ssp.upper()} | Response Function: {response_config["scaling_name"]} - Page {page_num}/{total_pages}',
                                  fontsize=16, fontweight='bold', y=0.96)

                # Plot all targets on this page
                for target_idx, target_name in enumerate(target_names):
                    target_config = config['gdp_targets'][target_idx]

                    # Calculate subplot position (1-indexed)
                    if subplot_cols == 1:
                        # Single column layout
                        subplot_idx = target_idx + 1
                    else:
                        # Two column layout
                        row = target_idx // subplot_cols
                        col = target_idx % subplot_cols
                        subplot_idx = row * subplot_cols + col + 1

                    # Get SSP-specific data
                    ssp_results = forward_results['forward_results'][ssp]
                    gdp_climate = ssp_results['gdp_climate']  # [lat, lon, response_func, target, time]
                    gdp_weather = ssp_results['gdp_weather']  # [lat, lon, response_func, target, time]

                    # Extract data for this combination: [lat, lon, time]
                    gdp_climate_combo = gdp_climate[:, :, response_idx, target_idx, :]
                    gdp_weather_combo = gdp_weather[:, :, response_idx, target_idx, :]

                    # Calculate time indices for target period using actual time coordinates
                    time_coords = forward_results['_coordinates']['years']
                    target_start_idx = np.where(time_coords == target_start)[0][0]
                    target_end_idx = np.where(time_coords == target_end)[0][0] + 1

                    # Calculate mean ratios over target period for each grid cell
                    nlat, nlon = valid_mask.shape
                    impact_ratio_linear = np.full((nlat, nlon), np.nan)  # (climate/weather) - 1
                    impact_ratio_log10 = np.full((nlat, nlon), np.nan)   # log10(climate/weather)

                    for lat_idx in range(nlat):
                        for lon_idx in range(nlon):
                            if valid_mask[lat_idx, lon_idx]:
                                # Extract target period time series for this grid cell
                                climate_target = gdp_climate_combo[lat_idx, lon_idx, target_start_idx:target_end_idx]
                                weather_target = gdp_weather_combo[lat_idx, lon_idx, target_start_idx:target_end_idx]

                                if len(climate_target) > 0 and len(weather_target) > 0:
                                    # Add epsilon to prevent division by zero
                                    epsilon = 1e-20
                                    ratios = climate_target / (weather_target + epsilon)
                                    mean_ratio = np.nanmean(ratios)

                                    # Linear scale: (climate/weather) - 1
                                    impact_ratio_linear[lat_idx, lon_idx] = mean_ratio - 1.0

                                    # Log10 scale: log10(climate/weather)
                                    if mean_ratio > 0:
                                        impact_ratio_log10[lat_idx, lon_idx] = np.log10(mean_ratio)
                                    # Zeros or negative values remain NaN (will be white)

                    # Create linear scale map
                    plt.figure(linear_fig.number)
                    linear_ax = plt.subplot(subplot_rows, subplot_cols, subplot_idx)

                    # Determine color scale for linear using zero-biased range
                    valid_linear = impact_ratio_linear[valid_mask & np.isfinite(impact_ratio_linear)]
                    if len(valid_linear) > 0:
                        lin_vmin, lin_vmax = calculate_zero_biased_range(valid_linear)
                        lin_actual_min = np.min(valid_linear)
                        lin_actual_max = np.max(valid_linear)
                    else:
                        lin_vmin, lin_vmax = -0.01, 0.01
                        lin_actual_min = lin_actual_max = 0.0

                    # Linear map: blue-red colormap (blue=positive, red=negative, white=zero)
                    lin_cmap = plt.cm.RdBu_r
                    lin_norm = mcolors.TwoSlopeNorm(vmin=lin_vmin, vcenter=0.0, vmax=lin_vmax)

                    masked_linear = np.where(valid_mask, impact_ratio_linear, np.nan)
                    lin_im = linear_ax.pcolormesh(lon_grid, lat_grid, masked_linear, cmap=lin_cmap, norm=lin_norm, shading='auto')

                    # Add coastlines
                    linear_ax.contour(lon_grid, lat_grid, valid_mask, levels=[0.5], colors='black', linewidths=0.5, alpha=0.7)

                    # Linear map formatting
                    linear_ax.set_xlabel('Longitude', fontsize=12)
                    linear_ax.set_ylabel('Latitude', fontsize=12)
                    linear_ax.set_title(f'{ssp.upper()} × {target_name} × {response_name}\n'
                                f'Climate Impact: (GDP_climate/GDP_weather - 1)\nTarget Period Mean: {target_start}-{target_end}',
                                fontsize=14, fontweight='bold')

                    # Linear max/min box
                    lin_max_min_text = f'Max: {lin_actual_max:.4f}\nMin: {lin_actual_min:.4f}'
                    linear_ax.text(0.02, 0.08, lin_max_min_text, transform=linear_ax.transAxes,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'),
                           fontsize=10, verticalalignment='bottom')

                    # Linear colorbar
                    lin_cbar = plt.colorbar(lin_im, ax=linear_ax, shrink=0.6, aspect=12)
                    lin_cbar.set_label('Climate Impact Ratio - 1', rotation=270, labelpad=15, fontsize=12)
                    lin_cbar.ax.tick_params(labelsize=10)
                    if hasattr(lin_cbar, 'ax'):
                        lin_cbar.ax.axhline(y=0.0, color='black', linestyle='-', linewidth=1, alpha=0.8)

                    # Set linear map aspect and limits
                    linear_ax.set_xlim(lon.min(), lon.max())
                    linear_ax.set_ylim(lat.min(), lat.max())
                    linear_ax.set_aspect('equal')

                    # Create log10 scale map
                    plt.figure(log10_fig.number)
                    log10_ax = plt.subplot(subplot_rows, subplot_cols, subplot_idx)

                    # Determine color scale for log10 - use FULL data range to show outliers
                    valid_log10 = impact_ratio_log10[valid_mask & np.isfinite(impact_ratio_log10)]
                    if len(valid_log10) > 0:
                        log_actual_min = np.min(valid_log10)
                        log_actual_max = np.max(valid_log10)
                        # Use full data range (not symmetric) to highlight outliers
                        log_vmin, log_vmax = log_actual_min, log_actual_max

                        # Report extreme ratios for diagnostic purposes
                        if log_actual_max > 5:  # log10(ratio) > 5 means ratio > 100,000
                            # Find indices of maximum ratio
                            max_indices = np.where((valid_mask & np.isfinite(impact_ratio_log10)) &
                                                 (impact_ratio_log10 == log_actual_max))
                            max_lat_idx, max_lon_idx = max_indices[0][0], max_indices[1][0]
                            print(f"    WARNING: Extreme high ratios detected for {ssp.upper()} × {target_name} × {response_name}")
                            print(f"             log10(max_ratio) = {log_actual_max:.2f} (ratio = {10**log_actual_max:.2e})")
                            print(f"             at grid cell indices: lat_idx={max_lat_idx}, lon_idx={max_lon_idx}")
                        if log_actual_min < -5:  # log10(ratio) < -5 means ratio < 0.00001
                            # Find indices of minimum ratio
                            min_indices = np.where((valid_mask & np.isfinite(impact_ratio_log10)) &
                                                 (impact_ratio_log10 == log_actual_min))
                            min_lat_idx, min_lon_idx = min_indices[0][0], min_indices[1][0]
                            print(f"    WARNING: Extreme low ratios detected for {ssp.upper()} × {target_name} × {response_name}")
                            print(f"             log10(min_ratio) = {log_actual_min:.2f} (ratio = {10**log_actual_min:.2e})")
                            print(f"             at grid cell indices: lat_idx={min_lat_idx}, lon_idx={min_lon_idx}")
                    else:
                        log_vmin, log_vmax = -0.1, 0.1
                        log_actual_min = log_actual_max = 0.0

                    # Log10 map: viridis colormap for non-zero-centered scales (standard for outlier detection)
                    log_cmap = plt.cm.viridis
                    log_norm = mcolors.Normalize(vmin=log_vmin, vmax=log_vmax)

                    masked_log10 = np.where(valid_mask, impact_ratio_log10, np.nan)
                    log_im = log10_ax.pcolormesh(lon_grid, lat_grid, masked_log10, cmap=log_cmap, norm=log_norm, shading='auto')

                    # Add coastlines
                    log10_ax.contour(lon_grid, lat_grid, valid_mask, levels=[0.5], colors='black', linewidths=0.5, alpha=0.7)

                    # Log10 map formatting
                    log10_ax.set_xlabel('Longitude', fontsize=12)
                    log10_ax.set_ylabel('Latitude', fontsize=12)
                    log10_ax.set_title(f'{ssp.upper()} × {target_name} × {response_name}\n'
                                f'Climate Impact: log10(GDP_climate/GDP_weather)\nTarget Period Mean: {target_start}-{target_end}',
                                fontsize=14, fontweight='bold')

                    # Log10 max/min box (show both log and original ratio values)
                    if len(valid_log10) > 0:
                        max_ratio = 10**log_actual_max
                        min_ratio = 10**log_actual_min
                        log_max_min_text = f'Max: {log_actual_max:.2f} (×{max_ratio:.2e})\nMin: {log_actual_min:.2f} (×{min_ratio:.2e})'
                    else:
                        log_max_min_text = 'No valid data'
                    log10_ax.text(0.02, 0.08, log_max_min_text, transform=log10_ax.transAxes,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'),
                           fontsize=10, verticalalignment='bottom')

                    # Log10 colorbar (no zero reference line for viridis)
                    log_cbar = plt.colorbar(log_im, ax=log10_ax, shrink=0.6, aspect=12)
                    log_cbar.set_label('log10(GDP_climate/GDP_weather)', rotation=270, labelpad=15, fontsize=12)
                    log_cbar.ax.tick_params(labelsize=10)

                    # Set log10 map aspect and limits
                    log10_ax.set_xlim(lon.min(), lon.max())
                    log10_ax.set_ylim(lat.min(), lat.max())
                    log10_ax.set_aspect('equal')

                # Save both pages after plotting all targets for this SSP × damage combination
                for fig, pdf in [(linear_fig, linear_pdf), (log10_fig, log10_pdf)]:
                    plt.figure(fig.number)
                    plt.tight_layout()
                    plt.subplots_adjust(top=0.93, bottom=0.05)
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)

    print(f"Forward model maps saved to:")
    print(f"  Linear scale: {linear_pdf_path}")
    print(f"  Log10 scale: {log10_pdf_path}")
    print(f"  ({total_pages} pages each, {n_targets} targets per page)")
    return (linear_pdf_path, log10_pdf_path)


def print_gdp_weighted_scaling_summary(scaling_results: Dict[str, Any], config: Dict[str, Any], all_data: Dict[str, Any], output_dir: str) -> None:
    """
    Generate GDP-weighted summary of scaling factors and write to CSV file.

    Parameters
    ----------
    scaling_results : Dict[str, Any]
        Results from Step 3 containing scaling factors
    config : Dict[str, Any]
        Configuration dictionary
    all_data : Dict[str, Any]
        Pre-loaded NetCDF data containing GDP information
    output_dir : str, optional
        Directory to write CSV file. If None, prints to terminal.
    """
    print("\n" + "="*80)
    print("STEP 3 SCALING FACTOR SUMMARY")
    print("="*80)

    # Extract data
    response_function_names = scaling_results['response_function_names']
    target_names = scaling_results['target_names']
    valid_mask = scaling_results['valid_mask']
    scaling_factors = scaling_results['scaling_factors']  # [lat, lon, response_func, target]
    optimization_errors = scaling_results['optimization_errors']  # [lat, lon, response_func, target]

    # Get reference SSP GDP data for weighting
    reference_ssp = config['ssp_scenarios']['reference_ssp']
    gdp_data = get_ssp_data(all_data, reference_ssp, 'gdp')  # [time, lat, lon]

    # Calculate target period GDP for weighting (use same period as target calculation)
    target_start = config['time_periods']['target_period']['start_year']
    target_end = config['time_periods']['target_period']['end_year']
    years = all_data['years']

    start_idx = target_start - years[0]
    end_idx = target_end - years[0] + 1

    # Average GDP over target period for weighting
    gdp_target_period = np.mean(gdp_data[start_idx:end_idx], axis=0)  # [lat, lon]

    print(f"GDP-weighted global statistics for scaling factors (using {reference_ssp} GDP, {target_start}-{target_end}):")
    print(f"Valid grid cells: {np.sum(valid_mask)} of {valid_mask.size}")

    # Prepare data for CSV
    csv_data = []

    # Get coordinates from metadata
    metadata = get_grid_metadata(all_data)
    lat_values = metadata['lat']

    # Calculate GDP-weighted global means for each combination
    for target_idx, target_name in enumerate(target_names):
        for resp_idx, resp_name in enumerate(response_function_names):
            # Extract scaling factor for this combination
            scale_data = scaling_factors[:, :, resp_idx, target_idx]  # [lat, lon]

            # Use calculate_global_mean with GDP*scaling data to get proper area+GDP weighted mean
            # Since GDP is in units per km², calculate_global_mean will properly handle the spatial weighting
            gdp_weighted_scaling_data = gdp_target_period * scale_data
            total_weighted_scaling = calculate_global_mean(gdp_weighted_scaling_data, lat_values, valid_mask)
            total_gdp = calculate_global_mean(gdp_target_period, lat_values, valid_mask)

            # GDP-weighted mean = area_weighted_mean(GDP * scaling) / area_weighted_mean(GDP)
            if total_gdp > 0:
                gdp_weighted_mean = total_weighted_scaling / total_gdp
            else:
                gdp_weighted_mean = np.nan

            # Calculate GDP-weighted median using user's algorithm
            # Create weights = cos(lat) * GDP and sort by scaling factor values
            area_weights = calculate_area_weights(lat_values)
            area_weights_2d = np.broadcast_to(area_weights[:, np.newaxis], scale_data.shape)

            # Flatten arrays and apply valid mask
            flat_scale = scale_data.flatten()
            flat_gdp = gdp_target_period.flatten()
            flat_area_weights = area_weights_2d.flatten()
            flat_valid = valid_mask.flatten()

            # Keep only valid entries
            valid_indices = flat_valid & ~np.isnan(flat_scale) & ~np.isnan(flat_gdp)
            valid_scale = flat_scale[valid_indices]
            valid_gdp = flat_gdp[valid_indices]
            valid_area_weights = flat_area_weights[valid_indices]

            if len(valid_scale) > 0:
                # Column 0: weights (cos(lat) * GDP), Column 1: scaling factors
                gdp_area_weights = valid_area_weights * valid_gdp
                combined = np.column_stack([gdp_area_weights, valid_scale])

                # Sort by scaling factor values (column 1)
                sorted_indices = np.argsort(combined[:, 1])
                sorted_combined = combined[sorted_indices]

                # Calculate cumulative sum of weights and find median
                cumsum_weights = np.cumsum(sorted_combined[:, 0])
                total_weight = cumsum_weights[-1]
                half_weight = total_weight / 2.0

                if half_weight <= cumsum_weights[0]:
                    gdp_weighted_median = sorted_combined[0, 1]
                elif half_weight >= cumsum_weights[-1]:
                    gdp_weighted_median = sorted_combined[-1, 1]
                else:
                    gdp_weighted_median = np.interp(half_weight, cumsum_weights, sorted_combined[:, 1])
            else:
                gdp_weighted_median = np.nan

            # Calculate scaling factor max/min statistics
            valid_scaling = scale_data[valid_mask & np.isfinite(scale_data)]
            if len(valid_scaling) > 0:
                scaling_max = np.max(valid_scaling)
                scaling_min = np.min(valid_scaling)
            else:
                scaling_max = scaling_min = np.nan

            # Calculate objective function statistics
            error_data = optimization_errors[:, :, resp_idx, target_idx]  # [lat, lon]
            valid_errors = error_data[valid_mask & np.isfinite(error_data)]

            if len(valid_errors) > 0:
                obj_func_max = np.max(valid_errors)
                obj_func_mean = np.mean(valid_errors)
                obj_func_std = np.std(valid_errors)
                obj_func_min = np.min(valid_errors)
            else:
                obj_func_max = obj_func_mean = obj_func_std = obj_func_min = np.nan

            # Collect data for CSV
            csv_data.append({
                'target_name': target_name,
                'response_function': resp_name,
                'gdp_weighted_mean': gdp_weighted_mean,
                'gdp_weighted_median': gdp_weighted_median,
                'scaling_max': scaling_max,
                'scaling_min': scaling_min,
                'obj_func_max': obj_func_max,
                'obj_func_mean': obj_func_mean,
                'obj_func_std': obj_func_std,
                'obj_func_min': obj_func_min
            })

    # Write to CSV file if output_dir provided
    if output_dir and csv_data:

        json_id = config['run_metadata']['json_id']
        model_name = config['climate_model']['model_name']
        reference_ssp = config['ssp_scenarios']['reference_ssp']
        csv_filename = f"step3_{json_id}_{model_name}_{reference_ssp}_scaling_factors_summary.csv"
        csv_path = os.path.join(output_dir, csv_filename)

        # Create DataFrame and write to CSV
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_path, index=False, float_format='%.6f')

        print(f"Scaling factors summary written to: {csv_path}")
        print()



def create_target_gdp_visualization(target_results: Dict[str, Any], config: Dict[str, Any],
                                   output_dir: str, reference_ssp: str, valid_mask: np.ndarray) -> str:
    """
    Create comprehensive visualization of target GDP reduction results.

    Generates a single-page PDF with:
    - Global maps showing spatial patterns of each target reduction type
    - Line plot showing response functions vs temperature (if coefficients available)

    Parameters
    ----------
    target_results : Dict[str, Any]
        Results from step1_calculate_target_gdp_changes()
    config : Dict[str, Any]
        Integrated configuration dictionary containing climate_model.model_name and run_metadata.json_id
    output_dir : str
        Output directory path
    reference_ssp : str
        Reference SSP scenario name

    Returns
    -------
    str
        Path to generated PDF file
    """

    # Extract configuration values for standardized naming
    json_id = config['run_metadata']['json_id']
    model_name = config['climate_model']['model_name']

    # Generate output filename using standardized pattern
    pdf_filename = f"step1_{json_id}_{model_name}_{reference_ssp}_target_gdp_visualization.pdf"
    pdf_path = os.path.join(output_dir, pdf_filename)

    # Extract metadata and coordinates
    metadata = target_results['_metadata']
    lat = metadata['lat']
    lon = metadata['lon']
    tas_ref = metadata['tas_ref']
    gdp_target = metadata['gdp_target']
    global_tas_ref = metadata['global_tas_ref']
    global_gdp_target = metadata['global_gdp_target']

    # Create meshgrid for plotting
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    # Determine color scale using zero-biased range from all target reduction data
    all_reduction_values = []
    for target_name, result in target_results.items():
        if target_name != '_metadata':
            reduction_data = result['reduction_array'][valid_mask]
            all_reduction_values.extend(reduction_data.flatten())

    if len(all_reduction_values) > 0:
        vmin, vmax = calculate_zero_biased_range(all_reduction_values)
    else:
        vmin, vmax = -0.25, 0.25  # Fallback

    # Use standard blue-red colormap (blue=positive, red=negative, white=zero)
    cmap = plt.cm.RdBu_r

    # Calculate GDP-weighted temperature for target period (2080-2100)
    target_period_start = config['time_periods']['target_period']['start_year']
    target_period_end = config['time_periods']['target_period']['end_year']
    gdp_weighted_tas_target = calculate_global_mean(gdp_target * tas_ref, lat, valid_mask) / global_gdp_target

    # Extract reduction arrays and calculate statistics
    reduction_arrays = {}
    global_means = {}
    data_ranges = {}

    # Get all available targets (flexible for different configurations)
    target_names = [key for key in target_results.keys() if key != '_metadata']

    for target_name in target_names:
        reduction_array = target_results[target_name]['reduction_array']
        global_mean = target_results[target_name]['global_mean_achieved']

        reduction_arrays[target_name] = reduction_array
        global_means[target_name] = global_mean

        # Calculate ranges using only valid cells
        valid_reduction_data = reduction_array[valid_mask]
        data_ranges[target_name] = {
            'min': float(np.min(valid_reduction_data)),
            'max': float(np.max(valid_reduction_data))
        }

    # Calculate overall data range for title annotation (using only valid cells)
    all_valid_data = np.concatenate([arr[valid_mask].flatten() for arr in reduction_arrays.values()])
    overall_min = np.min(all_valid_data)
    overall_max = np.max(all_valid_data)

    # Calculate GDP-weighted means for verification (like original code)
    gdp_weighted_means = {}
    for target_name in target_names:
        reduction_array = reduction_arrays[target_name]
        # GDP-weighted mean calculation: sum(gdp * reduction) / sum(gdp)
        gdp_weighted_mean = calculate_global_mean(gdp_target * (1 + reduction_array), lat, valid_mask) / calculate_global_mean(gdp_target, lat, valid_mask) - 1
        gdp_weighted_means[target_name] = gdp_weighted_mean

    with PdfPages(pdf_path) as pdf:
        # Single page with 4 panels: 3 maps + 1 line plot (2x2 layout)
        fig = plt.figure(figsize=(16, 12))

        # Overall title
        fig.suptitle(f'Target GDP Reductions - {model_name} {reference_ssp.upper()}\n'
                    f'GDP-weighted Mean Temperature ({target_period_start}-{target_period_end}): {gdp_weighted_tas_target:.2f}°C',
                    fontsize=16, fontweight='bold')

        # Calculate layout for maps + line plot
        n_targets = len(target_names)

        if n_targets <= 3:
            # Use 2x2 layout: 3 maps + 1 line plot
            subplot_rows, subplot_cols = 2, 2
            fig.set_size_inches(16, 12)
        else:
            # Use 3x2 layout: up to 5 maps + 1 line plot
            subplot_rows, subplot_cols = 3, 2
            fig.set_size_inches(18, 16)

        # Line plot will be in the last position
        line_plot_position = subplot_rows * subplot_cols

        for i, target_name in enumerate(target_names):  # Show all targets
            reduction_array = reduction_arrays[target_name]
            global_mean = global_means[target_name]
            gdp_weighted_mean = gdp_weighted_means[target_name]
            data_range = data_ranges[target_name]
            target_info = target_results[target_name]

            # Calculate subplot position for maps (1-indexed, avoiding last position)
            subplot_idx = i + 1
            ax = plt.subplot(subplot_rows, subplot_cols, subplot_idx)

            # Create map with zero-centered normalization (mask invalid cells)
            masked_reduction_array = np.where(valid_mask, reduction_array, np.nan)
            norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
            im = ax.pcolormesh(lon_grid, lat_grid, masked_reduction_array,
                             cmap=cmap, norm=norm, shading='auto')

            # Format target name for display
            display_name = target_name.replace('_', ' ').title()

            # Infer target type from configuration
            target_config = next(t for t in config['gdp_targets'] if t['target_name'] == target_name)
            if 'gdp_amount' in target_config:
                target_shape = 'constant'
            elif 'zero_amount_temperature' in target_config:
                target_shape = 'quadratic'
            elif 'global_mean_amount' in target_config:
                target_shape = 'linear'
            else:
                target_shape = 'unknown'

            ax.set_title(f'{display_name} ({target_shape})\n'
                        f'Range: {data_range["min"]:.4f} to {data_range["max"]:.4f}\n'
                        f'GDP-weighted: {gdp_weighted_mean:.6f}',
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('Longitude', fontsize=10)
            ax.set_ylabel('Latitude', fontsize=10)
            ax.grid(True, alpha=0.3)

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.7)
            cbar.set_label('Fractional GDP\nReduction', rotation=270, labelpad=20, fontsize=10)

        # Line plot in last position
        ax4 = plt.subplot(subplot_rows, subplot_cols, line_plot_position)

        # Temperature range for plotting
        tas_range = np.linspace(-10, 35, 1000)

        # Plot each function
        colors = ['black', 'red', 'blue', 'green', 'orange', 'purple']

        for i, target_name in enumerate(target_names):
            target_info = target_results[target_name]
            color = colors[i % len(colors)]

            # Infer target type from configuration
            target_config = next(t for t in config['gdp_targets'] if t['target_name'] == target_name)
            if 'gdp_amount' in target_config:
                target_shape = 'constant'
            elif 'zero_amount_temperature' in target_config:
                target_shape = 'quadratic'
            elif 'global_mean_amount' in target_config:
                target_shape = 'linear'
            else:
                target_shape = 'unknown'

            if target_shape == 'constant':
                # Constant function
                gdp_targets = config['gdp_targets']
                const_config = next(t for t in gdp_targets if t['target_name'] == target_name)
                constant_value = const_config['gdp_amount']
                function_values = np.full_like(tas_range, constant_value)
                label = f'Constant: {constant_value:.3f}'

                # Horizontal line for constant
                ax4.plot(tas_range, function_values, color=color, linewidth=2,
                        label=label, alpha=0.8)

            elif target_shape == 'linear':
                coefficients = target_info['coefficients']
                if coefficients:
                    # Linear function: reduction = a0 + a1 * T
                    a0, a1 = coefficients['a0'], coefficients['a1']
                    function_values = a0 + a1 * tas_range

                    ax4.plot(tas_range, function_values, color=color, linewidth=2,
                            label=f'Linear: {a0:.4f} + {a1:.4f}×T', alpha=0.8)

                    # Add calibration point from config
                    gdp_targets = config['gdp_targets']
                    linear_config = next(t for t in gdp_targets if t['target_name'] == target_name)
                    if 'reference_temperature' in linear_config:
                        ref_tas = linear_config['reference_temperature']
                        ref_value = linear_config['amount_at_reference_temp']
                        ax4.plot(ref_tas, ref_value, 'o', color=color, markersize=8,
                                label=f'Linear calib: {ref_tas}°C = {ref_value:.3f}')

            elif target_shape == 'quadratic':
                coefficients = target_info['coefficients']
                if coefficients:
                    # Quadratic function: reduction = a + b*T + c*T²
                    a, b, c = coefficients['a'], coefficients['b'], coefficients['c']
                    function_values = a + b * tas_range + c * tas_range**2

                    ax4.plot(tas_range, function_values, color=color, linewidth=2,
                            label=f'Quadratic: {a:.4f} + {b:.4f}×T + {c:.6f}×T²', alpha=0.8)

                    # Add calibration points from config
                    gdp_targets = config['gdp_targets']
                    quad_config = next(t for t in gdp_targets if t['target_name'] == target_name)

                    # Handle new derivative-based specification
                    zero_tas = quad_config['zero_amount_temperature']
                    derivative = quad_config['derivative_at_zero_amount_temperature']
                    ax4.plot(zero_tas, 0, 's', color=color, markersize=8,
                            label=f'Quad zero: {zero_tas}°C = 0 (slope={derivative:.3f})')

        # Add reference lines
        ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        # Format line plot
        ax4.set_xlabel('Temperature (°C)', fontsize=12)
        ax4.set_ylabel('Fractional GDP Reduction', fontsize=12)
        ax4.set_title('Target Functions vs Temperature', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=9, loc='best')

        # Set axis limits
        ax4.set_xlim(-10, 35)

        # Calculate y-axis limits from all function values
        all_y_values = []
        for target_name in target_names:
            target_info = target_results[target_name]

            # Infer target type from configuration
            target_config = next(t for t in config['gdp_targets'] if t['target_name'] == target_name)
            if 'gdp_amount' in target_config:
                target_shape = 'constant'
            elif 'zero_amount_temperature' in target_config:
                target_shape = 'quadratic'
            elif 'global_mean_amount' in target_config:
                target_shape = 'linear'
            else:
                target_shape = 'unknown'

            if target_shape == 'constant':
                gdp_targets = config['gdp_targets']
                const_config = next(t for t in gdp_targets if t['target_name'] == target_name)
                constant_value = const_config['gdp_amount']
                all_y_values.extend([constant_value])

            elif target_shape in ['linear', 'quadratic'] and target_info['coefficients']:
                coefficients = target_info['coefficients']
                if target_shape == 'linear':
                    a0, a1 = coefficients['a0'], coefficients['a1']
                    values = a0 + a1 * tas_range
                elif target_shape == 'quadratic':
                    a, b, c = coefficients['a'], coefficients['b'], coefficients['c']
                    values = a + b * tas_range + c * tas_range**2
                all_y_values.extend(values)

        if all_y_values:
            y_min, y_max = calculate_zero_biased_axis_range(all_y_values, padding_factor=0.1)
            ax4.set_ylim(y_min, y_max)

        # Adjust layout to prevent overlap
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)  # Make room for suptitle
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    print(f"Target GDP visualization saved to {pdf_path}")
    return pdf_path




def create_scaling_factors_visualization(scaling_results, config, output_dir):
    """
    Create comprehensive PDF visualization for Step 3 scaling factor results.

    Generates a multi-panel visualization with one map per response function × target combination.
    For typical case: 3 response functions × 2 targets = 6 small maps on one page.

    Parameters
    ----------
    scaling_results : dict
        Results from Step 3 scaling factor calculation
    config : dict
        Configuration dictionary
    output_dir : str
        Directory for output files
    model_name : str
        Climate model name for labeling

    Returns
    -------
    str
        Path to generated PDF file
    """

    # Extract configuration values for standardized naming
    json_id = config['run_metadata']['json_id']
    model_name = config['climate_model']['model_name']
    reference_ssp = config['ssp_scenarios']['reference_ssp']

    # Generate output filename using standardized pattern
    pdf_filename = f"step3_{json_id}_{model_name}_{reference_ssp}_scaling_factors_visualization.pdf"
    pdf_path = os.path.join(output_dir, pdf_filename)

    # Extract data arrays and metadata
    scaling_factors = scaling_results['scaling_factors']  # [lat, lon, response_func, target]
    valid_mask = scaling_results['valid_mask']  # [lat, lon]
    response_function_names = scaling_results['response_function_names']
    target_names = scaling_results['target_names']

    # Get coordinate information
    coordinates = scaling_results['_coordinates']
    lat = coordinates['lat']
    lon = coordinates['lon']

    # Create meshgrid for plotting
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    # Get dimensions
    nlat, nlon, n_response_functions, n_targets = scaling_factors.shape

    # Calculate adaptive layout based on number of targets
    subplot_rows, subplot_cols, fig_size = get_adaptive_subplot_layout(n_targets)

    # Calculate total pages (one page per response function)
    total_pages = n_response_functions

    print(f"Creating Step 3 visualization: {n_targets} targets per page across {total_pages} pages")
    print(f"  {n_response_functions} response functions")
    print(f"  Layout: {subplot_rows} rows × {subplot_cols} cols per page")

    # Create PDF with multi-page layout
    with PdfPages(pdf_path) as pdf:
        page_num = 0

        # Loop through response functions (each gets its own page)
        for response_idx, response_name in enumerate(response_function_names):
            response_config = config['response_function_scalings'][response_idx]

            # Create new page for this response function
            page_num += 1
            fig = plt.figure(figsize=fig_size)
            fig.suptitle(f'Step 3: Scaling Factors - {model_name} ({reference_ssp})\n'
                        f'Response Function: {response_config["scaling_name"]} - Page {page_num}/{total_pages}',
                        fontsize=16, fontweight='bold', y=0.96)

            # Plot all targets on this page
            for target_idx, target_name in enumerate(target_names):

                # Calculate subplot position (1-indexed)
                if subplot_cols == 1:
                    # Single column layout
                    subplot_idx = target_idx + 1
                    ax = plt.subplot(subplot_rows, subplot_cols, subplot_idx)
                else:
                    # Two column layout
                    row = target_idx // subplot_cols
                    col = target_idx % subplot_cols
                    subplot_idx = row * subplot_cols + col + 1
                    ax = plt.subplot(subplot_rows, subplot_cols, subplot_idx)

                # Extract scaling factor map for this combination
                sf_map = scaling_factors[:, :, response_idx, target_idx]

                # Mask invalid cells and ocean
                sf_map_masked = np.copy(sf_map)
                sf_map_masked[~valid_mask] = np.nan

                # Calculate independent zero-biased range for this map
                valid_values = sf_map[valid_mask & np.isfinite(sf_map)]
                if len(valid_values) > 0:
                    vmin, vmax = calculate_zero_biased_range(valid_values)
                    actual_min = np.min(valid_values)
                    actual_max = np.max(valid_values)
                else:
                    vmin, vmax = -0.01, 0.01  # Default range
                    actual_min = actual_max = 0.0

                # Create map with proper zero-centered normalization
                norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
                im = ax.pcolormesh(lon_grid, lat_grid, sf_map_masked,
                                 cmap='RdBu_r', norm=norm, shading='auto')

                # Add coastlines (basic grid)
                ax.contour(lon_grid, lat_grid, valid_mask.astype(float),
                          levels=[0.5], colors='black', linewidths=0.5, alpha=0.3)

                # Labels and formatting (larger fonts for better visibility)
                ax.set_title(f'{response_name}\n{target_name}', fontsize=14, fontweight='bold')
                ax.set_xlabel('Longitude', fontsize=12)
                ax.set_ylabel('Latitude', fontsize=12)
                ax.tick_params(labelsize=10)

                # Set aspect ratio and limits
                ax.set_xlim(lon.min(), lon.max())
                ax.set_ylim(lat.min(), lat.max())
                ax.set_aspect('equal')

                # Add max/min value box in lower part of the map
                max_min_text = f'Max: {actual_max:.4f}\nMin: {actual_min:.4f}'
                ax.text(0.02, 0.08, max_min_text, transform=ax.transAxes,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'),
                       fontsize=10, verticalalignment='bottom')

                # Add colorbar for each subplot (larger for better visibility)
                cbar = plt.colorbar(im, ax=ax, shrink=0.6, aspect=12)
                cbar.set_label('Scaling Factor', rotation=270, labelpad=15, fontsize=12)
                cbar.ax.tick_params(labelsize=10)

            # Save this page after plotting all targets for this response function
            plt.tight_layout()
            plt.subplots_adjust(top=0.93, bottom=0.05)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    print(f"Scaling factors visualization saved to {pdf_path} ({total_pages} pages, {n_targets} targets per page)")
    return pdf_path




def create_objective_function_visualization(scaling_results, config, output_dir):
    """
    Create comprehensive PDF visualization for Step 3 objective function values.

    Generates one page per 3 maps (same layout as scaling factors) showing optimization
    objective function values across grid cells. Lower values indicate better constraint
    satisfaction for each response function × target combination.

    Parameters
    ----------
    scaling_results : dict
        Results from Step 3 scaling factor calculation
    config : dict
        Configuration dictionary
    output_dir : str
        Directory for output files
    model_name : str
        Climate model name for labeling

    Returns
    -------
    str
        Path to generated PDF file
    """

    # Extract configuration values for standardized naming
    json_id = config['run_metadata']['json_id']
    model_name = config['climate_model']['model_name']
    reference_ssp = config['ssp_scenarios']['reference_ssp']

    # Generate output filename using standardized pattern
    pdf_filename = f"step3_{json_id}_{model_name}_{reference_ssp}_objective_function_visualization.pdf"
    pdf_path = os.path.join(output_dir, pdf_filename)

    # Extract data arrays and metadata
    optimization_errors = scaling_results['optimization_errors']  # [lat, lon, response_func, target]
    valid_mask = scaling_results['valid_mask']  # [lat, lon]
    response_function_names = scaling_results['response_function_names']
    target_names = scaling_results['target_names']

    # Get coordinate information
    coordinates = scaling_results['_coordinates']
    lat = coordinates['lat']
    lon = coordinates['lon']

    # Create meshgrid for plotting
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    # Get dimensions
    nlat, nlon, n_response_functions, n_targets = optimization_errors.shape

    # Calculate adaptive layout based on number of targets
    subplot_rows, subplot_cols, fig_size = get_adaptive_subplot_layout(n_targets)

    # Calculate total pages (one page per response function)
    total_pages = n_response_functions

    print(f"Creating Step 3 objective function visualization: {n_targets} targets per page across {total_pages} pages")
    print(f"  {n_response_functions} response functions")
    print(f"  Layout: {subplot_rows} rows × {subplot_cols} cols per page")

    # Create PDF with multi-page layout
    with PdfPages(pdf_path) as pdf:
        page_num = 0

        # Loop through response functions (each gets its own page)
        for response_idx, response_name in enumerate(response_function_names):
            response_config = config['response_function_scalings'][response_idx]

            # Create new page for this response function
            page_num += 1
            fig = plt.figure(figsize=fig_size)
            fig.suptitle(f'Step 3: Objective Function Values - {model_name} ({reference_ssp})\n'
                        f'Response Function: {response_config["scaling_name"]} - Page {page_num}/{total_pages}',
                        fontsize=16, fontweight='bold', y=0.96)

            # Plot all targets on this page
            for target_idx, target_name in enumerate(target_names):

                # Calculate subplot position (1-indexed)
                if subplot_cols == 1:
                    # Single column layout
                    subplot_idx = target_idx + 1
                    ax = plt.subplot(subplot_rows, subplot_cols, subplot_idx)
                else:
                    # Two column layout
                    row = target_idx // subplot_cols
                    col = target_idx % subplot_cols
                    subplot_idx = row * subplot_cols + col + 1
                    ax = plt.subplot(subplot_rows, subplot_cols, subplot_idx)

                # Extract objective function map for this combination
                obj_map = optimization_errors[:, :, response_idx, target_idx]

                # Mask invalid cells and ocean
                obj_map_masked = np.copy(obj_map)
                obj_map_masked[~valid_mask] = np.nan

                # Calculate range for this map (objective function values are always >= 0)
                valid_values = obj_map[valid_mask & np.isfinite(obj_map)]
                if len(valid_values) > 0:
                    actual_min = np.min(valid_values)
                    actual_max = np.max(valid_values)
                else:
                    actual_min = actual_max = 0.0

                # Apply log10 transformation for visualization
                # Set minimum threshold to avoid log(0) issues
                min_threshold = 1e-14
                obj_map_log = np.copy(obj_map_masked)

                # Replace values below threshold with threshold, and zeros/negatives with threshold
                valid_finite_mask = valid_mask & np.isfinite(obj_map) & (obj_map > 0)
                obj_map_log[valid_finite_mask] = np.maximum(obj_map[valid_finite_mask], min_threshold)
                obj_map_log[~valid_finite_mask] = np.nan

                # Take log10
                obj_map_log[valid_finite_mask] = np.log10(obj_map_log[valid_finite_mask])

                # Set fixed log10 range: 1e-14 to 1
                vmin_log = np.log10(min_threshold)  # log10(1e-14) = -14
                vmax_log = np.log10(1.0)            # log10(1) = 0

                # Create map with log10 color scaling (viridis good for objective functions)
                cmap = plt.cm.viridis  # Dark = low error (good), bright = high error (poor)
                im = ax.pcolormesh(lon_grid, lat_grid, obj_map_log,
                                 cmap=cmap, vmin=vmin_log, vmax=vmax_log, shading='auto')

                # Add coastlines (basic grid)
                ax.contour(lon_grid, lat_grid, valid_mask.astype(float),
                          levels=[0.5], colors='white', linewidths=0.5, alpha=0.7)

                # Labels and formatting (larger fonts for better visibility)
                ax.set_title(f'{response_name}\n{target_name}', fontsize=14, fontweight='bold')
                ax.set_xlabel('Longitude', fontsize=12)
                ax.set_ylabel('Latitude', fontsize=12)
                ax.tick_params(labelsize=10)

                # Set aspect ratio and limits
                ax.set_xlim(lon.min(), lon.max())
                ax.set_ylim(lat.min(), lat.max())
                ax.set_aspect('equal')

                # Add max/min value box in lower part of the map
                max_min_text = f'Max: {actual_max:.6f}\nMin: {actual_min:.6f}'
                ax.text(0.02, 0.08, max_min_text, transform=ax.transAxes,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'),
                       fontsize=10, verticalalignment='bottom')

                # Add colorbar for each subplot (larger for better visibility)
                cbar = plt.colorbar(im, ax=ax, shrink=0.6, aspect=12)
                cbar.set_label('log₁₀(Objective Function Value)\n(Lower = Better Fit)', rotation=270, labelpad=15, fontsize=12)
                cbar.ax.tick_params(labelsize=10)

            # Save this page after plotting all targets for this response function
            plt.tight_layout()
            plt.subplots_adjust(top=0.93, bottom=0.05)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    print(f"Objective function visualization saved to {pdf_path} ({total_pages} pages, {n_targets} targets per page)")
    return pdf_path




def create_baseline_tfp_visualization(tfp_results, config, output_dir, all_data):
    """
    Create comprehensive PDF visualization for Step 2 baseline TFP results.

    Generates one page per forward simulation SSP, each with 3-panel visualization:
    1. Map of mean TFP for reference period
    2. Map of mean TFP for target period
    3. Time series percentile plot (min, 10%, 25%, 50%, 75%, 90%, max)

    Parameters
    ----------
    tfp_results : dict
        Results from Step 2 baseline TFP calculation containing:
        - '_metadata': Coordinate and data information
        - SSP scenarios with TFP time series data
    config : dict
        Configuration dictionary containing time periods, SSP information, climate_model.model_name, and run_metadata.json_id
    output_dir : str
        Directory for output files

    Returns
    -------
    str
        Path to generated PDF file
    """

    # Extract configuration values for standardized naming
    json_id = config['run_metadata']['json_id']
    model_name = config['climate_model']['model_name']
    forward_ssps = config['ssp_scenarios']['forward_simulation_ssps']

    # Generate output filename using standardized pattern
    pdf_filename = f"step2_{json_id}_{model_name}_baseline_tfp_visualization.pdf"
    pdf_path = os.path.join(output_dir, pdf_filename)

    # Extract metadata and coordinates
    metadata = tfp_results['_metadata']
    lat = metadata['lat']
    lon = metadata['lon']
    years = all_data['years']  # Years stored at top level in all_data

    # Get time period information
    ref_start = config['time_periods']['reference_period']['start_year']
    ref_end = config['time_periods']['reference_period']['end_year']
    target_start = config['time_periods']['target_period']['start_year']
    target_end = config['time_periods']['target_period']['end_year']

    # Create meshgrid for plotting
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    # Find time indices for reference and target periods
    ref_mask = (years >= ref_start) & (years <= ref_end)
    target_mask = (years >= target_start) & (years <= target_end)

    # Create PDF with multiple pages (one per forward SSP)
    with PdfPages(pdf_path) as pdf:
        for ssp_idx, viz_ssp in enumerate(forward_ssps):
            print(f"Creating TFP visualization page for {viz_ssp} ({ssp_idx+1}/{len(forward_ssps)})")

            # Extract TFP data for this SSP
            tfp_timeseries = tfp_results[viz_ssp]['tfp_baseline']  # Shape: [time, lat, lon]

            # Calculate period means (axis=0 for time dimension in [time, lat, lon])
            tfp_ref_mean = np.mean(tfp_timeseries[ref_mask], axis=0)  # [lat, lon]
            tfp_target_mean = np.mean(tfp_timeseries[target_mask], axis=0)  # [lat, lon]

            # Use pre-computed valid mask from TFP results (computed once during data loading)
            valid_mask = tfp_results[viz_ssp]['valid_mask']
            print(f"  Using pre-computed valid mask: {np.sum(valid_mask)} valid cells")

            # If no valid cells found, use fallback
            if np.sum(valid_mask) == 0:
                print(f"  WARNING: No valid cells found for {viz_ssp} - using sample cells for visualization")
                valid_mask = np.zeros_like(tfp_ref_mean, dtype=bool)
                # Set a few cells as valid for basic visualization
                valid_mask[32, 64] = True  # Single test cell
                valid_mask[16, 32] = True  # Another test cell

            # Calculate percentiles across valid grid cells for time series
            percentiles = [0, 10, 25, 50, 75, 90, 100]  # min, 10%, 25%, 50%, 75%, 90%, max
            percentile_labels = ['Min', '10%', '25%', 'Median', '75%', '90%', 'Max']
            percentile_colors = ['red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'purple']

            # Extract time series for valid cells only
            tfp_percentiles = np.zeros((len(percentiles), len(years)))

            for t_idx, year in enumerate(years):
                tfp_slice = tfp_timeseries[t_idx]  # [lat, lon]
                valid_values = tfp_slice[valid_mask]

                # Diagnostic output for first few time steps
                if t_idx < 3:
                    print(f"  DEBUG: t_idx={t_idx}, year={year}")
                    print(f"    tfp_slice shape: {tfp_slice.shape}, range: {np.nanmin(tfp_slice):.6e} to {np.nanmax(tfp_slice):.6e}")
                    print(f"    valid_mask sum: {np.sum(valid_mask)}")
                    print(f"    valid_values shape: {valid_values.shape}, range: {np.nanmin(valid_values):.6e} to {np.nanmax(valid_values):.6e}")
                    if len(valid_values) > 0:
                        calculated_percentiles = np.percentile(valid_values, percentiles)
                        print(f"    calculated percentiles: {calculated_percentiles}")
                        print(f"    percentile spread: {calculated_percentiles[-1] - calculated_percentiles[0]:.6e}")

                if len(valid_values) > 0:
                    tfp_percentiles[:, t_idx] = np.percentile(valid_values, percentiles)
                else:
                    tfp_percentiles[:, t_idx] = np.nan

            # Diagnostic output for percentile results
            print(f"  DEBUG: PERCENTILE CALCULATION COMPLETE for {viz_ssp}")
            print(f"    tfp_percentiles shape: {tfp_percentiles.shape}")
            print(f"    tfp_percentiles range: {np.nanmin(tfp_percentiles):.6e} to {np.nanmax(tfp_percentiles):.6e}")
            print(f"    tfp_percentiles NaN count: {np.sum(np.isnan(tfp_percentiles))}")
            print(f"    Sample percentiles for year 0: {tfp_percentiles[:, 0]}")
            print(f"    Sample percentiles for year 50: {tfp_percentiles[:, 50] if tfp_percentiles.shape[1] > 50 else 'N/A'}")

            # Check if all percentiles are identical (explaining overlapping lines)
            for p_idx, percentile_name in enumerate(percentile_labels):
                percentile_timeseries = tfp_percentiles[p_idx, :]
                percentile_range = np.nanmax(percentile_timeseries) - np.nanmin(percentile_timeseries)
                print(f"    {percentile_name} percentile range over time: {percentile_range:.6e}")

            # Determine color scale for maps (TFP values are always positive, use 0-to-max range)
            all_tfp_values = np.concatenate([tfp_ref_mean[valid_mask], tfp_target_mean[valid_mask]])
            vmin = 0.0  # TFP values should always be positive
            vmax = np.percentile(all_tfp_values, 95)  # Use 95th percentile to handle outliers

            # Create colormap for TFP (use viridis - good for scientific data)
            cmap = plt.cm.viridis

            # Create page layout for this SSP
            fig = plt.figure(figsize=(18, 10))

            # Overall title
            fig.suptitle(f'Baseline Total Factor Productivity - {model_name} {viz_ssp.upper()}\n'
                        f'Reference Period: {ref_start}-{ref_end} | Target Period: {target_start}-{target_end}',
                        fontsize=16, fontweight='bold')

            # Panel 1: Reference period mean TFP map
            ax1 = plt.subplot(2, 3, (1, 2))  # Top left, spans 2 columns
            im1 = ax1.pcolormesh(lon_grid, lat_grid, tfp_ref_mean,
                                cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
            ax1.set_title(f'Mean TFP: Reference Period ({ref_start}-{ref_end})',
                         fontsize=14, fontweight='bold')
            ax1.set_xlabel('Longitude')
            ax1.set_ylabel('Latitude')

            # Add coastlines if available
            ax1.set_xlim(lon.min(), lon.max())
            ax1.set_ylim(lat.min(), lat.max())

            # Colorbar for reference map
            cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8, aspect=20)
            cbar1.set_label('TFP', rotation=270, labelpad=15)

            # Panel 2: Target period mean TFP map
            ax2 = plt.subplot(2, 3, (4, 5))  # Bottom left, spans 2 columns
            im2 = ax2.pcolormesh(lon_grid, lat_grid, tfp_target_mean,
                                cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
            ax2.set_title(f'Mean TFP: Target Period ({target_start}-{target_end})',
                         fontsize=14, fontweight='bold')
            ax2.set_xlabel('Longitude')
            ax2.set_ylabel('Latitude')

            ax2.set_xlim(lon.min(), lon.max())
            ax2.set_ylim(lat.min(), lat.max())

            # Colorbar for target map
            cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8, aspect=20)
            cbar2.set_label('TFP', rotation=270, labelpad=15)

            # Panel 3: Time series percentiles
            ax3 = plt.subplot(1, 3, 3)  # Right side, full height

            for i, (percentile, label, color) in enumerate(zip(percentiles, percentile_labels, percentile_colors)):
                ax3.plot(years, tfp_percentiles[i], color=color, linewidth=2,
                        label=label, alpha=0.8)

            ax3.set_xlabel('Year', fontsize=12)
            ax3.set_ylabel('Total Factor Productivity', fontsize=12)
            ax3.set_title('TFP Percentiles Across Valid Grid Cells', fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            ax3.legend(fontsize=10, loc='best')

            # Add reference lines for time periods
            ax3.axvspan(ref_start, ref_end, alpha=0.2, color='blue', label='Reference Period')
            ax3.axvspan(target_start, target_end, alpha=0.2, color='red', label='Target Period')

            # Set reasonable axis limits
            ax3.set_xlim(years.min(), years.max())

            # Set y-axis limits using 90th percentile to avoid outlier distortion
            if np.all(np.isnan(tfp_percentiles)):
                print("  WARNING: All TFP percentile values are NaN - using default y-axis limits")
                ax3.set_ylim(0, 1)
            else:
                # Use 90th percentile (index 5) for max, 0 for min to avoid outlier distortion
                percentile_90_max = np.nanmax(tfp_percentiles[5, :])  # 90th percentile line maximum
                global_min = np.nanmin(tfp_percentiles)
                global_max = np.nanmax(tfp_percentiles)

                # Find coordinates of global min and max values in the full timeseries data
                global_min_full = np.nanmin(tfp_timeseries)
                global_max_full = np.nanmax(tfp_timeseries)

                # Find indices of min and max values (first occurrence if multiple)
                min_indices = np.unravel_index(np.nanargmin(tfp_timeseries), tfp_timeseries.shape)
                max_indices = np.unravel_index(np.nanargmax(tfp_timeseries), tfp_timeseries.shape)
                min_t, min_lat, min_lon = min_indices
                max_t, max_lat, max_lon = max_indices

                # Convert time index to year
                min_year = years[min_t]
                max_year = years[max_t]

                if np.isfinite(percentile_90_max) and percentile_90_max > 0:
                    # Set y-axis from 0 to 90th percentile max with small buffer
                    y_max = percentile_90_max * 1.1
                    ax3.set_ylim(0, y_max)

                    # Add text annotation showing global min/max ranges with coordinates
                    annotation_text = (f'Global range: {global_min_full:.3f} to {global_max_full:.1f}\n'
                                     f'Min: year {min_year}, lat[{min_lat}], lon[{min_lon}]\n'
                                     f'Max: year {max_year}, lat[{max_lat}], lon[{max_lon}]')
                    ax3.text(0.02, 0.98, annotation_text,
                            transform=ax3.transAxes, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                            fontsize=9)

                    # Create CSV file with time series for min and max grid points (per SSP)
                    csv_filename = f"step2_{json_id}_{model_name}_{viz_ssp}_baseline_tfp_extremes.csv"
                    csv_path = os.path.join(output_dir, csv_filename)

                    # Extract time series for min and max grid cells
                    min_tfp_series = tfp_timeseries[:, min_lat, min_lon]
                    max_tfp_series = tfp_timeseries[:, max_lat, max_lon]

                    # Get GDP and population data for these cells
                    # Access data directly from the current function call via load_gridded_data
                    min_pop_series = np.full(len(years), np.nan)
                    min_gdp_series = np.full(len(years), np.nan)
                    max_pop_series = np.full(len(years), np.nan)
                    max_gdp_series = np.full(len(years), np.nan)

                    # Extract data from pre-loaded all_data
                    # NOTE: all_data has shape [time, lat, lon] same as TFP
                    ssp_data = all_data[viz_ssp]
                    min_pop_series = ssp_data['pop'][:, min_lat, min_lon]
                    max_pop_series = ssp_data['pop'][:, max_lat, max_lon]
                    min_gdp_series = ssp_data['gdp'][:, min_lat, min_lon]
                    max_gdp_series = ssp_data['gdp'][:, max_lat, max_lon]

                    # Create DataFrame and save to CSV
                    extremes_data = {
                        'year': years,
                        'min_pop': min_pop_series,
                        'min_gdp': min_gdp_series,
                        'min_tfp': min_tfp_series,
                        'max_pop': max_pop_series,
                        'max_gdp': max_gdp_series,
                        'max_tfp': max_tfp_series
                    }
                    df = pd.DataFrame(extremes_data)
                    df.to_csv(csv_path, index=False)
                    print(f"  Extremes CSV saved: {csv_path}")
                else:
                    print("  WARNING: Invalid 90th percentile range - using default y-axis limits")
                    ax3.set_ylim(0, 1)

            # Adjust layout
            plt.tight_layout()
            plt.subplots_adjust(top=0.9)  # Make room for suptitle

            # Save this page to PDF
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    print(f"Baseline TFP visualization saved to {pdf_path} ({len(forward_ssps)} pages)")
    return pdf_path



