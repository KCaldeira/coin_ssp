import os
import numpy as np
import statsmodels.api as sm
import copy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
import xarray as xr
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List
from coin_ssp_core import ScalingParams

def create_serializable_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a JSON-serializable version of the configuration by filtering out non-serializable objects.

    Parameters
    ----------
    config : Dict[str, Any]
        Original configuration dictionary

    Returns
    -------
    Dict[str, Any]
        Filtered configuration dictionary safe for JSON serialization
    """
    import json
    import copy

    # Make a deep copy to avoid modifying the original
    filtered_config = copy.deepcopy(config)

    # Remove known non-serializable objects
    non_serializable_keys = ['model_params_factory']

    def remove_non_serializable(obj, path=""):
        if isinstance(obj, dict):
            keys_to_remove = []
            for key, value in obj.items():
                current_path = f"{path}.{key}" if path else key
                if key in non_serializable_keys:
                    keys_to_remove.append(key)
                else:
                    try:
                        # Test if the value is JSON serializable
                        json.dumps(value)
                    except (TypeError, ValueError):
                        # If not serializable, try to recurse or remove
                        if isinstance(value, dict):
                            remove_non_serializable(value, current_path)
                        elif isinstance(value, list):
                            remove_non_serializable(value, current_path)
                        else:
                            keys_to_remove.append(key)

            for key in keys_to_remove:
                obj.pop(key)

        elif isinstance(obj, list):
            items_to_remove = []
            for i, item in enumerate(obj):
                try:
                    json.dumps(item)
                except (TypeError, ValueError):
                    if isinstance(item, (dict, list)):
                        remove_non_serializable(item, f"{path}[{i}]")
                    else:
                        items_to_remove.append(i)

            for i in reversed(items_to_remove):
                obj.pop(i)

    remove_non_serializable(filtered_config)
    return filtered_config

def apply_time_series_filter(time_series, filter_width, ref_start_idx, ref_end_idx):
    """
    Apply LOESS filter to time series and remove trend relative to reference period mean.

    Parameters
    ----------
    time_series : array-like
        Annual time series values (first element corresponds to year 0)
    filter_width : int
        Width parameter (approx. number of years to smooth over)
    ref_start_idx : int
        Start index of reference period (0-indexed)
    ref_end_idx : int
        End index of reference period (0-indexed, inclusive)

    Returns
    -------
    numpy.ndarray
        Filtered time series with trend removed and reference period mean added back
    """
    ts = np.array(time_series, dtype=float)
    n = len(ts)
    t = np.arange(n, dtype=float)

    # Calculate reference period mean
    mean_of_reference_period = np.mean(ts[ref_start_idx:ref_end_idx+1])

    # LOWESS expects frac = proportion of data used in each local regression
    # Map filter_width (years) to fraction of total series
    frac = min(1.0, filter_width / n)

    # Run LOESS smoothing
    filtered_series = sm.nonparametric.lowess(ts, t, frac=frac,
                                              it=1, return_sorted=False)

    # Apply detrending to all years: remove trend and add reference period mean
    result = ts - filtered_series + mean_of_reference_period

    return result

def create_country_scaling_page(country, scaling_name, results, scaling_result, params, fig):
    """Create a single page with three panels for one country and scaling set."""
    fig.suptitle(f'{country} - {scaling_name}', fontsize=16, fontweight='bold')
    
    years = results['years']
    
    # Panel 1: GDP
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(years, results['gdp_observed'], 'k-', label='Baseline', linewidth=2)
    ax1.plot(years, scaling_result['gdp_climate'], 'r-', label='Climate', linewidth=1.5)
    ax1.plot(years, scaling_result['gdp_weather'], 'b--', label='Weather', linewidth=1.5)
    ax1.set_ylabel('GDP (billion $)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Add scaling info box in lower right corner with all climate parameters
    ps = scaling_result["params_scaled"]
    scaling_text = (f'Scaling: {scaling_name}\n'
                   f'Scale factor: {scaling_result["optimal_scale"]:.4f}\n'
                   f'Target: {params.amount_scale:.1%} by {params.year_scale}\n'
                   f'k_tas1: {ps.k_tas1:.6f}  k_tas2: {ps.k_tas2:.6f}\n'
                   f'tfp_tas1: {ps.tfp_tas1:.6f}  tfp_tas2: {ps.tfp_tas2:.6f}\n'
                   f'y_tas1: {ps.y_tas1:.6f}  y_tas2: {ps.y_tas2:.6f}\n'
                   f'k_pr1: {ps.k_pr1:.6f}  k_pr2: {ps.k_pr2:.6f}\n'
                   f'tfp_pr1: {ps.tfp_pr1:.6f}  tfp_pr2: {ps.tfp_pr2:.6f}\n'
                   f'y_pr1: {ps.y_pr1:.6f}  y_pr2: {ps.y_pr2:.6f}')
    
    ax1.text(0.98, 0.02, scaling_text, transform=ax1.transAxes, fontsize=8,
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Panel 2: TFP
    ax2 = fig.add_subplot(3, 1, 2)
    ax2.plot(years, results['tfp_baseline'], 'k-', label='Baseline', linewidth=2)
    ax2.plot(years, scaling_result['tfp_climate'], 'r-', label='Climate', linewidth=1.5)
    ax2.plot(years, scaling_result['tfp_weather'], 'b--', label='Weather', linewidth=1.5)
    ax2.set_ylabel('Total Factor Productivity')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Capital Stock
    ax3 = fig.add_subplot(3, 1, 3)
    ax3.plot(years, results['k_baseline'], 'k-', label='Baseline', linewidth=2)
    ax3.plot(years, scaling_result['k_climate'], 'r-', label='Climate', linewidth=1.5)
    ax3.plot(years, scaling_result['k_weather'], 'b--', label='Weather', linewidth=1.5)
    ax3.set_ylabel('Capital Stock (normalized)')
    ax3.set_xlabel('Year')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def filter_scaling_params(scaling_config):
    allowed_keys = set(ScalingParams.__dataclass_fields__.keys())
    return {k: v for k, v in scaling_config.items() if k in allowed_keys}

def create_forward_model_visualization(forward_results, config, output_dir, model_name, all_netcdf_data):
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
        Configuration dictionary with scenarios and damage functions
    output_dir : str
        Directory for output files
    model_name : str
        Climate model name for labeling
    all_netcdf_data : dict
        All loaded NetCDF data for baseline GDP access

    Returns
    -------
    str
        Path to generated PDF file
    """
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import os

    # Generate output filename
    pdf_filename = f"step4_forward_model_visualization_{model_name}.pdf"
    pdf_path = os.path.join(output_dir, pdf_filename)

    # Extract metadata
    valid_mask = forward_results['valid_mask']
    lat = forward_results['_coordinates']['lat']
    response_function_names = forward_results['response_function_names']
    target_names = forward_results['target_names']
    forward_ssps = config['ssp_scenarios']['forward_simulation_ssps']

    # Calculate total charts and pages
    total_charts = len(target_names) * len(response_function_names) * len(forward_ssps)
    charts_per_page = 3
    total_pages = (total_charts + charts_per_page - 1) // charts_per_page  # Ceiling division

    print(f"Creating Step 4 line charts: {total_charts} charts across {total_pages} pages (3 charts per page)")
    print(f"  {len(target_names)} targets × {len(response_function_names)} response functions × {len(forward_ssps)} SSPs")

    # Create PDF with multi-page layout
    with PdfPages(pdf_path) as pdf:
        chart_idx = 0
        page_num = 0

        # Loop through all combinations (target innermost for 3-per-page grouping)
        for damage_idx, damage_name in enumerate(response_function_names):
            damage_config = config['response_function_scalings'][damage_idx]

            for ssp in forward_ssps:

                for target_idx, target_name in enumerate(target_names):
                    target_config = config['gdp_targets'][target_idx]

                    # Start new page if needed (every 3 charts)
                    if chart_idx % charts_per_page == 0:
                        if chart_idx > 0:
                            # Save previous page
                            plt.tight_layout()
                            plt.subplots_adjust(top=0.93, bottom=0.05)
                            pdf.savefig(fig, bbox_inches='tight')
                            plt.close(fig)

                        # Create new page
                        page_num += 1
                        fig = plt.figure(figsize=(12, 16))  # Taller figure for vertical arrangement
                        fig.suptitle(f'Step 4: Forward Model Time Series - {model_name} - Page {page_num}/{total_pages}',
                                    fontsize=16, fontweight='bold', y=0.98)

                    # Position on current page (1-3)
                    subplot_idx = (chart_idx % charts_per_page) + 1
                    ax = plt.subplot(charts_per_page, 1, subplot_idx)  # 3 rows, 1 column

                    # Get SSP-specific data
                    ssp_results = forward_results['forward_results'][ssp]
                    gdp_climate = ssp_results['gdp_climate']  # [lat, lon, response_func, target, time]
                    gdp_weather = ssp_results['gdp_weather']  # [lat, lon, response_func, target, time]

                    # Get baseline GDP data from all_netcdf_data
                    baseline_gdp = get_ssp_data(all_netcdf_data, ssp, 'gdp')  # [time, lat, lon]
                    years = all_netcdf_data[ssp]['gdp_years']

                    # Extract time series for this combination
                    ntime = gdp_climate.shape[4]
                    y_climate_series = np.zeros(ntime)
                    y_weather_series = np.zeros(ntime)
                    baseline_series = np.zeros(ntime)

                    for t in range(ntime):
                        # Extract spatial slice for this time
                        gdp_climate_t = gdp_climate[:, :, damage_idx, target_idx, t]
                        gdp_weather_t = gdp_weather[:, :, damage_idx, target_idx, t]
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
                    ax.set_title(f'{target_config["target_name"]} × {damage_config["scaling_name"]} × {ssp.upper()}\n'
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

                    # Increment chart counter
                    chart_idx += 1

        # Save final page if there are any charts on it
        if chart_idx > 0:
            plt.tight_layout()
            plt.subplots_adjust(top=0.93, bottom=0.05)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        print(f"Generated {total_pages} pages in Step 4 visualization")

    print(f"Forward model visualization saved to {pdf_path}")
    return pdf_path


def create_forward_model_ratio_visualization(forward_results, config, output_dir, model_name, all_netcdf_data):
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
        Configuration dictionary with scenarios and damage functions
    output_dir : str
        Directory for output files
    model_name : str
        Climate model name for labeling
    all_netcdf_data : dict
        All loaded NetCDF data for baseline GDP access

    Returns
    -------
    str
        Path to generated PDF file
    """
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import os

    # Generate output filename
    pdf_filename = f"step4_forward_model_ratios_{model_name}.pdf"
    pdf_path = os.path.join(output_dir, pdf_filename)

    # Extract metadata
    valid_mask = forward_results['valid_mask']
    lat = forward_results['_coordinates']['lat']
    lon = forward_results['_coordinates']['lon']
    years = forward_results['_coordinates']['years']
    response_function_names = forward_results['response_function_names']
    target_names = forward_results['target_names']
    forward_ssps = config['ssp_scenarios']['forward_simulation_ssps']

    # Calculate total pages
    total_pages = len(forward_ssps) * len(response_function_names)
    print(f"Creating Step 4 ratio visualization: {total_pages} pages (3 targets per page)")

    # Create PDF with multi-page layout
    with PdfPages(pdf_path) as pdf:
        page_num = 0

        # Loop through combinations (target innermost for 3-per-page grouping)
        for ssp in forward_ssps:
            for damage_idx, damage_name in enumerate(response_function_names):
                page_num += 1

                # Create new page with 3 subplots (one per target)
                fig = plt.figure(figsize=(12, 16))  # Taller figure for vertical arrangement
                fig.suptitle(f'Step 4: GDP Ratios to Baseline - {model_name}\n'
                           f'SSP: {ssp.upper()} | Response Function: {damage_name}',
                           fontsize=16, fontweight='bold', y=0.98)

                # Get SSP-specific data
                ssp_results = forward_results['forward_results'][ssp]
                gdp_climate = ssp_results['gdp_climate']  # [lat, lon, response_func, target, time]
                gdp_weather = ssp_results['gdp_weather']  # [lat, lon, response_func, target, time]

                # Get baseline GDP for this SSP
                baseline_gdp = get_ssp_data(all_netcdf_data, ssp, 'gdp')  # [time, lat, lon]

                # Calculate global means for baseline (area-weighted using valid cells only)
                area_weights = calculate_area_weights(lat)
                baseline_global = []
                for t_idx in range(len(years)):
                    baseline_slice = baseline_gdp[t_idx, :, :]  # [lat, lon]
                    baseline_global.append(calculate_global_mean(baseline_slice, lat, valid_mask))
                baseline_global = np.array(baseline_global)

                # Plot each target on this page
                for target_idx, target_name in enumerate(target_names):
                    ax = plt.subplot(3, 1, target_idx + 1)  # 3 rows, 1 column

                    # Extract data for this combination [lat, lon, time]
                    gdp_climate_combo = gdp_climate[:, :, damage_idx, target_idx, :]
                    gdp_weather_combo = gdp_weather[:, :, damage_idx, target_idx, :]

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


def calculate_zero_biased_range(data_values, percentile_low=2.5, percentile_high=97.5):
    """
    Calculate visualization range that includes zero when appropriate.

    Logic:
    - If all positive and min > 0.5 * max: extend range to include zero
    - If all negative and max < 0.5 * min: extend range to include zero
    - Always make symmetric around zero for proper colormap centering

    Parameters
    ----------
    data_values : array-like
        Data values to calculate range for (should be finite values only)
    percentile_low : float
        Lower percentile for range calculation (default 2.5 for 95% coverage)
    percentile_high : float
        Upper percentile for range calculation (default 97.5 for 95% coverage)

    Returns
    -------
    tuple
        (vmin, vmax) range values, symmetric around zero
    """
    if len(data_values) == 0:
        return -0.01, 0.01

    # Calculate percentile range
    p_min = np.percentile(data_values, percentile_low)
    p_max = np.percentile(data_values, percentile_high)

    # Apply zero-bias logic
    if p_min > 0 and p_max > 0:  # All positive
        if p_min > 0.5 * p_max:  # Lower end is > 50% of upper end
            range_min = 0.0  # Extend to zero
        else:
            range_min = p_min
        range_max = p_max
    elif p_min < 0 and p_max < 0:  # All negative
        if p_max < 0.5 * p_min:  # Upper end is > 50% of |lower end|
            range_max = 0.0  # Extend to zero
        else:
            range_max = p_max
        range_min = p_min
    else:  # Mixed positive and negative
        range_min = p_min
        range_max = p_max

    # Always include zero if not already included
    range_min = min(range_min, 0.0)
    range_max = max(range_max, 0.0)

    # Make symmetric around zero for proper colormap centering
    abs_max = max(abs(range_min), abs(range_max))

    # Ensure non-zero range
    if abs_max == 0:
        abs_max = 0.01

    return -abs_max, abs_max


def calculate_zero_biased_axis_range(data_values, padding_factor=0.05):
    """
    Calculate axis range for x-y plots that includes zero when appropriate.

    Same zero-bias logic as color scales but without forcing symmetry.
    Adds padding for visual clarity.

    Parameters
    ----------
    data_values : array-like
        Data values to calculate range for (should be finite values only)
    padding_factor : float
        Fraction of range to add as padding (default 0.05 for 5%)

    Returns
    -------
    tuple
        (ymin, ymax) range values with zero-bias logic and padding
    """
    if len(data_values) == 0:
        return -0.1, 0.1

    # Calculate data range
    data_min = np.nanmin(data_values)
    data_max = np.nanmax(data_values)

    # Apply zero-bias logic (same as colormap function)
    if data_min > 0 and data_max > 0:  # All positive
        if data_min > 0.5 * data_max:  # Lower end is > 50% of upper end
            range_min = 0.0  # Extend to zero
        else:
            range_min = data_min
        range_max = data_max
    elif data_min < 0 and data_max < 0:  # All negative
        if data_max < 0.5 * data_min:  # Upper end is > 50% of |lower end|
            range_max = 0.0  # Extend to zero
        else:
            range_max = data_max
        range_min = data_min
    else:  # Mixed positive and negative
        range_min = data_min
        range_max = data_max

    # Add padding for visual clarity
    data_range = range_max - range_min
    if data_range > 0:
        padding = data_range * padding_factor
        range_min -= padding
        range_max += padding

    # Ensure non-zero range
    if abs(range_max - range_min) < 1e-10:
        center = (range_min + range_max) / 2
        range_min = center - 0.1
        range_max = center + 0.1

    return range_min, range_max


def load_step3_results_from_netcdf(netcdf_path: str) -> Dict[str, Any]:
    """
    Load Step 3 scaling factor results from NetCDF file.

    Parameters
    ----------
    netcdf_path : str
        Path to step3_scaling_factors_{model}.nc file

    Returns
    -------
    Dict[str, Any]
        Dictionary containing Step 3 results in the same format as step3_calculate_scaling_factors_per_cell()
    """
    import xarray as xr

    print(f"Loading Step 3 results from: {netcdf_path}")

    # Load NetCDF file
    ds = xr.open_dataset(netcdf_path)

    # Extract arrays
    scaling_factors = ds.scaling_factors.values
    optimization_errors = ds.optimization_errors.values
    convergence_flags = ds.convergence_flags.values
    scaled_parameters = ds.scaled_parameters.values
    valid_mask = ds.valid_mask.values

    # Extract coordinate labels (handle both old and new dimension names)
    if 'response_func' in ds.coords:
        response_function_names = [str(name) for name in ds.response_func.values]
    elif 'damage_func' in ds.coords:
        response_function_names = [str(name) for name in ds.damage_func.values]
    else:
        raise ValueError("Neither 'response_func' nor 'damage_func' dimension found in NetCDF file")

    target_names = [str(name) for name in ds.target.values]
    scaled_param_names = [str(name) for name in ds.param.values]

    # Calculate summary statistics
    total_grid_cells = int(np.sum(valid_mask))
    successful_optimizations = int(np.sum(convergence_flags[valid_mask]))

    # Create coordinate metadata
    coordinates = {
        'lat': ds.lat.values,
        'lon': ds.lon.values,
    }

    # Create results dictionary matching Step 3 output format
    scaling_results = {
        'scaling_factors': scaling_factors,
        'optimization_errors': optimization_errors,
        'convergence_flags': convergence_flags,
        'scaled_parameters': scaled_parameters,
        'scaled_param_names': scaled_param_names,
        'response_function_names': response_function_names,
        'target_names': target_names,
        'total_grid_cells': total_grid_cells,
        'successful_optimizations': successful_optimizations,
        'reference_ssp': ds.attrs.get('reference_ssp', 'unknown'),
        'valid_mask': valid_mask,
        '_coordinates': coordinates
    }

    ds.close()

    print(f"Loaded Step 3 results:")
    print(f"  Valid grid cells: {total_grid_cells}")
    print(f"  Successful optimizations: {successful_optimizations}")
    print(f"  Success rate: {100*successful_optimizations/max(1, total_grid_cells):.1f}%")
    print(f"  Response functions: {len(response_function_names)}")
    print(f"  Target patterns: {len(target_names)}")

    return scaling_results


def create_forward_model_maps_visualization(forward_results, config, output_dir, model_name, all_netcdf_data):
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
    all_netcdf_data : dict
        All loaded NetCDF data (not used but kept for consistency)

    Returns
    -------
    tuple
        (linear_pdf_path, log10_pdf_path) - Paths to both generated PDF files
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.backends.backend_pdf import PdfPages
    import os

    # Generate output filenames
    linear_pdf_filename = f"step4_forward_model_maps_{model_name}.pdf"
    log10_pdf_filename = f"step4_forward_model_maps_log10_{model_name}.pdf"
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

    # Calculate total maps and pages
    total_maps = len(target_names) * len(response_function_names) * len(forward_ssps)
    maps_per_page = 3
    total_pages = (total_maps + maps_per_page - 1) // maps_per_page  # Ceiling division

    print(f"Creating Step 4 maps: {total_maps} maps across {total_pages} pages (3 maps per page)")
    print(f"  {len(target_names)} targets × {len(response_function_names)} response functions × {len(forward_ssps)} SSPs")
    print(f"  Generating both linear and log10 scale PDFs in parallel")

    # Create both PDFs with multi-page layout
    with PdfPages(linear_pdf_path) as linear_pdf, PdfPages(log10_pdf_path) as log10_pdf:
        map_idx = 0
        page_num = 0
        linear_fig = None
        log10_fig = None

        # Loop through all combinations (target innermost for 3-per-page grouping)
        for ssp in forward_ssps:

            for damage_idx, damage_name in enumerate(response_function_names):
                damage_config = config['response_function_scalings'][damage_idx]

                for target_idx, target_name in enumerate(target_names):
                    target_config = config['gdp_targets'][target_idx]

                    # Start new page if needed (every 3 maps)
                    if map_idx % maps_per_page == 0:
                        if map_idx > 0:
                            # Save previous pages to both PDFs
                            for fig, pdf in [(linear_fig, linear_pdf), (log10_fig, log10_pdf)]:
                                plt.figure(fig.number)
                                plt.tight_layout()
                                plt.subplots_adjust(top=0.93, bottom=0.05)
                                pdf.savefig(fig, bbox_inches='tight')
                                plt.close(fig)

                        # Create new pages for both PDFs
                        page_num += 1
                        linear_fig = plt.figure(figsize=(12, 16))
                        linear_fig.suptitle(f'Step 4: Forward Model Results (Linear Scale) - {model_name} - Page {page_num}/{total_pages}',
                                    fontsize=16, fontweight='bold', y=0.98)

                        log10_fig = plt.figure(figsize=(12, 16))
                        log10_fig.suptitle(f'Step 4: Forward Model Results (Log10 Scale) - {model_name} - Page {page_num}/{total_pages}',
                                    fontsize=16, fontweight='bold', y=0.98)

                    # Position on current page (1-3)
                    subplot_idx = (map_idx % maps_per_page) + 1

                    # Get SSP-specific data
                    ssp_results = forward_results['forward_results'][ssp]
                    gdp_climate = ssp_results['gdp_climate']  # [lat, lon, response_func, target, time]
                    gdp_weather = ssp_results['gdp_weather']  # [lat, lon, response_func, target, time]

                    # Extract data for this combination: [lat, lon, time]
                    gdp_climate_combo = gdp_climate[:, :, damage_idx, target_idx, :]
                    gdp_weather_combo = gdp_weather[:, :, damage_idx, target_idx, :]

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
                    linear_ax = plt.subplot(maps_per_page, 1, subplot_idx)

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
                    linear_ax.set_title(f'{ssp.upper()} × {target_name} × {damage_name}\n'
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
                    log10_ax = plt.subplot(maps_per_page, 1, subplot_idx)

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
                            print(f"    WARNING: Extreme high ratios detected for {ssp.upper()} × {target_name} × {damage_name}")
                            print(f"             log10(max_ratio) = {log_actual_max:.2f} (ratio = {10**log_actual_max:.2e})")
                            print(f"             at grid cell indices: lat_idx={max_lat_idx}, lon_idx={max_lon_idx}")
                        if log_actual_min < -5:  # log10(ratio) < -5 means ratio < 0.00001
                            # Find indices of minimum ratio
                            min_indices = np.where((valid_mask & np.isfinite(impact_ratio_log10)) &
                                                 (impact_ratio_log10 == log_actual_min))
                            min_lat_idx, min_lon_idx = min_indices[0][0], min_indices[1][0]
                            print(f"    WARNING: Extreme low ratios detected for {ssp.upper()} × {target_name} × {damage_name}")
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
                    log10_ax.set_title(f'{ssp.upper()} × {target_name} × {damage_name}\n'
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

                    map_idx += 1

        # Save the final pages
        if map_idx > 0:
            for fig, pdf in [(linear_fig, linear_pdf), (log10_fig, log10_pdf)]:
                plt.figure(fig.number)
                plt.tight_layout()
                plt.subplots_adjust(top=0.93, bottom=0.05)
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

    print(f"Forward model maps saved to:")
    print(f"  Linear scale: {linear_pdf_path}")
    print(f"  Log10 scale: {log10_pdf_path}")
    print(f"  ({total_pages} pages each, 3 maps per page)")
    return (linear_pdf_path, log10_pdf_path)


def create_country_pdf_books(all_results, params, output_dir, run_name, timestamp):
    """Create PDF books with one book per country, one page per scaling set."""
    output_dir = Path(output_dir)
    
    print(f"\nCreating PDF books for {len(all_results)} countries...")
    
    pdf_files = []
    for i, (country, results) in enumerate(sorted(all_results.items()), 1):
        print(f"  [{i}/{len(all_results)}] Creating book for {country}...")
        
        pdf_file = output_dir / f"COIN_SSP_{country.replace(' ', '_')}_Book_{run_name}_{timestamp}.pdf"
        
        with PdfPages(pdf_file) as pdf:
            for j, (scaling_name, scaling_result) in enumerate(results['scaling_results'].items(), 1):
                print(f"    Page {j}: {scaling_name}")
                
                # Create figure for this scaling set
                fig = plt.figure(figsize=(8.5, 11))  # Letter size portrait
                create_country_scaling_page(country, scaling_name, results, scaling_result, params, fig)
                
                # Save to PDF
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
        
        pdf_files.append(pdf_file)
        print(f"    Saved: {pdf_file}")
    
    print(f"  Created {len(pdf_files)} PDF books")
    return pdf_files

# =============================================================================
# Temporal Alignment Utilities
# =============================================================================

def extract_year_coordinate(dataset, coord_names=None):
    """
    Extract actual year values from NetCDF time coordinate.

    Parameters
    ----------
    dataset : xarray.Dataset
        NetCDF dataset with time coordinate
    coord_names : list, optional
        Coordinate names to try, by default ['year', 'time', 'axis_0']

    Returns
    -------
    tuple
        (years, valid_mask) where years are the extracted year values
        and valid_mask indicates which time indices are valid
    """
    if coord_names is None:
        coord_names = ['year', 'time', 'axis_0']

    time_coord = None
    for coord_name in coord_names:
        if coord_name in dataset.coords:
            time_coord = dataset.coords[coord_name]
            break

    if time_coord is None:
        raise ValueError(f"No time coordinate found. Tried: {coord_names}")

    # Handle different time coordinate types
    time_values = time_coord.values

    # If already integers (years), return as-is
    if np.issubdtype(time_values.dtype, np.integer):
        return time_values.astype(int), np.ones(len(time_values), dtype=bool)

    # If datetime-like objects, extract year
    if hasattr(time_values[0], 'year'):
        years = np.array([t.year for t in time_values])
        return years, np.ones(len(years), dtype=bool)

    # If floating point values, treat as raw years (ignore "years since" units)
    if np.issubdtype(time_values.dtype, np.floating):
        # Filter for reasonable year values and ignore units encoding
        valid_mask = np.isfinite(time_values) & (time_values > 1800) & (time_values < 2200)
        if not np.any(valid_mask):
            raise ValueError("No valid years found in time coordinate")

        valid_years = time_values[valid_mask].astype(int)
        if np.sum(valid_mask) != len(time_values):
            print(f"    Warning: Filtered out {len(time_values) - np.sum(valid_mask)} time values with incomplete data")

        return valid_years, valid_mask

    raise ValueError(f"Cannot extract years from time coordinate with dtype {time_values.dtype}")

def interpolate_to_annual_grid(original_years, data_array, target_years):
    """
    Linearly interpolate data to annual time grid.

    Parameters
    ----------
    original_years : numpy.ndarray
        Original year values (1D array)
    data_array : numpy.ndarray
        Data array with time as first dimension: (time, ...)
    target_years : numpy.ndarray
        Target annual year values (1D array)

    Returns
    -------
    numpy.ndarray
        Interpolated data array with shape (len(target_years), ...)
    """
    # If already on target grid, return as-is
    if len(original_years) == len(target_years) and np.allclose(original_years, target_years):
        return data_array

    # Get spatial dimensions
    spatial_shape = data_array.shape[1:]

    # Flatten spatial dimensions for vectorized interpolation
    data_flat = data_array.reshape(len(original_years), -1)

    # Interpolate each spatial point
    interpolated_flat = np.zeros((len(target_years), data_flat.shape[1]))

    for i in range(data_flat.shape[1]):
        interpolated_flat[:, i] = np.interp(target_years, original_years, data_flat[:, i])

    # Reshape back to original spatial dimensions
    return interpolated_flat.reshape((len(target_years),) + spatial_shape)

# =============================================================================
# Target GDP Utilities (moved from coin_ssp_target_gdp.py)
# =============================================================================

def load_and_concatenate_climate_data(config, ssp_name, data_type):
    """
    Load and concatenate historical and SSP-specific climate data files.

    For temperature and precipitation, loads from:
    1. CLIMATE_{model_name}_historical.nc
    2. CLIMATE_{model_name}_{ssp_name}.nc
    Then concatenates along time dimension.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary containing climate_model.model_name and file patterns
    ssp_name : str
        SSP scenario name
    data_type : str
        'temperature' or 'precipitation'

    Returns
    -------
    tuple
        (concatenated_data, concatenated_years, valid_mask, coordinates)
    """
    import os

    climate_model = config['climate_model']
    input_dir = climate_model['input_directory']
    model_name = climate_model['model_name']

    # Get file prefix and variable name from new configuration structure
    if data_type == 'temperature':
        prefix = climate_model['file_prefixes']['tas_file_prefix']
        var_name = climate_model['variable_names']['tas_var_name']
    elif data_type == 'precipitation':
        prefix = climate_model['file_prefixes']['pr_file_prefix']
        var_name = climate_model['variable_names']['pr_var_name']
    else:
        raise ValueError(f"Unsupported data_type for climate data: {data_type}")

    # Historical file
    hist_filename = f"{prefix}_{model_name}_historical.nc"
    hist_filepath = os.path.join(input_dir, hist_filename)

    # SSP-specific file
    ssp_filename = f"{prefix}_{model_name}_{ssp_name}.nc"
    ssp_filepath = os.path.join(input_dir, ssp_filename)

    print(f"    Loading historical: {hist_filename}")
    print(f"    Loading SSP: {ssp_filename}")

    # Load historical data
    hist_ds = xr.open_dataset(hist_filepath, decode_times=False)
    hist_years, hist_valid_mask = extract_year_coordinate(hist_ds)

    # Load SSP data
    ssp_ds = xr.open_dataset(ssp_filepath, decode_times=False)
    ssp_years, ssp_valid_mask = extract_year_coordinate(ssp_ds)

    # Extract data arrays
    hist_data_all = hist_ds[var_name].values
    ssp_data_all = ssp_ds[var_name].values

    # Apply valid masks
    hist_data = hist_data_all[hist_valid_mask]
    ssp_data = ssp_data_all[ssp_valid_mask]

    # Convert temperature from Kelvin to Celsius
    if data_type == 'temperature':
        hist_data = hist_data - 273.15
        ssp_data = ssp_data - 273.15

    # Concatenate along time dimension
    concatenated_data = np.concatenate([hist_data, ssp_data], axis=0)
    concatenated_years = np.concatenate([hist_years, ssp_years])

    # Get coordinates from historical file (should be same for both)
    lat = hist_ds.lat.values
    lon = hist_ds.lon.values

    print(f"    Historical: {len(hist_years)} years ({hist_years[0]}-{hist_years[-1]})")
    print(f"    SSP: {len(ssp_years)} years ({ssp_years[0]}-{ssp_years[-1]})")
    print(f"    Concatenated: {len(concatenated_years)} years ({concatenated_years[0]}-{concatenated_years[-1]})")

    hist_ds.close()
    ssp_ds.close()

    return concatenated_data, concatenated_years, lat, lon


def load_and_concatenate_population_data(config, ssp_name):
    """
    Load and concatenate historical and SSP-specific population data files.

    Loads from:
    1. POP_{model_name}_hist.nc
    2. POP_{model_name}_{short_ssp_name}.nc
    Then concatenates along time dimension.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary containing climate_model.model_name and file patterns
    ssp_name : str
        SSP scenario name (e.g., 'ssp245')

    Returns
    -------
    tuple
        (concatenated_data, concatenated_years, coordinates)
    """
    import os

    climate_model = config['climate_model']
    input_dir = climate_model['input_directory']
    model_name = climate_model['model_name']
    prefix = climate_model['file_prefixes']['pop_file_prefix']
    var_name = climate_model['variable_names']['pop_var_name']

    # Truncate SSP name to short form (e.g., 'ssp245' -> 'ssp2')
    if ssp_name.startswith('ssp') and len(ssp_name) >= 4:
        short_ssp = ssp_name[:4]  # Keep 'ssp' + first digit
    else:
        short_ssp = ssp_name

    # Historical file
    hist_filename = f"{prefix}_{model_name}_hist.nc"
    hist_filepath = os.path.join(input_dir, hist_filename)

    # SSP-specific file
    ssp_filename = f"{prefix}_{model_name}_{short_ssp}.nc"
    ssp_filepath = os.path.join(input_dir, ssp_filename)

    print(f"    Loading historical: {hist_filename}")
    print(f"    Loading SSP: {ssp_filename}")

    # Load historical data
    hist_ds = xr.open_dataset(hist_filepath, decode_times=False)
    hist_years, hist_valid_mask = extract_year_coordinate(hist_ds)
    hist_data_all = hist_ds[var_name].values
    hist_data = hist_data_all[hist_valid_mask]

    # Load SSP data
    ssp_ds = xr.open_dataset(ssp_filepath, decode_times=False)
    ssp_years, ssp_valid_mask = extract_year_coordinate(ssp_ds)
    ssp_data_all = ssp_ds[var_name].values
    ssp_data = ssp_data_all[ssp_valid_mask]

    # Concatenate along time dimension
    concatenated_data = np.concatenate([hist_data, ssp_data], axis=0)
    concatenated_years = np.concatenate([hist_years, ssp_years])

    # Get coordinates from historical file
    lat = hist_ds.lat.values
    lon = hist_ds.lon.values

    print(f"    Historical: {len(hist_years)} years ({hist_years[0]}-{hist_years[-1]})")
    print(f"    SSP: {len(ssp_years)} years ({ssp_years[0]}-{ssp_years[-1]})")
    print(f"    Concatenated: {len(concatenated_years)} years ({concatenated_years[0]}-{concatenated_years[-1]})")

    hist_ds.close()
    ssp_ds.close()

    return concatenated_data, concatenated_years, lat, lon


def load_gridded_data(config, case_name):
    """
    Load all four NetCDF files and return as a temporally-aligned dataset.

    All variables are interpolated to annual resolution and aligned to the
    same common year range that all variables share after interpolation.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary containing climate_model.model_name, file patterns, and time_periods.prediction_period.start_year
    case_name : str
        SSP scenario name

    Returns
    -------
    dict
        Dictionary containing temporally-aligned data:
        - 'tas': temperature data (time, lat, lon) - annual, common years
        - 'pr': precipitation data (time, lat, lon) - annual, common years
        - 'gdp': GDP data (time, lat, lon) - annual, common years (interpolated, exponential growth applied before prediction year)
        - 'pop': population data (time, lat, lon) - annual, common years (exponential growth applied before prediction year)
        - 'lat': latitude coordinates
        - 'lon': longitude coordinates
        - 'tas_years': temperature time axis (annual years)
        - 'pr_years': precipitation time axis (annual years)
        - 'gdp_years': GDP time axis (annual years, interpolated)
        - 'pop_years': population time axis (annual years)
        - 'common_years': final common year range for all variables
    """
    import os

    # Extract configuration values
    model_name = config['climate_model']['model_name']
    prediction_year = config['time_periods']['prediction_period']['start_year']

    print(f"Loading and aligning NetCDF data for {model_name} {case_name}...")

    # Load temperature data (concatenate historical + SSP)
    print("  Loading temperature data...")
    tas_raw, tas_years_raw, lat, lon = load_and_concatenate_climate_data(config, case_name, 'temperature')

    # Load precipitation data (concatenate historical + SSP)
    print("  Loading precipitation data...")
    pr_raw, pr_years_raw, _, _ = load_and_concatenate_climate_data(config, case_name, 'precipitation')

    # Load GDP data (single file with short SSP name, no concatenation)
    print("  Loading GDP data...")
    climate_model = config['climate_model']
    input_dir = climate_model['input_directory']
    gdp_prefix = climate_model['file_prefixes']['gdp_file_prefix']
    gdp_var_name = climate_model['variable_names']['gdp_var_name']

    # Truncate SSP name to short form (e.g., 'ssp245' -> 'ssp2')
    if case_name.startswith('ssp') and len(case_name) >= 4:
        short_ssp = case_name[:4]  # Keep 'ssp' + first digit
    else:
        short_ssp = case_name

    gdp_filename = f"{gdp_prefix}_{model_name}_{short_ssp}.nc"
    gdp_file = os.path.join(input_dir, gdp_filename)
    print(f"    Loading: {gdp_filename}")

    gdp_ds = xr.open_dataset(gdp_file, decode_times=False)
    gdp_raw_all = gdp_ds[gdp_var_name].values
    gdp_years_raw, gdp_valid_mask = extract_year_coordinate(gdp_ds)
    gdp_raw = gdp_raw_all[gdp_valid_mask]
    gdp_ds.close()

    # Load population data (concatenate historical + SSP)
    print("  Loading population data...")
    pop_raw, pop_years_raw, _, _ = load_and_concatenate_population_data(config, case_name)

    print(f"  Original time ranges:")
    print(f"    Temperature: {tas_years_raw.min()}-{tas_years_raw.max()} ({len(tas_years_raw)} points)")
    print(f"    Precipitation: {pr_years_raw.min()}-{pr_years_raw.max()} ({len(pr_years_raw)} points)")
    print(f"    GDP: {gdp_years_raw.min()}-{gdp_years_raw.max()} ({len(gdp_years_raw)} points)")
    print(f"    Population: {pop_years_raw.min()}-{pop_years_raw.max()} ({len(pop_years_raw)} points)")

    # Create annual grids for each variable
    print("  Interpolating to annual grids...")
    tas_years_annual = np.arange(tas_years_raw.min(), tas_years_raw.max() + 1)
    pr_years_annual = np.arange(pr_years_raw.min(), pr_years_raw.max() + 1)
    gdp_years_annual = np.arange(gdp_years_raw.min(), gdp_years_raw.max() + 1)
    pop_years_annual = np.arange(pop_years_raw.min(), pop_years_raw.max() + 1)

    # Interpolate each variable to annual grid
    tas_annual = interpolate_to_annual_grid(tas_years_raw, tas_raw, tas_years_annual)
    pr_annual = interpolate_to_annual_grid(pr_years_raw, pr_raw, pr_years_annual)
    gdp_annual = interpolate_to_annual_grid(gdp_years_raw, gdp_raw, gdp_years_annual)
    pop_annual = interpolate_to_annual_grid(pop_years_raw, pop_raw, pop_years_annual)

    # Find common year range (intersection of all ranges)
    common_start = max(tas_years_annual.min(), pr_years_annual.min(),
                      gdp_years_annual.min(), pop_years_annual.min())
    common_end = min(tas_years_annual.max(), pr_years_annual.max(),
                    gdp_years_annual.max(), pop_years_annual.max())
    common_years = np.arange(common_start, common_end + 1)

    print(f"  Common year range: {common_start}-{common_end} ({len(common_years)} years)")

    # Subset all variables to common years
    def subset_to_common_years(data, years, common_years):
        start_idx = np.where(years == common_years[0])[0][0]
        end_idx = np.where(years == common_years[-1])[0][0] + 1
        return data[start_idx:end_idx], years[start_idx:end_idx]

    tas_aligned, tas_years_aligned = subset_to_common_years(tas_annual, tas_years_annual, common_years)
    pr_aligned, pr_years_aligned = subset_to_common_years(pr_annual, pr_years_annual, common_years)
    gdp_aligned, gdp_years_aligned = subset_to_common_years(gdp_annual, gdp_years_annual, common_years)
    pop_aligned, pop_years_aligned = subset_to_common_years(pop_annual, pop_years_annual, common_years)

    # Verify alignment
    assert np.array_equal(tas_years_aligned, common_years), "Temperature years not aligned"
    assert np.array_equal(pr_years_aligned, common_years), "Precipitation years not aligned"
    assert np.array_equal(gdp_years_aligned, common_years), "GDP years not aligned"
    assert np.array_equal(pop_years_aligned, common_years), "Population years not aligned"

    print(f"  ✅ All variables aligned to {len(common_years)} common years")

    # Apply exponential growth modification for GDP and population before prediction year
    if prediction_year in common_years:
        idx_prediction_year = np.where(common_years == prediction_year)[0][0]

        if idx_prediction_year > 0:
            print(f"  Applying exponential growth modification for years {common_years[0]}-{prediction_year}")

            # For each grid cell, modify GDP and population using exponential interpolation
            for lat_idx in range(gdp_aligned.shape[1]):
                for lon_idx in range(gdp_aligned.shape[2]):
                    # GDP exponential growth
                    gdp_first = gdp_aligned[0, lat_idx, lon_idx]
                    gdp_prediction = gdp_aligned[idx_prediction_year, lat_idx, lon_idx]

                    if gdp_first > 0 and gdp_prediction > 0:
                        for idx in range(1, idx_prediction_year):
                            gdp_aligned[idx, lat_idx, lon_idx] = gdp_first * (gdp_prediction / gdp_first) ** (idx / idx_prediction_year)

                    # Population exponential growth
                    pop_first = pop_aligned[0, lat_idx, lon_idx]
                    pop_prediction = pop_aligned[idx_prediction_year, lat_idx, lon_idx]

                    if pop_first > 0 and pop_prediction > 0:
                        for idx in range(1, idx_prediction_year):
                            pop_aligned[idx, lat_idx, lon_idx] = pop_first * (pop_prediction / pop_first) ** (idx / idx_prediction_year)

    # Note: Individual datasets are closed within their respective loading functions

    return {
        'tas': tas_aligned,
        'pr': pr_aligned,
        'gdp': gdp_aligned,
        'pop': pop_aligned,
        'lat': lat,
        'lon': lon,
        'tas_years': tas_years_aligned,
        'pr_years': pr_years_aligned,
        'gdp_years': gdp_years_aligned,
        'pop_years': pop_years_aligned,
        'common_years': common_years
    }

def calculate_area_weights(lat):
    """
    Calculate area weights proportional to cosine of latitude.
    
    Parameters
    ---------- 
    lat : array
        Latitude coordinates in degrees
        
    Returns
    -------
    array
        Area weights with same shape as lat, normalized to sum to 1
    """
    weights = np.cos(np.radians(lat))
    return weights / weights.sum()

def calculate_time_means(data, years, start_year, end_year):
    """
    Calculate temporal mean over specified year range.
    
    Parameters
    ----------
    data : array
        Data array with time as first dimension
    years : array  
        Year values corresponding to time dimension
    start_year : int
        Start year (inclusive)
    end_year : int
        End year (inclusive)
        
    Returns
    -------
    array
        Time-averaged data with time dimension removed
    """
    mask = (years >= start_year) & (years <= end_year)
    return data[mask].mean(axis=0)

def calculate_global_mean(data, lat, valid_mask):
    """
    Calculate area-weighted global mean using only valid grid cells.

    Parameters
    ----------
    data : array
        2D spatial data (lat, lon)
    lat : array
        Latitude coordinates
    valid_mask : array
        2D boolean mask (lat, lon) indicating valid grid cells

    Returns
    -------
    float
        Area-weighted global mean over valid cells
    """
    weights = calculate_area_weights(lat)
    # Expand weights to match data shape: (lat,) -> (lat, lon)
    weights_2d = np.broadcast_to(weights[:, np.newaxis], data.shape)

    # Apply mask to both data and weights
    masked_data = np.where(valid_mask, data, np.nan)
    masked_weights = np.where(valid_mask, weights_2d, 0.0)

    # Use np.nansum and np.sum to handle NaN values properly
    return np.nansum(masked_data * masked_weights) / np.sum(masked_weights)


def calculate_global_median(data, lat, valid_mask):
    """
    Calculate area-weighted global median using only valid grid cells.

    Uses the weighted median algorithm: sort data by value, compute cumulative
    sum of weights, and interpolate to find the value at 50% cumulative weight.

    Parameters
    ----------
    data : array
        2D spatial data (lat, lon)
    lat : array
        Latitude coordinates
    valid_mask : array
        2D boolean mask (lat, lon) indicating valid grid cells

    Returns
    -------
    float
        Area-weighted global median over valid cells
    """
    weights = calculate_area_weights(lat)
    # Expand weights to match data shape: (lat,) -> (lat, lon)
    weights_2d = np.broadcast_to(weights[:, np.newaxis], data.shape)

    # Apply mask to both data and weights
    masked_data = np.where(valid_mask, data, np.nan)
    masked_weights = np.where(valid_mask, weights_2d, 0.0)

    # Flatten and remove NaN values
    flat_data = masked_data.flatten()
    flat_weights = masked_weights.flatten()

    # Remove NaN entries
    valid_indices = ~np.isnan(flat_data)
    valid_data = flat_data[valid_indices]
    valid_weights = flat_weights[valid_indices]

    if len(valid_data) == 0:
        return np.nan

    # Create 2-column array and sort by data values (column 1)
    # Column 0: weights, Column 1: data values
    combined = np.column_stack([valid_weights, valid_data])
    sorted_indices = np.argsort(combined[:, 1])  # Sort by column 1 (data values)
    sorted_combined = combined[sorted_indices]

    # Calculate cumulative sum of weights
    cumsum_weights = np.cumsum(sorted_combined[:, 0])
    total_weight = cumsum_weights[-1]
    half_weight = total_weight / 2.0

    # Find interpolated value at half total weight
    if half_weight <= cumsum_weights[0]:
        return sorted_combined[0, 1]  # First value
    elif half_weight >= cumsum_weights[-1]:
        return sorted_combined[-1, 1]  # Last value
    else:
        # Interpolate
        return np.interp(half_weight, cumsum_weights, sorted_combined[:, 1])


# =============================================================================
# Target GDP Reduction Calculation Functions
# Extracted from calculate_target_gdp_amounts.py for reuse in integrated workflow
# =============================================================================

def calculate_constant_target_reduction(gdp_amount_value, temp_ref_shape):
    """
    Calculate constant GDP reduction across all grid cells.
    
    Parameters
    ----------
    gdp_amount_value : float
        Constant reduction value (e.g., -0.10 for 10% reduction)
    temp_ref_shape : tuple
        Shape of temperature reference array for output sizing
        
    Returns
    -------
    np.ndarray
        Constant reduction array with shape temp_ref_shape
    """
    return np.full(temp_ref_shape, gdp_amount_value, dtype=np.float64)


def calculate_linear_target_reduction(linear_config, temp_ref, gdp_target, lat, valid_mask):
    """
    Calculate linear temperature-dependent GDP reduction using constraint satisfaction.
    
    Implements the mathematical framework:
    reduction(T) = a0 + a1 * T
    
    Subject to two constraints:
    1. Point constraint: reduction(T_ref) = value_at_ref
    2. GDP-weighted global mean: ∑[w_i * gdp_i * (1 + reduction(T_i))] / ∑[w_i * gdp_i] = target_mean
    
    Parameters
    ----------
    linear_config : dict
        Configuration containing:
        - 'global_mean_amount': Target GDP-weighted global mean (e.g., -0.10)
        - 'reference_temperature': Reference temperature point (e.g., 30.0)
        - 'amount_at_reference_temp': Reduction at reference temperature (e.g., -0.25)
    temp_ref : np.ndarray
        Reference period temperature array [lat, lon]
    gdp_target : np.ndarray
        Target period GDP array [lat, lon]
    lat : np.ndarray
        Latitude coordinate array for area weighting
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'reduction_array': Linear reduction array [lat, lon]
        - 'coefficients': {'a0': intercept, 'a1': slope}
        - 'constraint_verification': Verification of constraint satisfaction
    """
    # Extract configuration parameters
    global_mean_linear = linear_config['global_mean_amount']
    T_ref_linear = linear_config['reference_temperature'] 
    value_at_ref_linear = linear_config['amount_at_reference_temp']
    
    # Calculate GDP-weighted global means
    global_gdp_target = calculate_global_mean(gdp_target, lat, valid_mask)
    gdp_weighted_temp_mean = np.float64(calculate_global_mean(gdp_target * temp_ref, lat, valid_mask) / global_gdp_target)
    
    # Set up weighted least squares system for exact constraint satisfaction
    X = np.array([
        [1.0, T_ref_linear],                    # Point constraint equation
        [1.0, gdp_weighted_temp_mean]           # GDP-weighted global mean equation
    ], dtype=np.float64)
    
    y = np.array([
        value_at_ref_linear,                    # Target at reference temperature
        global_mean_linear                      # Target GDP-weighted global mean  
    ], dtype=np.float64)
    
    # Solve for coefficients: [a0, a1]
    coefficients = np.linalg.solve(X, y)
    a0_linear, a1_linear = coefficients
    
    # Calculate linear reduction array
    linear_reduction = a0_linear + a1_linear * temp_ref
    
    # Verify constraint satisfaction
    constraint1_check = a0_linear + a1_linear * T_ref_linear  # Should equal value_at_ref_linear
    constraint2_check = calculate_global_mean(gdp_target * (1 + linear_reduction), lat, valid_mask) / global_gdp_target - 1  # Should equal global_mean_linear
    
    return {
        'reduction_array': linear_reduction.astype(np.float64),
        'coefficients': {'a0': float(a0_linear), 'a1': float(a1_linear)},
        'constraint_verification': {
            'point_constraint': {
                'achieved': float(constraint1_check),
                'target': float(value_at_ref_linear),
                'error': float(abs(constraint1_check - value_at_ref_linear))
            },
            'global_mean_constraint': {
                'achieved': float(constraint2_check),
                'target': float(global_mean_linear),
                'error': float(abs(constraint2_check - global_mean_linear))
            }
        },
        'gdp_weighted_temp_mean': float(gdp_weighted_temp_mean)
    }


def calculate_quadratic_target_reduction(quadratic_config, temp_ref, gdp_target, lat, valid_mask):
    """
    Calculate quadratic temperature-dependent GDP reduction using derivative constraint.

    Implements the mathematical framework:
    reduction(T) = a + b*T + c*T²

    Subject to three constraints:
    1. Zero point: reduction(T₀) = 0
    2. Derivative at zero: reduction'(T₀) = derivative_at_zero_amount_temperature
    3. GDP-weighted global mean: ∑[w_i * gdp_i * (1 + reduction(T_i))] / ∑[w_i * gdp_i] = target_mean

    Parameters
    ----------
    quadratic_config : dict
        Configuration containing:
        - 'global_mean_amount': Target GDP-weighted global mean (e.g., -0.10)
        - 'zero_amount_temperature': Temperature with zero reduction (e.g., 13.5)
        - 'derivative_at_zero_amount_temperature': Slope at T₀ (e.g., -0.01)
    temp_ref : np.ndarray
        Reference period temperature array [lat, lon]
    gdp_target : np.ndarray
        Target period GDP array [lat, lon]
    lat : np.ndarray
        Latitude coordinate array for area weighting

    Returns
    -------
    dict
        Dictionary containing:
        - 'reduction_array': Quadratic reduction array [lat, lon]
        - 'coefficients': {'a': constant, 'b': linear, 'c': quadratic}
        - 'constraint_verification': Verification of constraint satisfaction
    """
    # Extract configuration parameters
    global_mean_quad = quadratic_config['global_mean_amount']
    T0 = quadratic_config['zero_amount_temperature']
    derivative_at_T0 = quadratic_config['derivative_at_zero_amount_temperature']

    # Calculate GDP-weighted global means
    global_gdp_target = calculate_global_mean(gdp_target, lat, valid_mask)
    gdp_weighted_temp_mean = np.float64(calculate_global_mean(gdp_target * temp_ref, lat, valid_mask) / global_gdp_target)
    gdp_weighted_temp2_mean = np.float64(calculate_global_mean(gdp_target * temp_ref**2, lat, valid_mask) / global_gdp_target)

    # Mathematical solution for quadratic: f(T) = a + b*T + c*T²
    # Given constraints:
    # 1. f(T₀) = 0 (zero reduction at T₀)
    # 2. f'(T₀) = derivative_at_T0 (slope at T₀)
    # 3. GDP-weighted global mean = global_mean_quad

    # From constraint derivation:
    # c = (global_mean_amount - derivative_at_T0*(GDP_weighted_temp_mean - T0)) /
    #     (T0² - 2*T0*GDP_weighted_temp_mean + GDP_weighted_temp2_mean)
    # b = derivative_at_T0 - 2*c*T0
    # a = -derivative_at_T0*T0 + c*T0²

    denominator = T0**2 - 2*T0*gdp_weighted_temp_mean + gdp_weighted_temp2_mean
    c_quad = (global_mean_quad - derivative_at_T0*(gdp_weighted_temp_mean - T0)) / denominator
    b_quad = derivative_at_T0 - 2*c_quad*T0
    a_quad = -derivative_at_T0*T0 + c_quad*T0**2

    # Calculate quadratic reduction array using absolute temperature
    quadratic_reduction = a_quad + b_quad * temp_ref + c_quad * temp_ref**2

    # Verify constraint satisfaction
    constraint1_check = a_quad + b_quad * T0 + c_quad * T0**2  # Should be 0 at T0
    constraint2_check = b_quad + 2 * c_quad * T0  # Derivative at T0: should equal derivative_at_T0
    constraint3_check = calculate_global_mean(gdp_target * (1 + quadratic_reduction), lat, valid_mask) / global_gdp_target - 1

    return {
        'reduction_array': quadratic_reduction.astype(np.float64),
        'coefficients': {'a': float(a_quad), 'b': float(b_quad), 'c': float(c_quad)},
        'constraint_verification': {
            'zero_point_constraint': {
                'achieved': float(constraint1_check),
                'target': 0.0,
                'error': float(abs(constraint1_check))
            },
            'derivative_constraint': {
                'achieved': float(constraint2_check),
                'target': float(derivative_at_T0),
                'error': float(abs(constraint2_check - derivative_at_T0))
            },
            'global_mean_constraint': {
                'achieved': float(constraint3_check),
                'target': float(global_mean_quad),
                'error': float(abs(constraint3_check - global_mean_quad))
            }
        },
        'gdp_weighted_temp_mean': float(gdp_weighted_temp_mean),
        'gdp_weighted_temp2_mean': float(gdp_weighted_temp2_mean),
        'derivative_at_zero_temp': float(derivative_at_T0),
        'zero_amount_temperature': float(T0)
    }


def calculate_all_target_reductions(target_configs, gridded_data):
    """
    Calculate all configured target GDP reductions using gridded data.

    This function processes multiple target configurations and automatically
    determines the reduction type from available parameters.

    Parameters
    ----------
    target_configs : list
        List of target configuration dictionaries, each containing:
        - 'target_name': Unique identifier
        - Type-specific parameters (determines calculation method):
          * Constant: 'gdp_amount'
          * Linear: 'global_mean_amount' (without zero point)
          * Quadratic: 'zero_amount_temperature'
    gridded_data : dict
        Dictionary containing gridded data arrays:
        - 'temp_ref': Reference period temperature [lat, lon]
        - 'gdp_target': Target period GDP [lat, lon]
        - 'lat': Latitude coordinates

    Returns
    -------
    dict
        Dictionary with target_name keys, each containing:
        - 'reduction_array': Calculated reduction array [lat, lon]
        - 'coefficients': Function coefficients (if applicable)
        - 'constraint_verification': Constraint satisfaction results
        - 'global_statistics': Global mean calculations
    """
    results = {}
    
    temp_ref = gridded_data['temp_ref']
    gdp_target = gridded_data['gdp_target']
    lat = gridded_data['lat']
    valid_mask = gridded_data['valid_mask']
    
    for target_config in target_configs:
        target_name = target_config['target_name']

        # Use explicit target_shape from configuration
        target_shape = target_config['target_shape']

        if target_shape == 'constant':
            # Constant reduction
            reduction_array = calculate_constant_target_reduction(
                target_config['gdp_amount'], temp_ref.shape
            )
            result = {
                'target_shape': target_shape,
                'reduction_array': reduction_array,
                'coefficients': None,
                'constraint_verification': None,
                'global_statistics': {
                    'gdp_weighted_mean': target_config['gdp_amount']
                }
            }

        elif target_shape == 'quadratic':
            # Quadratic reduction (has zero point)
            result = calculate_quadratic_target_reduction(target_config, temp_ref, gdp_target, lat, valid_mask)
            result['target_shape'] = target_shape

        elif target_shape == 'linear':
            # Linear reduction (has global mean constraint)
            result = calculate_linear_target_reduction(target_config, temp_ref, gdp_target, lat, valid_mask)
            result['target_shape'] = target_shape

        else:
            raise ValueError(f"Unknown target_shape '{target_shape}' for target '{target_name}'. "
                           f"Must be 'constant', 'linear', or 'quadratic'.")

        results[target_name] = result
    
    return results


# =============================================================================
# Centralized NetCDF Data Loading Functions
# For efficient loading of all SSP scenario data upfront
# =============================================================================

def load_all_netcdf_data(config: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    """
    Load all NetCDF files for all SSP scenarios at start of processing.
    
    This function loads all gridded data upfront to avoid repeated file I/O
    operations during processing steps. Since NetCDF files are small, this 
    approach optimizes performance by loading everything into memory once.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Integrated configuration dictionary containing:
        - climate_model: model name and file patterns
        - ssp_scenarios: reference_ssp and forward_simulation_ssps
        - time_periods: reference and target period specifications
    output_dir : str, optional
        Output directory path. If provided, writes NetCDF file with all loaded data
    
    Returns
    -------
    Dict[str, Any]
        Nested dictionary organized as:
        data[ssp_name][data_type] = array
        
        Structure:
        {
            'ssp245': {
                'temperature': np.array([time, lat, lon]),  # °C
                'precipitation': np.array([time, lat, lon]), # mm/day
                'gdp': np.array([time, lat, lon]),          # economic units
                'population': np.array([time, lat, lon])    # people
            },
            'ssp585': { ... },
            '_metadata': {
                'lat': np.array([lat]),           # latitude coordinates
                'lon': np.array([lon]),           # longitude coordinates  
                'time_periods': time_periods,     # reference and target periods
                'ssp_list': ['ssp245', 'ssp585', ...],  # all loaded SSPs
                'grid_shape': (nlat, nlon),       # spatial dimensions
                'time_shape': ntime               # temporal dimension
            }
        }
    """
    import os
    
    print("\n" + "="*60)
    print("LOADING ALL NETCDF DATA")  
    print("="*60)
    
    # Extract configuration
    model_name = config['climate_model']['model_name']
    reference_ssp = config['ssp_scenarios']['reference_ssp']
    forward_ssps = config['ssp_scenarios']['forward_simulation_ssps']
    time_periods = config['time_periods']

    # Create comprehensive SSP list (reference + forward, deduplicated)
    all_ssps = list(set([reference_ssp] + forward_ssps))
    all_ssps.sort()  # Consistent ordering

    print(f"Climate model: {model_name}")
    print(f"Reference SSP: {reference_ssp}")
    print(f"All SSPs to load: {all_ssps}")
    print(f"Total SSP scenarios: {len(all_ssps)}")

    # Log prediction period for exponential growth assumption
    prediction_start = time_periods['prediction_period']['start_year']
    print(f"Exponential growth assumption before year: {prediction_start}")
    
    # Initialize data structure
    all_data = {
        '_metadata': {
            'ssp_list': all_ssps,
            'time_periods': time_periods,
            'model_name': model_name
        }
    }
    
    # Load data for each SSP scenario
    for i, ssp_name in enumerate(all_ssps):
        print(f"\nLoading SSP scenario: {ssp_name} ({i+1}/{len(all_ssps)})")
        
        try:
            # Resolve file paths for this SSP
            temp_file = resolve_netcdf_filepath(config, 'temperature', ssp_name) 
            precip_file = resolve_netcdf_filepath(config, 'precipitation', ssp_name)
            gdp_file = resolve_netcdf_filepath(config, 'gdp', ssp_name)
            pop_file = resolve_netcdf_filepath(config, 'population', ssp_name)
            
            print(f"  Temperature: {os.path.basename(temp_file)}")
            print(f"  Precipitation: {os.path.basename(precip_file)}")
            print(f"  GDP: {os.path.basename(gdp_file)}")
            print(f"  Population: {os.path.basename(pop_file)}")
            
            # Load gridded data for this SSP using existing function
            ssp_data = load_gridded_data(config, ssp_name)
            
            # Store in organized structure
            all_data[ssp_name] = {
                'temperature': ssp_data['tas'],      # [time, lat, lon]
                'precipitation': ssp_data['pr'],     # [time, lat, lon]
                'gdp': ssp_data['gdp'],              # [time, lat, lon]
                'population': ssp_data['pop'],       # [time, lat, lon]
                'temperature_years': ssp_data['tas_years'],
                'precipitation_years': ssp_data['pr_years'],
                'gdp_years': ssp_data['gdp_years'],
                'population_years': ssp_data['pop_years']
            }
            
            # Store metadata from first SSP (coordinates same for all)
            if i == 0:
                all_data['_metadata'].update({
                    'lat': ssp_data['lat'],
                    'lon': ssp_data['lon'],
                    'grid_shape': (len(ssp_data['lat']), len(ssp_data['lon'])),
                    'time_shape': len(ssp_data['tas_years']),  # Assuming all same length
                    'years': ssp_data['common_years']  # Common years array computed once
                })
            
            print(f"  Data shape: {ssp_data['tas'].shape}")
            print(f"  ✅ Successfully loaded {ssp_name}")
            
        except Exception as e:
            print(f"  ❌ Failed to load {ssp_name}: {e}")
            raise RuntimeError(f"Could not load data for {ssp_name}: {e}")
    
    # Summary information
    nlat, nlon = all_data['_metadata']['grid_shape']
    ntime = all_data['_metadata']['time_shape']
    total_grid_cells = nlat * nlon
    
    print(f"\n📊 Data Loading Summary:")
    print(f"  Grid dimensions: {nlat} × {nlon} = {total_grid_cells} cells")
    print(f"  Time dimension: {ntime} years")
    print(f"  SSP scenarios loaded: {len(all_ssps)}")
    print(f"  Total data arrays: {len(all_ssps) * 4} (4 variables × {len(all_ssps)} SSPs)")
    
    # Estimate memory usage
    bytes_per_array = nlat * nlon * ntime * 8  # 8 bytes per float64
    total_arrays = len(all_ssps) * 4  # 4 data types per SSP
    total_bytes = bytes_per_array * total_arrays
    total_mb = total_bytes / (1024 * 1024)
    
    print(f"  Estimated memory usage: {total_mb:.1f} MB")
    print("  ✅ All NetCDF data loaded successfully")

    # Create global valid mask (check all time points for economic activity)
    print("Computing global valid grid cell mask...")

    # Use reference SSP for validity checking
    ref_gdp = all_data[reference_ssp]['gdp']  # [time, lat, lon] (not [lat, lon, time])
    ref_pop = all_data[reference_ssp]['population']  # [time, lat, lon]

    # Climate model data convention: [time, lat, lon]
    ntime_actual, nlat_actual, nlon_actual = ref_gdp.shape
    print(f"  Actual data dimensions: {ntime_actual} time × {nlat_actual} lat × {nlon_actual} lon")

    valid_mask = np.zeros((nlat_actual, nlon_actual), dtype=bool)

    # Diagnostic counters
    total_cells = nlat_actual * nlon_actual
    final_valid_count = 0

    for lat_idx in range(nlat_actual):
        for lon_idx in range(nlon_actual):
            gdp_timeseries = ref_gdp[:, lat_idx, lon_idx]  # [time] - extract all time points for this location
            pop_timeseries = ref_pop[:, lat_idx, lon_idx]  # [time]

            # Check if GDP and population are positive at ALL time points
            if np.all(gdp_timeseries > 0) and np.all(pop_timeseries > 0):
                valid_mask[lat_idx, lon_idx] = True
                final_valid_count += 1

    print(f"  Grid cell validation results:")
    print(f"    Total cells: {total_cells}")
    print(f"    Valid economic grid cells (non-zero GDP and population for all years): {final_valid_count} / {total_cells} ({100*final_valid_count/total_cells:.1f}%)")

    # Update metadata with correct dimensions
    all_data['_metadata']['grid_shape'] = (nlat_actual, nlon_actual)
    all_data['_metadata']['time_shape'] = ntime_actual

    # Add valid mask to metadata
    all_data['_metadata']['valid_mask'] = valid_mask
    all_data['_metadata']['valid_count'] = final_valid_count

    # Write NetCDF file with all loaded data if output directory is provided
    if output_dir is not None:
        write_all_loaded_data_netcdf(all_data, config, output_dir)

    return all_data


def write_all_loaded_data_netcdf(all_data: Dict[str, Any], config: Dict[str, Any], output_dir: str) -> str:
    """
    Write all loaded NetCDF data to a single output file for reference and validation.

    Parameters
    ----------
    all_data : Dict[str, Any]
        All loaded NetCDF data from load_all_netcdf_data()
    config : Dict[str, Any]
        Configuration dictionary
    output_dir : str
        Output directory path

    Returns
    -------
    str
        Path to written NetCDF file
    """
    import xarray as xr
    import numpy as np
    import os
    from datetime import datetime

    # Extract metadata
    metadata = all_data['_metadata']
    lat = metadata['lat']
    lon = metadata['lon']
    years = metadata['years']
    valid_mask = metadata['valid_mask']
    model_name = config['climate_model']['model_name']

    # Get SSP list (excluding metadata)
    ssp_names = [key for key in all_data.keys() if key != '_metadata']

    print(f"Writing all loaded data to NetCDF file...")

    # Extract configuration values for standardized naming
    json_id = config['run_metadata']['json_id']
    model_name = config['climate_model']['model_name']

    # Create filename using standardized pattern
    netcdf_filename = f"all_loaded_data_{json_id}_{model_name}.nc"
    netcdf_path = os.path.join(output_dir, netcdf_filename)

    # Prepare data arrays for xarray (all SSPs combined)
    n_ssp = len(ssp_names)
    n_time, n_lat, n_lon = len(years), len(lat), len(lon)

    # Initialize arrays [ssp, time, lat, lon]
    temperature_all = np.full((n_ssp, n_time, n_lat, n_lon), np.nan)
    precipitation_all = np.full((n_ssp, n_time, n_lat, n_lon), np.nan)
    gdp_all = np.full((n_ssp, n_time, n_lat, n_lon), np.nan)
    population_all = np.full((n_ssp, n_time, n_lat, n_lon), np.nan)

    # Fill arrays
    for i, ssp_name in enumerate(ssp_names):
        ssp_data = all_data[ssp_name]
        temperature_all[i] = ssp_data['temperature']      # [time, lat, lon]
        precipitation_all[i] = ssp_data['precipitation']  # [time, lat, lon]
        gdp_all[i] = ssp_data['gdp']                      # [time, lat, lon]
        population_all[i] = ssp_data['population']        # [time, lat, lon]

    # Create xarray Dataset
    ds = xr.Dataset(
        {
            'temperature': (['ssp', 'time', 'lat', 'lon'], temperature_all),
            'precipitation': (['ssp', 'time', 'lat', 'lon'], precipitation_all),
            'gdp': (['ssp', 'time', 'lat', 'lon'], gdp_all),
            'population': (['ssp', 'time', 'lat', 'lon'], population_all),
            'valid_mask': (['lat', 'lon'], valid_mask)
        },
        coords={
            'ssp': ssp_names,
            'time': years,
            'lat': lat,
            'lon': lon
        }
    )

    # Add variable attributes
    ds.temperature.attrs = {
        'long_name': 'Surface Air Temperature',
        'units': 'degrees_celsius',
        'description': 'Annual surface air temperature, converted from Kelvin'
    }

    ds.precipitation.attrs = {
        'long_name': 'Precipitation Rate',
        'units': 'mm/day',
        'description': 'Annual precipitation rate'
    }

    ds.gdp.attrs = {
        'long_name': 'GDP Density',
        'units': 'economic_units',
        'description': 'GDP density with exponential growth applied before prediction year'
    }

    ds.population.attrs = {
        'long_name': 'Population Density',
        'units': 'people',
        'description': 'Population density with exponential growth applied before prediction year'
    }

    ds.valid_mask.attrs = {
        'long_name': 'Valid Economic Grid Cells',
        'units': 'boolean',
        'description': 'True for grid cells with positive GDP and population for all years'
    }

    # Add global attributes
    import json
    serializable_config = create_serializable_config(config)
    ds.attrs = {
        'title': 'All Loaded NetCDF Data for COIN-SSP Processing',
        'description': f'Combined dataset from {model_name} containing all SSP scenarios with harmonized temporal alignment and exponential growth applied',
        'climate_model': model_name,
        'creation_date': datetime.now().isoformat(),
        'institution': 'COIN-SSP Climate Economics Model',
        'ssp_scenarios': ', '.join(ssp_names),
        'time_range': f'{years[0]}-{years[-1]}',
        'grid_shape': f'{n_lat}x{n_lon}',
        'valid_grid_cells': f'{metadata["valid_count"]}/{n_lat*n_lon} ({100*metadata["valid_count"]/(n_lat*n_lon):.1f}%)',
        'prediction_year': config['time_periods']['prediction_period']['start_year'],
        'exponential_growth_applied': 'GDP and population modified to exponential growth before prediction year',
        'configuration_json': json.dumps(serializable_config, indent=2)
    }

    # Write to NetCDF
    ds.to_netcdf(netcdf_path)

    print(f"  ✅ All loaded data written to: {os.path.basename(netcdf_path)}")
    print(f"     File size: {os.path.getsize(netcdf_path) / (1024*1024):.1f} MB")
    print(f"     SSP scenarios: {len(ssp_names)}")
    print(f"     Grid cells: {n_lat} × {n_lon} = {n_lat*n_lon}")
    print(f"     Valid cells: {metadata['valid_count']} ({100*metadata['valid_count']/(n_lat*n_lon):.1f}%)")
    print(f"     Time span: {years[0]}-{years[-1]} ({len(years)} years)")

    return netcdf_path


def resolve_netcdf_filepath(config: Dict[str, Any], data_type: str, ssp_name: str) -> str:
    """
    Resolve NetCDF file path using new configuration structure.

    NOTE: This function is deprecated in favor of specialized loading functions.
    It's maintained for backward compatibility and file path display purposes.

    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary
    data_type : str
        Type of data ('temperature', 'precipitation', 'gdp', 'population', 'target_reductions')
    ssp_name : str
        SSP scenario name (e.g., 'ssp245', 'ssp585')

    Returns
    -------
    str
        Full path to NetCDF file (returns first file for concatenated data types)
    """
    climate_model = config['climate_model']
    model_name = climate_model['model_name']
    input_dir = climate_model['input_directory']

    # Map old data_type names to new configuration structure
    if data_type == 'temperature':
        prefix = climate_model['file_prefixes']['tas_file_prefix']
        filename = f"{prefix}_{model_name}_historical.nc"  # Return historical file
    elif data_type == 'precipitation':
        prefix = climate_model['file_prefixes']['pr_file_prefix']
        filename = f"{prefix}_{model_name}_historical.nc"  # Return historical file
    elif data_type == 'gdp':
        prefix = climate_model['file_prefixes']['gdp_file_prefix']
        # Truncate SSP name for GDP files
        if ssp_name.startswith('ssp') and len(ssp_name) >= 4:
            short_ssp = ssp_name[:4]
        else:
            short_ssp = ssp_name
        filename = f"{prefix}_{model_name}_{short_ssp}.nc"
    elif data_type == 'population':
        prefix = climate_model['file_prefixes']['pop_file_prefix']
        filename = f"{prefix}_{model_name}_hist.nc"  # Return historical file
    elif data_type == 'target_reductions':
        prefix = climate_model['file_prefixes']['target_reductions_file_prefix']
        filename = f"{prefix}_{model_name}_{ssp_name}.nc"
    else:
        raise ValueError(f"Unknown data_type: {data_type}")

    filepath = os.path.join(input_dir, filename)
    return filepath


def get_ssp_data(all_data: Dict[str, Any], ssp_name: str, data_type: str) -> np.ndarray:
    """
    Extract specific data array from loaded NetCDF data structure.
    
    Parameters
    ----------
    all_data : Dict[str, Any]
        Result from load_all_netcdf_data()
    ssp_name : str
        SSP scenario name (e.g., 'ssp245')
    data_type : str
        Data type ('temperature', 'precipitation', 'gdp', 'population')
        
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


def get_grid_metadata(all_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract grid metadata from loaded NetCDF data structure.
    
    Parameters
    ----------
    all_data : Dict[str, Any]
        Result from load_all_netcdf_data()
        
    Returns
    -------
    Dict[str, Any]
        Metadata dictionary containing coordinates and dimensions
    """
    return all_data['_metadata']


# =============================================================================
# TFP Calculation Functions
# =============================================================================

def calculate_tfp_coin_ssp(pop, gdp, params):
    """
    Calculate total factor productivity time series using the Solow-Swan growth model.

    Parameters
    ----------
    pop : array-like
        Time series of population (L) in people
    gdp : array-like
        Time series of gross domestic product (Y) in $/yr
    params : dict or ModelParams
        Model parameters containing:
        - 's': savings rate (dimensionless)
        - 'alpha': elasticity of output with respect to capital (dimensionless)
        - 'delta': depreciation rate in 1/yr
        
    Returns
    -------
    a : numpy.ndarray
        Total factor productivity time series, normalized to year 0 (A(t)/A(0))
    k : numpy.ndarray
        Capital stock time series, normalized to year 0 (K(t)/K(0))
        
    Notes
    -----
    Assumes system is in steady-state at year 0 with normalized values of 1.
    Uses discrete time integration with 1-year time steps.
    """
    y = gdp/gdp[0] # output normalized to year 0
    l = pop/pop[0] # population normalized to year 0
    k = np.copy(y) # capital stock normalized to year 0
    a = np.copy(y) # total factor productivity normalized to year 0
    s = params.s # savings rate
    alpha = params.alpha # elasticity of output with respect to capital
    delta = params.delta # depreciation rate in units of 1/yr

    # Let's assume that at year 0, the system is in steady-state, do d k / dt = 0 at year 0, and a[0] = 1.
    # 0 == s * y[0] - delta * k[0]
    k[0] = (s/delta) # everything is non0dimensionalized to 1 at year 0
    # y[0] ==  a[0] * k[0]**alpha * l[0]**(1-alpha)

    a[0] = k[0]**(-alpha) # nondimensionalized Total Factor Productivity is 0 in year 0

    # since we are assuming steady state, the capital stock will be the same at the start of year 1

    for t in range(len(y)-1):
        # I want y(t+1) ==  a(t+1) * k(t+1)**alpha * l(t)**(1-alpha)
        #
        # so this means that a(t+1) = y(t + 1) / (k(t+1)**alpha * l(t+1)**(1-alpha))

        dkdt = s * y[t] - delta *k[t]
        k[t+1] = k[t] + dkdt  # assumed time step is one year

        a[t+1] = y[t+1] / (k[t+1]**alpha * l[t+1]**(1-alpha))

    return a, k


# =============================================================================
# Output Writing Functions
# =============================================================================

def save_step1_results_netcdf(target_results: Dict[str, Any], output_path: str, config: Dict[str, Any]) -> str:
    """
    Save Step 1 target GDP reduction results to NetCDF file.

    Parameters
    ----------
    target_results : Dict[str, Any]
        Results from step1_calculate_target_gdp_changes()
    output_path : str
        Complete output file path
    config : Dict[str, Any]
        Full configuration dictionary to embed in NetCDF file

    Returns
    -------
    str
        Path to saved NetCDF file
    """
    import xarray as xr
    import os
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Extract metadata
    metadata = target_results['_metadata']
    
    # Create arrays for each target type
    target_arrays = []
    target_names = []
    
    for target_name, target_data in target_results.items():
        if target_name != '_metadata':
            target_arrays.append(target_data['reduction_array'])
            target_names.append(target_name)
    
    # Stack arrays: (target_name, lat, lon)
    target_reductions = np.stack(target_arrays, axis=0)

    # Create xarray dataset
    ds = xr.Dataset(
        {
            'target_gdp_amounts': (['target_name', 'lat', 'lon'], target_reductions),
            'temperature_ref': (['lat', 'lon'], metadata['temp_ref']),
            'gdp_target': (['lat', 'lon'], metadata['gdp_target'])
        },
        coords={
            'target_name': target_names,
            'lat': metadata['lat'],
            'lon': metadata['lon']
        }
    )
    
    # Add attributes
    ds.target_gdp_amounts.attrs = {
        'long_name': 'Target GDP reductions',
        'units': 'fractional reduction',
        'description': f'Target reduction patterns for {len(target_names)} cases'
    }
    
    ds.temperature_ref.attrs = {
        'long_name': 'Reference period temperature',
        'units': '°C'
    }
    
    ds.gdp_target.attrs = {
        'long_name': 'Target period GDP',
        'units': 'economic units'
    }
    
    # Add global attributes
    import json
    serializable_config = create_serializable_config(config)
    ds.attrs = {
        'title': 'COIN-SSP Target GDP Reductions - Step 1 Results',
        'reference_ssp': metadata['reference_ssp'],
        'reference_period': f"{metadata['time_periods']['reference_period']['start_year']}-{metadata['time_periods']['reference_period']['end_year']}",
        'target_period': f"{metadata['time_periods']['target_period']['start_year']}-{metadata['time_periods']['target_period']['end_year']}",
        'global_temp_ref': metadata['global_temp_ref'],
        'global_gdp_target': metadata['global_gdp_target'],
        'creation_date': datetime.now().isoformat(),
        'configuration_json': json.dumps(serializable_config, indent=2)
    }
    
    # Save to file
    ds.to_netcdf(output_path)
    print(f"Step 1 results saved to {output_path}")
    return output_path


def save_step2_results_netcdf(tfp_results: Dict[str, Any], output_path: str, model_name: str, config: Dict[str, Any]) -> str:
    """
    Save Step 2 baseline TFP results to NetCDF file.
    
    Parameters
    ----------
    tfp_results : Dict[str, Any]
        Results from step2_calculate_baseline_tfp()
    output_path : str
        Complete output file path
    model_name : str
        Climate model name
    config : Dict[str, Any]
        Full configuration dictionary to embed in NetCDF file
        
    Returns
    -------
    str
        Path to saved NetCDF file
    """
    import xarray as xr
    import os
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Get SSP names and dimensions from first result
    ssp_names = [ssp for ssp in tfp_results.keys() if ssp not in ['_coordinates', '_metadata']]
    first_ssp = tfp_results[ssp_names[0]]
    ntime, nlat, nlon = first_ssp['tfp_baseline'].shape  # [time, lat, lon] convention
    
    # Create coordinate arrays that match the actual data dimensions
    lat_coords = np.arange(nlat)  # Use actual latitude dimension
    lon_coords = np.arange(nlon)  # Use actual longitude dimension
    
    # Create arrays for all SSPs [ssp, time, lat, lon]
    tfp_all_ssps = np.full((len(ssp_names), ntime, nlat, nlon), np.nan)
    k_all_ssps = np.full((len(ssp_names), ntime, nlat, nlon), np.nan)
    valid_masks = np.full((len(ssp_names), nlat, nlon), False)

    for i, ssp_name in enumerate(ssp_names):
        tfp_all_ssps[i] = tfp_results[ssp_name]['tfp_baseline']  # [time, lat, lon]
        k_all_ssps[i] = tfp_results[ssp_name]['k_baseline']      # [time, lat, lon]
        valid_masks[i] = tfp_results[ssp_name]['valid_mask']
    
    # Create coordinate arrays (assuming annual time steps starting from year 0)
    time_coords = np.arange(ntime)
    
    # Create xarray dataset with [time, lat, lon] convention
    ds = xr.Dataset(
        {
            'tfp_baseline': (['ssp', 'time', 'lat', 'lon'], tfp_all_ssps),
            'k_baseline': (['ssp', 'time', 'lat', 'lon'], k_all_ssps),
            'valid_mask': (['ssp', 'lat', 'lon'], valid_masks)
        },
        coords={
            'ssp': ssp_names,
            'time': time_coords,
            'lat': lat_coords,
            'lon': lon_coords
        }
    )
    
    # Add attributes
    ds.tfp_baseline.attrs = {
        'long_name': 'Baseline Total Factor Productivity',
        'units': 'normalized to year 0',
        'description': 'TFP time series without climate effects, calculated using Solow-Swan model'
    }
    
    ds.k_baseline.attrs = {
        'long_name': 'Baseline Capital Stock',
        'units': 'normalized to year 0', 
        'description': 'Capital stock time series without climate effects'
    }
    
    ds.valid_mask.attrs = {
        'long_name': 'Valid economic grid cells',
        'units': 'boolean',
        'description': 'True for grid cells with economic activity (GDP > 0 and population > 0)'
    }
    
    # Add global attributes
    import json
    serializable_config = create_serializable_config(config)
    total_processed = sum(result['grid_cells_processed'] for ssp_name, result in tfp_results.items()
                         if ssp_name not in ['_coordinates', '_metadata'])
    ds.attrs = {
        'title': 'COIN-SSP Baseline TFP - Step 2 Results',
        'model_name': model_name,
        'total_grid_cells_processed': total_processed,
        'ssp_scenarios': ', '.join(ssp_names),
        'description': 'Baseline Total Factor Productivity calculated for each SSP scenario without climate effects',
        'creation_date': datetime.now().isoformat(),
        'configuration_json': json.dumps(serializable_config, indent=2)
    }
    
    # Save to file
    ds.to_netcdf(output_path)
    print(f"Step 2 results saved to {output_path}")
    return output_path


def save_step3_results_netcdf(scaling_results: Dict[str, Any], output_path: str, model_name: str, config: Dict[str, Any]) -> str:
    """
    Save Step 3 scaling factor results to NetCDF file.
    
    Parameters
    ----------
    scaling_results : Dict[str, Any]
        Results from step3_calculate_scaling_factors_per_cell()
    output_path : str
        Complete output file path
    model_name : str
        Climate model name
    config : Dict[str, Any]
        Full configuration dictionary to embed in NetCDF file
        
    Returns
    -------
    str
        Path to saved NetCDF file
    """
    import xarray as xr
    import os
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Extract data arrays and metadata
    scaling_factors = scaling_results['scaling_factors']  # [lat, lon, response_func, target]
    optimization_errors = scaling_results['optimization_errors']  # [lat, lon, response_func, target]
    convergence_flags = scaling_results['convergence_flags']  # [lat, lon, response_func, target]
    scaled_parameters = scaling_results['scaled_parameters']  # [lat, lon, response_func, target, param]
    valid_mask = scaling_results['valid_mask']  # [lat, lon]
    
    # Get dimensions and coordinate info
    nlat, nlon, n_response_func, n_target = scaling_factors.shape
    n_scaled_params = scaled_parameters.shape[4]
    response_function_names = scaling_results['response_function_names']
    target_names = scaling_results['target_names']
    scaled_param_names = scaling_results['scaled_param_names']
    coordinates = scaling_results['_coordinates']
    
    # Transpose arrays to put lat,lon last: [lat, lon, response_func, target] -> [response_func, target, lat, lon]
    scaling_factors_t = scaling_factors.transpose(2, 3, 0, 1)  # [response_func, target, lat, lon]
    optimization_errors_t = optimization_errors.transpose(2, 3, 0, 1)  # [response_func, target, lat, lon]
    convergence_flags_t = convergence_flags.transpose(2, 3, 0, 1)  # [response_func, target, lat, lon]
    scaled_parameters_t = scaled_parameters.transpose(2, 3, 4, 0, 1)  # [response_func, target, param, lat, lon]

    # Create xarray dataset with lat,lon as last dimensions
    ds = xr.Dataset(
        {
            'scaling_factors': (['response_func', 'target', 'lat', 'lon'], scaling_factors_t),
            'optimization_errors': (['response_func', 'target', 'lat', 'lon'], optimization_errors_t),
            'convergence_flags': (['response_func', 'target', 'lat', 'lon'], convergence_flags_t),
            'scaled_parameters': (['response_func', 'target', 'param', 'lat', 'lon'], scaled_parameters_t),
            'valid_mask': (['lat', 'lon'], valid_mask)
        },
        coords={
            'response_func': response_function_names,
            'target': target_names,
            'param': scaled_param_names,
            'lat': coordinates['lat'],
            'lon': coordinates['lon']
        }
    )
    
    # Add attributes
    ds.scaling_factors.attrs = {
        'long_name': 'Climate response scaling factors',
        'units': 'dimensionless',
        'description': 'Optimized scaling factors for climate damage functions per grid cell'
    }
    
    ds.optimization_errors.attrs = {
        'long_name': 'Optimization error values',
        'units': 'dimensionless', 
        'description': 'Final objective function values from scaling factor optimization'
    }
    
    ds.convergence_flags.attrs = {
        'long_name': 'Optimization convergence flags',
        'units': 'boolean',
        'description': 'True where optimization converged successfully'
    }
    
    ds.scaled_parameters.attrs = {
        'long_name': 'Scaled climate damage function parameters',
        'units': 'various',
        'description': 'Climate damage function parameters (scaling_factor × base_parameter) for each grid cell and combination',
        'parameter_names': ', '.join(scaled_param_names),
        'parameter_groups': 'Capital: k_tas1,k_tas2,k_pr1,k_pr2; TFP: tfp_tas1,tfp_tas2,tfp_pr1,tfp_pr2; Output: y_tas1,y_tas2,y_pr1,y_pr2',
        'climate_variables': 'tas=temperature, pr=precipitation; 1=linear, 2=quadratic'
    }
    
    ds.valid_mask.attrs = {
        'long_name': 'Valid economic grid cells',
        'units': 'boolean',
        'description': 'True for grid cells with economic activity used in optimization'
    }
    
    # Add global attributes
    import json
    serializable_config = create_serializable_config(config)
    ds.attrs = {
        'title': 'COIN-SSP Scaling Factors - Step 3 Results',
        'model_name': model_name,
        'reference_ssp': scaling_results['reference_ssp'],
        'total_grid_cells': scaling_results['total_grid_cells'],
        'successful_optimizations': scaling_results['successful_optimizations'],
        'success_rate_percent': 100 * scaling_results['successful_optimizations'] / max(1, scaling_results['total_grid_cells']),
        'response_functions': ', '.join(response_function_names),
        'target_patterns': ', '.join(target_names),
        'description': 'Per-grid-cell scaling factors optimized using reference SSP scenario',
        'creation_date': datetime.now().isoformat(),
        'configuration_json': json.dumps(serializable_config, indent=2)
    }
    
    # Save to file
    ds.to_netcdf(output_path)
    print(f"Step 3 results saved to {output_path}")
    return output_path


def save_step4_results_netcdf_split(step4_results: Dict[str, Any], output_dir: str, model_name: str, config: Dict[str, Any]) -> List[str]:
    """
    Save Step 4 forward model results to separate NetCDF files per SSP/variable combination.

    Creates 15 files (5 SSPs × 3 variables), each containing climate and weather variants
    of the variable with coordinate ordering: (target, response_func, time, lat, lon).

    Parameters
    ----------
    step4_results : Dict[str, Any]
        Results from step4_forward_integration_all_ssps()
    output_dir : str
        Output directory path
    model_name : str
        Climate model name
    config : Dict[str, Any]
        Full configuration dictionary to embed in NetCDF file

    Returns
    -------
    List[str]
        List of paths to saved NetCDF files
    """
    import xarray as xr
    import os

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Extract metadata and structure
    forward_results = step4_results['forward_results']
    response_function_names = step4_results['response_function_names']
    target_names = step4_results['target_names']
    valid_mask = step4_results['valid_mask']
    coordinates = step4_results['_coordinates']

    # Get SSP names and dimensions from first SSP result
    ssp_names = list(forward_results.keys())
    first_ssp = forward_results[ssp_names[0]]
    nlat, nlon, n_response_func, n_target, ntime = first_ssp['gdp_climate'].shape

    # Variables to process (climate and weather variants paired)
    variable_pairs = [
        ('gdp_climate', 'gdp_weather', 'gdp'),
        ('tfp_climate', 'tfp_weather', 'tfp'),
        ('k_climate', 'k_weather', 'capital')
    ]

    saved_files = []

    # Create separate file for each SSP/variable combination
    for ssp_name in ssp_names:
        ssp_result = forward_results[ssp_name]

        for climate_var, weather_var, var_base in variable_pairs:
            # Reorder coordinates from [lat, lon, response_func, target, time]
            # to [target, response_func, time, lat, lon]
            climate_data = ssp_result[climate_var].transpose(3, 2, 4, 0, 1)  # [target, response_func, time, lat, lon]
            weather_data = ssp_result[weather_var].transpose(3, 2, 4, 0, 1)  # [target, response_func, time, lat, lon]

            # Create dataset for this SSP/variable combination
            ds = xr.Dataset(
                {
                    f'{var_base}_climate': (['target', 'response_func', 'time', 'lat', 'lon'], climate_data),
                    f'{var_base}_weather': (['target', 'response_func', 'time', 'lat', 'lon'], weather_data),
                    'valid_mask': (['lat', 'lon'], valid_mask)
                },
                coords={
                    'target': target_names,
                    'response_func': response_function_names,
                    'time': coordinates['years'],  # Use actual years
                    'lat': coordinates['lat'],
                    'lon': coordinates['lon']
                }
            )

            # Add comprehensive attributes
            import json
            serializable_config = create_serializable_config(config)
            ds.attrs.update({
                'title': f'COIN-SSP Step 4 Forward Model Results - {ssp_name.upper()} - {var_base.upper()}',
                'description': f'Forward economic modeling results for {ssp_name.upper()} scenario, {var_base} variables',
                'model': model_name,
                'ssp_scenario': ssp_name,
                'variable_type': var_base,
                'coordinate_order': 'target, response_func, time, lat, lon',
                'variables_included': f'{var_base}_climate, {var_base}_weather, valid_mask',
                'creation_date': pd.Timestamp.now().isoformat(),
                'contact': 'Generated by COIN-SSP pipeline',
                'configuration_json': json.dumps(serializable_config, indent=2)
            })

            # Variable-specific attributes
            ds[f'{var_base}_climate'].attrs = {
                'long_name': f'{var_base.upper()} with climate effects',
                'description': f'{var_base.upper()} projections including both weather variability and climate trends',
                'units': 'model units'
            }

            ds[f'{var_base}_weather'].attrs = {
                'long_name': f'{var_base.upper()} with weather only',
                'description': f'{var_base.upper()} projections with weather variability but climate trends removed',
                'units': 'model units'
            }

            ds.valid_mask.attrs = {
                'long_name': 'Valid economic grid cell mask',
                'description': 'Boolean mask indicating grid cells with valid economic data (positive GDP and population for all years)',
                'note': 'Cells with non-positive GDP or population values in any year excluded'
            }

            # Coordinate attributes
            ds.target.attrs = {'long_name': 'GDP reduction target scenarios'}
            ds.response_func.attrs = {'long_name': 'Climate response function names'}
            ds.time.attrs = {'long_name': 'Year', 'units': 'years'}
            ds.lat.attrs = {'long_name': 'Latitude', 'units': 'degrees_north'}
            ds.lon.attrs = {'long_name': 'Longitude', 'units': 'degrees_east'}

            # Generate filename and save
            filename = f"step4_forward_{ssp_name}_{var_base}_{model_name}.nc"
            filepath = os.path.join(output_dir, filename)

            # Save with compression
            encoding = {
                f'{var_base}_climate': {'zlib': True, 'complevel': 4},
                f'{var_base}_weather': {'zlib': True, 'complevel': 4},
                'valid_mask': {'zlib': True, 'complevel': 4}
            }

            ds.to_netcdf(filepath, encoding=encoding)
            saved_files.append(filepath)

            print(f"  Saved {ssp_name.upper()} {var_base} data: {filename}")

    print(f"Step 4 results saved to {len(saved_files)} separate NetCDF files")
    return saved_files


def save_step4_results_netcdf_legacy(step4_results: Dict[str, Any], output_path: str, model_name: str, config: Dict[str, Any]) -> str:
    """
    Save Step 4 forward model results to NetCDF file.
    
    Parameters
    ----------
    step4_results : Dict[str, Any]
        Results from step4_forward_integration_all_ssps()
    output_path : str
        Complete output file path
    model_name : str
        Climate model name
    config : Dict[str, Any]
        Full configuration dictionary to embed in NetCDF file
        
    Returns
    -------
    str
        Path to saved NetCDF file
    """
    import xarray as xr
    import os
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Extract metadata and structure
    forward_results = step4_results['forward_results']
    response_function_names = step4_results['response_function_names']
    target_names = step4_results['target_names']
    valid_mask = step4_results['valid_mask']
    coordinates = step4_results['_coordinates']
    
    # Get SSP names and dimensions from first SSP result
    ssp_names = list(forward_results.keys())
    first_ssp = forward_results[ssp_names[0]]
    nlat, nlon, n_response_func, n_target, ntime = first_ssp['gdp_climate'].shape
    
    # Create arrays for all SSPs: [ssp, response_func, target, time, lat, lon]
    n_ssps = len(ssp_names)
    gdp_climate_all = np.full((n_ssps, n_response_func, n_target, ntime, nlat, nlon), np.nan)
    gdp_weather_all = np.full((n_ssps, n_response_func, n_target, ntime, nlat, nlon), np.nan)
    tfp_climate_all = np.full((n_ssps, n_response_func, n_target, ntime, nlat, nlon), np.nan)
    tfp_weather_all = np.full((n_ssps, n_response_func, n_target, ntime, nlat, nlon), np.nan)
    k_climate_all = np.full((n_ssps, n_response_func, n_target, ntime, nlat, nlon), np.nan)
    k_weather_all = np.full((n_ssps, n_response_func, n_target, ntime, nlat, nlon), np.nan)

    # Stack results from all SSPs and transpose to put time,lat,lon last
    for i, ssp_name in enumerate(ssp_names):
        ssp_result = forward_results[ssp_name]
        # Input shape: [lat, lon, response_func, target, time] -> [response_func, target, time, lat, lon]
        gdp_climate_all[i] = ssp_result['gdp_climate'].transpose(2, 3, 4, 0, 1)
        gdp_weather_all[i] = ssp_result['gdp_weather'].transpose(2, 3, 4, 0, 1)
        tfp_climate_all[i] = ssp_result['tfp_climate'].transpose(2, 3, 4, 0, 1)
        tfp_weather_all[i] = ssp_result['tfp_weather'].transpose(2, 3, 4, 0, 1)
        k_climate_all[i] = ssp_result['k_climate'].transpose(2, 3, 4, 0, 1)
        k_weather_all[i] = ssp_result['k_weather'].transpose(2, 3, 4, 0, 1)
    
    # Create time coordinate using actual data years
    time_coords = np.arange(ntime)
    
    # Create xarray dataset with time,lat,lon as last dimensions
    ds = xr.Dataset(
        {
            'gdp_climate': (['ssp', 'response_func', 'target', 'time', 'lat', 'lon'], gdp_climate_all),
            'gdp_weather': (['ssp', 'response_func', 'target', 'time', 'lat', 'lon'], gdp_weather_all),
            'tfp_climate': (['ssp', 'response_func', 'target', 'time', 'lat', 'lon'], tfp_climate_all),
            'tfp_weather': (['ssp', 'response_func', 'target', 'time', 'lat', 'lon'], tfp_weather_all),
            'k_climate': (['ssp', 'response_func', 'target', 'time', 'lat', 'lon'], k_climate_all),
            'k_weather': (['ssp', 'response_func', 'target', 'time', 'lat', 'lon'], k_weather_all),
            'valid_mask': (['lat', 'lon'], valid_mask)
        },
        coords={
            'ssp': ssp_names,
            'response_func': response_function_names,
            'target': target_names,
            'time': time_coords,
            'lat': coordinates['lat'],
            'lon': coordinates['lon']
        }
    )
    
    # Add attributes
    ds.gdp_climate.attrs = {
        'long_name': 'GDP with climate effects',
        'units': 'economic units per year',
        'description': 'GDP projections including full climate change effects'
    }
    
    ds.gdp_weather.attrs = {
        'long_name': 'GDP with weather effects only',
        'units': 'economic units per year',
        'description': 'GDP projections with weather variability but no climate trends'
    }
    
    ds.tfp_climate.attrs = {
        'long_name': 'Total Factor Productivity with climate effects', 
        'units': 'normalized to year 0',
        'description': 'TFP projections including full climate change effects'
    }
    
    ds.tfp_weather.attrs = {
        'long_name': 'Total Factor Productivity with weather effects only',
        'units': 'normalized to year 0',
        'description': 'TFP projections with weather variability but no climate trends'
    }
    
    ds.k_climate.attrs = {
        'long_name': 'Capital stock with climate effects',
        'units': 'normalized to year 0',
        'description': 'Capital stock projections including full climate change effects'
    }
    
    ds.k_weather.attrs = {
        'long_name': 'Capital stock with weather effects only', 
        'units': 'normalized to year 0',
        'description': 'Capital stock projections with weather variability but no climate trends'
    }
    
    ds.valid_mask.attrs = {
        'long_name': 'Valid economic grid cells',
        'units': 'boolean',
        'description': 'True for grid cells with economic activity used in forward modeling'
    }
    
    # Add global attributes
    total_successful = sum(forward_results[ssp]['successful_forward_runs'] for ssp in ssp_names)
    total_runs = sum(forward_results[ssp]['total_forward_runs'] for ssp in ssp_names)
    
    import json
    serializable_config = create_serializable_config(config)
    ds.attrs = {
        'title': 'COIN-SSP Forward Model Results - Step 4 Results',
        'model_name': model_name,
        'total_ssps_processed': step4_results['total_ssps_processed'],
        'ssp_scenarios': ', '.join(ssp_names),
        'response_functions': ', '.join(response_function_names),
        'target_patterns': ', '.join(target_names),
        'total_forward_runs': total_runs,
        'successful_forward_runs': total_successful,
        'overall_success_rate_percent': 100 * total_successful / max(1, total_runs),
        'description': 'Climate-integrated economic projections using per-grid-cell scaling factors for all SSP scenarios',
        'creation_date': datetime.now().isoformat(),
        'configuration_json': json.dumps(serializable_config, indent=2)
    }
    
    # Save to file
    ds.to_netcdf(output_path)
    print(f"Step 4 results saved to {output_path}")
    return output_path


def create_target_gdp_visualization(target_results: Dict[str, Any], config: Dict[str, Any],
                                   output_dir: str, reference_ssp: str, valid_mask: np.ndarray) -> str:
    """
    Create comprehensive visualization of target GDP reduction results.

    Generates a single-page PDF with:
    - Global maps showing spatial patterns of each target reduction type
    - Line plot showing damage functions vs temperature (if coefficients available)

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
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.backends.backend_pdf import PdfPages
    import os

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
    temp_ref = metadata['temp_ref']
    gdp_target = metadata['gdp_target']
    global_temp_ref = metadata['global_temp_ref']
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
    gdp_weighted_temp_target = calculate_global_mean(gdp_target * temp_ref, lat, valid_mask) / global_gdp_target

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
                    f'GDP-weighted Mean Temperature ({target_period_start}-{target_period_end}): {gdp_weighted_temp_target:.2f}°C',
                    fontsize=16, fontweight='bold')

        # Create 3 map subplots (assuming we have 3 targets: constant, linear, quadratic)
        map_positions = [(2, 2, 1), (2, 2, 2), (2, 2, 3)]  # Top row + bottom left

        for i, target_name in enumerate(target_names[:3]):  # Limit to 3 maps
            reduction_array = reduction_arrays[target_name]
            global_mean = global_means[target_name]
            gdp_weighted_mean = gdp_weighted_means[target_name]
            data_range = data_ranges[target_name]
            target_info = target_results[target_name]

            # Create subplot
            ax = plt.subplot(*map_positions[i])

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

        # Fourth panel: Line plot (bottom right)
        ax4 = plt.subplot(2, 2, 4)

        # Temperature range for plotting
        temp_range = np.linspace(-10, 35, 1000)

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
                function_values = np.full_like(temp_range, constant_value)
                label = f'Constant: {constant_value:.3f}'

                # Horizontal line for constant
                ax4.plot(temp_range, function_values, color=color, linewidth=2,
                        label=label, alpha=0.8)

            elif target_shape == 'linear':
                coefficients = target_info['coefficients']
                if coefficients:
                    # Linear function: reduction = a0 + a1 * T
                    a0, a1 = coefficients['a0'], coefficients['a1']
                    function_values = a0 + a1 * temp_range

                    ax4.plot(temp_range, function_values, color=color, linewidth=2,
                            label=f'Linear: {a0:.4f} + {a1:.4f}×T', alpha=0.8)

                    # Add calibration point from config
                    gdp_targets = config['gdp_targets']
                    linear_config = next(t for t in gdp_targets if t['target_name'] == target_name)
                    if 'reference_temperature' in linear_config:
                        ref_temp = linear_config['reference_temperature']
                        ref_value = linear_config['amount_at_reference_temp']
                        ax4.plot(ref_temp, ref_value, 'o', color=color, markersize=8,
                                label=f'Linear calib: {ref_temp}°C = {ref_value:.3f}')

            elif target_shape == 'quadratic':
                coefficients = target_info['coefficients']
                if coefficients:
                    # Quadratic function: reduction = a + b*T + c*T²
                    a, b, c = coefficients['a'], coefficients['b'], coefficients['c']
                    function_values = a + b * temp_range + c * temp_range**2

                    ax4.plot(temp_range, function_values, color=color, linewidth=2,
                            label=f'Quadratic: {a:.4f} + {b:.4f}×T + {c:.6f}×T²', alpha=0.8)

                    # Add calibration points from config
                    gdp_targets = config['gdp_targets']
                    quad_config = next(t for t in gdp_targets if t['target_name'] == target_name)

                    # Handle new derivative-based specification
                    if 'derivative_at_zero_amount_temperature' in quad_config:
                        zero_temp = quad_config['zero_amount_temperature']
                        derivative = quad_config['derivative_at_zero_amount_temperature']
                        ax4.plot(zero_temp, 0, 's', color=color, markersize=8,
                                label=f'Quad zero: {zero_temp}°C = 0 (slope={derivative:.3f})')
                    # Handle legacy reference point specification
                    elif 'reference_temperature' in quad_config:
                        ref_temp = quad_config['reference_temperature']
                        ref_value = quad_config['amount_at_reference_temp']
                        ax4.plot(ref_temp, ref_value, 'o', color=color, markersize=8,
                                label=f'Quad calib: {ref_temp}°C = {ref_value:.3f}')

                        if 'zero_amount_temperature' in quad_config:
                            zero_temp = quad_config['zero_amount_temperature']
                            ax4.plot(zero_temp, 0, 's', color=color, markersize=8,
                                    label=f'Quad zero: {zero_temp}°C = 0')

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
                    values = a0 + a1 * temp_range
                elif target_shape == 'quadratic':
                    a, b, c = coefficients['a'], coefficients['b'], coefficients['c']
                    values = a + b * temp_range + c * temp_range**2
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


def create_scaling_factors_visualization(scaling_results, config, output_dir, model_name):
    """
    Create comprehensive PDF visualization for Step 3 scaling factor results.

    Generates a multi-panel visualization with one map per damage function × target combination.
    For typical case: 3 damage functions × 2 targets = 6 small maps on one page.

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
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.backends.backend_pdf import PdfPages
    import os

    # Generate output filename
    reference_ssp = config['ssp_scenarios']['reference_ssp']
    pdf_filename = f"step3_scaling_factors_visualization_{model_name}_{reference_ssp}.pdf"
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
    nlat, nlon, n_damage, n_targets = scaling_factors.shape
    total_maps = n_damage * n_targets

    # New layout: 3 maps per page, arranged vertically
    maps_per_page = 3
    total_pages = (total_maps + maps_per_page - 1) // maps_per_page  # Ceiling division

    print(f"Creating Step 3 visualization: {total_maps} maps across {total_pages} pages (3 maps per page)")

    # Create PDF with multi-page layout
    with PdfPages(pdf_path) as pdf:
        map_idx = 0
        page_num = 0

        for damage_idx, damage_name in enumerate(response_function_names):
            for target_idx, target_name in enumerate(target_names):

                # Start new page if needed (every 3 maps)
                if map_idx % maps_per_page == 0:
                    if map_idx > 0:
                        # Save previous page
                        plt.tight_layout()
                        plt.subplots_adjust(top=0.93, bottom=0.05)
                        pdf.savefig(fig, bbox_inches='tight')
                        plt.close(fig)

                    # Create new page
                    page_num += 1
                    fig = plt.figure(figsize=(12, 16))  # Taller figure for vertical arrangement
                    fig.suptitle(f'Step 3: Scaling Factors - {model_name} ({reference_ssp}) - Page {page_num}/{total_pages}',
                                fontsize=16, fontweight='bold', y=0.98)

                # Position on current page (1-3)
                subplot_idx = (map_idx % maps_per_page) + 1
                ax = plt.subplot(maps_per_page, 1, subplot_idx)  # 3 rows, 1 column

                # Extract scaling factor map for this combination
                sf_map = scaling_factors[:, :, damage_idx, target_idx]

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
                ax.set_title(f'{damage_name}\n{target_name}', fontsize=14, fontweight='bold')
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

                map_idx += 1

        # Save the final page
        if map_idx > 0:
            plt.tight_layout()
            plt.subplots_adjust(top=0.93, bottom=0.05)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    print(f"Scaling factors visualization saved to {pdf_path} ({total_pages} pages, 3 maps per page)")
    return pdf_path


def create_objective_function_visualization(scaling_results, config, output_dir, model_name):
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
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.backends.backend_pdf import PdfPages
    import os

    # Generate output filename
    reference_ssp = config['ssp_scenarios']['reference_ssp']
    pdf_filename = f"step3_objective_function_visualization_{model_name}_{reference_ssp}.pdf"
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
    nlat, nlon, n_response, n_targets = optimization_errors.shape
    total_maps = n_response * n_targets

    # New layout: 3 maps per page, arranged vertically
    maps_per_page = 3
    total_pages = (total_maps + maps_per_page - 1) // maps_per_page  # Ceiling division

    print(f"Creating Step 3 objective function visualization: {total_maps} maps across {total_pages} pages (3 maps per page)")

    # Create PDF with multi-page layout
    with PdfPages(pdf_path) as pdf:
        map_idx = 0
        page_num = 0

        for response_idx, response_name in enumerate(response_function_names):
            for target_idx, target_name in enumerate(target_names):

                # Start new page if needed (every 3 maps)
                if map_idx % maps_per_page == 0:
                    if map_idx > 0:
                        # Save previous page
                        plt.tight_layout()
                        plt.subplots_adjust(top=0.93, bottom=0.05)
                        pdf.savefig(fig, bbox_inches='tight')
                        plt.close(fig)

                    # Create new page
                    page_num += 1
                    fig = plt.figure(figsize=(12, 16))  # Taller figure for vertical arrangement
                    fig.suptitle(f'Step 3: Objective Function Values - {model_name} ({reference_ssp}) - Page {page_num}/{total_pages}',
                                fontsize=16, fontweight='bold', y=0.98)

                # Position on current page (1-3)
                subplot_idx = (map_idx % maps_per_page) + 1
                ax = plt.subplot(maps_per_page, 1, subplot_idx)  # 3 rows, 1 column

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

                map_idx += 1

        # Save the final page
        if map_idx > 0:
            plt.tight_layout()
            plt.subplots_adjust(top=0.93, bottom=0.05)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    print(f"Objective function visualization saved to {pdf_path} ({total_pages} pages, 3 maps per page)")
    return pdf_path


def create_baseline_tfp_visualization(tfp_results, config, output_dir, all_netcdf_data):
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
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.backends.backend_pdf import PdfPages
    import os

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
    years = metadata['years']

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

                    # Extract data from pre-loaded all_netcdf_data
                    # NOTE: all_netcdf_data has shape [time, lat, lon] same as TFP
                    ssp_data = all_netcdf_data[viz_ssp]
                    min_pop_series = ssp_data['population'][:, min_lat, min_lon]
                    max_pop_series = ssp_data['population'][:, max_lat, max_lon]
                    min_gdp_series = ssp_data['gdp'][:, min_lat, min_lon]
                    max_gdp_series = ssp_data['gdp'][:, max_lat, max_lon]

                    # Create DataFrame and save to CSV
                    import pandas as pd
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
