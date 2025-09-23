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
# Import all moved functions from refactored modules
from coin_ssp_netcdf import (
    write_all_loaded_data_netcdf, save_step4_results_netcdf_split,
    load_step3_results_from_netcdf, save_step1_results_netcdf,
    save_step2_results_netcdf, save_step3_results_netcdf,
    create_serializable_config, extract_year_coordinate, interpolate_to_annual_grid,
    resolve_netcdf_filepath
)
from coin_ssp_math_utils import (
    apply_time_series_filter, calculate_zero_biased_range, calculate_zero_biased_axis_range,
    calculate_area_weights, calculate_time_means, calculate_global_mean
)
from coin_ssp_target_calculations import (
    calculate_constant_target_reduction, calculate_linear_target_reduction,
    calculate_quadratic_target_reduction, calculate_all_target_reductions
)

def filter_scaling_params(scaling_config):
    allowed_keys = set(ScalingParams.__dataclass_fields__.keys())
    return {k: v for k, v in scaling_config.items() if k in allowed_keys}


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
        'tas' or 'pr'

    Returns
    -------
    tuple
        (concatenated_data, concatenated_years, valid_mask, coordinates)
    """

    climate_model = config['climate_model']
    input_dir = climate_model['input_directory']
    model_name = climate_model['model_name']

    # Get file prefix and variable name from new configuration structure
    if data_type == 'tas':
        prefix = climate_model['file_prefixes']['tas_file_prefix']
        var_name = climate_model['variable_names']['tas_var_name']
    elif data_type == 'pr':
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
    if data_type == 'tas':
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


def load_and_concatenate_pop_data(config, ssp_name):
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
        - 'years': common year range for all variables
    """

    # Extract configuration values
    model_name = config['climate_model']['model_name']
    prediction_year = config['time_periods']['prediction_period']['start_year']

    print(f"Loading and aligning NetCDF data for {model_name} {case_name}...")

    # Load temperature data (concatenate historical + SSP)
    print("  Loading temperature data...")
    tas_raw, tas_years_raw, lat, lon = load_and_concatenate_climate_data(config, case_name, 'tas')

    # Load precipitation data (concatenate historical + SSP)
    print("  Loading precipitation data...")
    pr_raw, pr_years_raw, _, _ = load_and_concatenate_climate_data(config, case_name, 'pr')

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
    pop_raw, pop_years_raw, _, _ = load_and_concatenate_pop_data(config, case_name)

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







# =============================================================================
# Target GDP Reduction Calculation Functions
# Extracted from calculate_target_gdp_amounts.py for reuse in integrated workflow
# =============================================================================









# =============================================================================
# Centralized NetCDF Data Loading Functions
# For efficient loading of all SSP scenario data upfront
# =============================================================================

def load_all_data(config: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
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
                'tas': np.array([time, lat, lon]),  # °C
                'pr': np.array([time, lat, lon]), # mm/day
                'gdp': np.array([time, lat, lon]),          # economic units
                'pop': np.array([time, lat, lon])    # people
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
            tas_file = resolve_netcdf_filepath(config, 'tas', ssp_name)
            pr_file = resolve_netcdf_filepath(config, 'pr', ssp_name)
            gdp_file = resolve_netcdf_filepath(config, 'gdp', ssp_name)
            pop_file = resolve_netcdf_filepath(config, 'pop', ssp_name)
            
            print(f"  Temperature: {os.path.basename(tas_file)}")
            print(f"  Precipitation: {os.path.basename(pr_file)}")
            print(f"  GDP: {os.path.basename(gdp_file)}")
            print(f"  Population: {os.path.basename(pop_file)}")
            
            # Load gridded data for this SSP using existing function
            ssp_data = load_gridded_data(config, ssp_name)
            
            # Store in organized structure
            all_data[ssp_name] = {
                'tas': ssp_data['tas'],      # [time, lat, lon]
                'pr': ssp_data['pr'],     # [time, lat, lon]
                'gdp': ssp_data['gdp'],              # [time, lat, lon]
                'pop': ssp_data['pop']      # [time, lat, lon]
            }
            
            # Store metadata from first SSP (coordinates same for all)
            if i == 0:
                all_data['_metadata'].update({
                    'lat': ssp_data['lat'],
                    'lon': ssp_data['lon'],
                    'grid_shape': (len(ssp_data['lat']), len(ssp_data['lon'])),
                    'time_shape': len(ssp_data['tas_years'])  # Assuming all same length
                })
                # Store years at top level for easy access across all functions
                all_data['years'] = ssp_data['common_years']
            
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
    ref_pop = all_data[reference_ssp]['pop']  # [time, lat, lon]

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





def calculate_tfp_coin_ssp(pop, gdp, params):
    """
    Calculate total factor productivity time series using the Solow-Swan growth model.

    Parameters
    ----------
    pop : array-like
        Time series of pop (L) in people
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









def calculate_weather_vars(all_data, config):
    """
    Calculate weather (filtered) variables and reference baselines for all SSPs.

    Applies 30-year LOESS filtering to temperature and precipitation data relative to
    reference period mean, creating tas_weather and pr_weather arrays for each SSP.
    Also computes reference climate baselines (tas0_2d, pr0_2d) once and stores them.

    Parameters
    ----------
    all_data : dict
        Data structure containing climate data for all SSPs
    config : dict
        Configuration containing time period definitions

    Returns
    -------
    dict
        Updated all_data with tas_weather, pr_weather, tas0_2d, and pr0_2d added
    """

    print("Computing weather variables (filtered climate data)...")

    # Get reference period indices
    time_periods = config['time_periods']
    years = all_data['years']  # Use years from top-level location
    ref_start_year = time_periods['reference_period']['start_year']
    ref_end_year = time_periods['reference_period']['end_year']

    filter_width = 30  # years (consistent with existing code)

    # Process each SSP scenario
    ssp_scenarios = config['ssp_scenarios']
    for ssp_name in ssp_scenarios['forward_simulation_ssps']:

        ssp_data = all_data[ssp_name]

        # Find reference period indices
        ref_start_idx = np.where(years == ref_start_year)[0][0]
        ref_end_idx = np.where(years == ref_end_year)[0][0]

        # Get climate data arrays [time, lat, lon]
        tas_data = ssp_data['tas']
        pr_data = ssp_data['pr']

        # Get dimensions
        ntime, nlat, nlon = tas_data.shape

        # Initialize weather arrays
        tas_weather = np.zeros_like(tas_data)
        pr_weather = np.zeros_like(pr_data)

        print(f"  Processing {ssp_name}: {nlat}x{nlon} grid cells...")

        # Apply filtering to each grid cell
        for lat_idx in range(nlat):
            if lat_idx % 20 == 0:  # Progress indicator
                print(f"    Latitude band {lat_idx+1}/{nlat}")

            for lon_idx in range(nlon):
                # Extract time series for this grid cell
                cell_tas = tas_data[:, lat_idx, lon_idx]
                cell_pr = pr_data[:, lat_idx, lon_idx]

                # Apply weather filtering
                tas_weather[:, lat_idx, lon_idx] = apply_time_series_filter(
                    cell_tas, filter_width, ref_start_idx, ref_end_idx
                )
                pr_weather[:, lat_idx, lon_idx] = apply_time_series_filter(
                    cell_pr, filter_width, ref_start_idx, ref_end_idx
                )

        # Add weather variables to SSP data
        ssp_data['tas_weather'] = tas_weather
        ssp_data['pr_weather'] = pr_weather

        print(f"  ✅ {ssp_name} weather variables computed")

    print("✅ All weather variables computed")

    # Compute reference climate baselines using reference SSP
    print("Computing reference climate baselines...")
    reference_ssp = config['ssp_scenarios']['reference_ssp']

    # Get reference SSP data
    ref_ssp_data = all_data[reference_ssp]
    tas_data = ref_ssp_data['tas']
    pr_data = ref_ssp_data['pr']

    # Get reference period indices
    ref_start_idx = np.where(years == ref_start_year)[0][0]
    ref_end_idx = np.where(years == ref_end_year)[0][0]

    # Calculate reference period means for all grid cells
    tas0_2d = np.mean(tas_data[ref_start_idx:ref_end_idx+1, :, :], axis=0)  # [lat, lon]
    pr0_2d = np.mean(pr_data[ref_start_idx:ref_end_idx+1, :, :], axis=0)  # [lat, lon]

    # Store reference baselines in all_data for easy access
    all_data['tas0_2d'] = tas0_2d
    all_data['pr0_2d'] = pr0_2d

    print("✅ Reference climate baselines computed and stored")
    return all_data
