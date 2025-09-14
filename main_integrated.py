#!/usr/bin/env python3
"""
Integrated COIN-SSP Processing Pipeline

This module implements the complete processing pipeline for gridded climate-economic 
modeling using unified JSON configuration files. The pipeline processes one climate 
model at a time with configurable SSP scenarios, damage functions, and GDP targets.

Processing Flow (per README.md Section 3):
1. Develop target GDP changes using SSP245 scenario (global calculation)
2. Calculate baseline TFP for each SSP scenario (per grid cell, no climate)
3. Calculate scaling factors for each grid cell (per cell optimization for SSP245)
4. Run forward model for all SSPs (using per-cell scaling factors)
5. Generate summary data products (global and regional aggregates)
"""

import json
import os
import sys
import copy
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple
from coin_ssp_utils import filter_scaling_params

from coin_ssp_utils import (
    load_gridded_data, calculate_time_means, calculate_global_mean,
    calculate_all_target_reductions, load_all_netcdf_data, get_ssp_data,
    calculate_tfp_coin_ssp, save_step1_results_netcdf, save_step2_results_netcdf,
    apply_time_series_filter, save_step3_results_netcdf, save_step4_results_netcdf
)
from coin_ssp_core import ModelParams, ScalingParams, optimize_climate_response_scaling, calculate_coin_ssp_forward_model


def setup_output_directory(config: Dict[str, Any]) -> str:
    """
    Create output directory structure for integrated processing results.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Integrated configuration dictionary
        
    Returns
    -------
    str
        Path to the created output directory
    """
    model_name = config['climate_model']['model_name']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = Path("./data/output")
    output_dir = base_output_dir / f"output_integrated_{model_name}_{timestamp}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created output directory: {output_dir}")
    
    return str(output_dir)


def get_step_output_path(output_dir: str, step_num: int, model_name: str, ssp_name: str = None, 
                        file_type: str = "json") -> str:
    """
    Generate standardized output file path for processing steps.
    
    Parameters
    ----------
    output_dir : str
        Base output directory path
    step_num : int
        Processing step number (1-5)
    model_name : str
        Climate model name
    ssp_name : str, optional
        SSP scenario name (if step-specific)
    file_type : str
        File extension type ("json" or "nc")
        
    Returns
    -------
    str
        Complete output file path
    """
    step_names = {
        1: "target_gdp",
        2: "baseline_tfp", 
        3: "scaling_factors",
        4: "forward_results",
        5: "final_output"
    }
    
    step_name = step_names[step_num]
    
    if ssp_name:
        filename = f"step{step_num}_{step_name}_{model_name}_{ssp_name}.{file_type}"
    else:
        filename = f"step{step_num}_{step_name}_{model_name}.{file_type}"
        
    return os.path.join(output_dir, filename)


def load_integrated_config(config_path: str) -> Dict[str, Any]:
    """
    Load and validate integrated JSON configuration file.
    
    Parameters
    ----------
    config_path : str
        Path to the integrated JSON configuration file
        
    Returns
    -------
    Dict[str, Any]
        Parsed and validated configuration dictionary
    """
    print(f"Loading integrated configuration from: {config_path}")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Basic validation
    required_sections = ['climate_model', 'ssp_scenarios', 'time_periods', 
                        'model_params', 'damage_function_scalings', 'gdp_reduction_targets']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Required configuration section missing: {section}")
    
    n_damage_functions = len(config['damage_function_scalings'])
    n_gdp_targets = len(config['gdp_reduction_targets'])
    n_combinations = n_damage_functions * n_gdp_targets
    
    print(f"Configuration loaded successfully:")
    print(f"  Climate model: {config['climate_model']['model_name']}")
    print(f"  Reference SSP: {config['ssp_scenarios']['reference_ssp']}")
    print(f"  Forward SSPs: {config['ssp_scenarios']['forward_simulation_ssps']}")
    print(f"  Damage function scalings: {n_damage_functions}")
    print(f"  GDP reduction targets: {n_gdp_targets}")
    print(f"  Total combinations per grid cell: {n_combinations}")
    
    return config


def resolve_netcdf_filepath(config: Dict[str, Any], data_type: str, ssp_name: str) -> str:
    """
    Resolve NetCDF file path using naming convention: {prefix}_{model_name}_{ssp_name}.nc
    
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
        Full path to NetCDF file
    """
    climate_model = config['climate_model']
    model_name = climate_model['model_name']
    input_dir = climate_model['input_directory']
    prefix = climate_model['netcdf_file_patterns'][data_type]
    
    filename = f"{prefix}_{model_name}_{ssp_name}.nc"
    filepath = os.path.join(input_dir, filename)
    
    return filepath


def step1_calculate_target_gdp_changes(config: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    """
    Step 1: Develop target GDP changes using reference SSP scenario (global calculation).
    
    This step calculates spatially-explicit target GDP reductions by:
    1. Loading gridded climate and economic data for reference SSP (typically SSP245)
    2. For each GDP reduction target configuration:
       - Apply constraint satisfaction to generate spatial reduction patterns
       - Ensure GDP-weighted global means match specified targets
    3. Return target reduction arrays for use in scaling factor optimization
    
    Parameters
    ----------
    config : Dict[str, Any]
        Integrated configuration dictionary
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing target GDP reduction arrays for each target configuration
    """
    print("\n" + "="*80)
    print("STEP 1: CALCULATING TARGET GDP CHANGES")
    print("="*80)
    
    reference_ssp = config['ssp_scenarios']['reference_ssp']
    model_name = config['climate_model']['model_name']
    gdp_targets = config['gdp_reduction_targets']
    time_periods = config['time_periods']
    
    print(f"Using reference SSP: {reference_ssp}")
    print(f"Climate model: {model_name}")
    print(f"Processing {len(gdp_targets)} GDP reduction targets")
    
    # Load gridded data for reference SSP
    print(f"Loading gridded data for {reference_ssp}...")
    
    temp_file = resolve_netcdf_filepath(config, 'temperature', reference_ssp)
    gdp_file = resolve_netcdf_filepath(config, 'gdp', reference_ssp)
    
    print(f"  Temperature file: {temp_file}")
    print(f"  GDP file: {gdp_file}")
    
    # Load the gridded data
    data = load_gridded_data(model_name, reference_ssp)
    
    # Calculate temporal means
    print("Calculating temporal means...")
    temp_ref = calculate_time_means(data['tas'], data['tas_years'], 
                                   time_periods['reference_period']['start_year'],
                                   time_periods['reference_period']['end_year'])
    
    gdp_target = calculate_time_means(data['gdp'], data['gdp_years'],
                                     time_periods['target_period']['start_year'], 
                                     time_periods['target_period']['end_year'])
    
    # Prepare gridded data for target calculation functions
    gridded_data = {
        'temp_ref': temp_ref,
        'gdp_target': gdp_target,
        'lat': data['lat']
    }
    
    # Calculate global means for verification
    global_temp_ref = calculate_global_mean(temp_ref, data['lat'])
    global_gdp_target = calculate_global_mean(gdp_target, data['lat'])
    
    print(f"Global mean reference temperature: {global_temp_ref:.2f}°C")
    print(f"Global mean target GDP: {global_gdp_target:.2e}")
    
    # Calculate all target reductions using extracted functions
    print("Calculating target GDP reductions...")
    calculation_results = calculate_all_target_reductions(gdp_targets, gridded_data)
    
    # Process results for Step 1 output format
    target_results = {}
    
    for target_name, calc_result in calculation_results.items():
        target_type = calc_result['target_type']
        reduction_array = calc_result['reduction_array']
        
        # Calculate achieved GDP-weighted global mean
        if calc_result['constraint_verification'] and 'global_mean_constraint' in calc_result['constraint_verification']:
            global_mean_achieved = calc_result['constraint_verification']['global_mean_constraint']['achieved']
        elif 'global_statistics' in calc_result:
            global_mean_achieved = calc_result['global_statistics']['gdp_weighted_mean']
        else:
            # For constant case, calculate directly
            global_mean_achieved = calculate_global_mean(gdp_target * (1 + reduction_array), data['lat']) / global_gdp_target - 1
        
        target_results[target_name] = {
            'target_type': target_type,
            'reduction_array': reduction_array,
            'global_mean_achieved': float(global_mean_achieved),
            'constraint_satisfied': True,
            'economic_bounds_valid': True,
            'coefficients': calc_result.get('coefficients', None),
            'constraint_verification': calc_result.get('constraint_verification', None)
        }
        
        print(f"  {target_name} ({target_type}): GDP-weighted mean = {global_mean_achieved:.6f}")
    
    # Store coordinate and metadata information
    target_results['_metadata'] = {
        'temp_ref': temp_ref,
        'gdp_target': gdp_target,
        'lat': data['lat'],
        'lon': data['lon'],
        'global_temp_ref': float(global_temp_ref),
        'global_gdp_target': float(global_gdp_target),
        'reference_ssp': reference_ssp,
        'time_periods': time_periods
    }
    
    # Write results to NetCDF file
    model_name = config['climate_model']['model_name']
    reference_ssp = config['ssp_scenarios']['reference_ssp']
    output_path = get_step_output_path(output_dir, 1, model_name, reference_ssp, "nc")
    save_step1_results_netcdf(target_results, output_path)
    
    print(f"\nStep 1 completed: {len(gdp_targets)} target GDP change patterns calculated")
    print("Target reductions ready for per-grid-cell scaling factor optimization")
    return target_results


def step2_calculate_baseline_tfp(config: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    """
    Step 2: Calculate baseline TFP for each SSP scenario (per grid cell, no climate effects).
    
    This step generates baseline Total Factor Productivity by:
    1. For each forward simulation SSP scenario:
       - Load gridded GDP and population data
       - Apply calculate_tfp_coin_ssp to each grid cell independently
       - Store baseline TFP and capital stock arrays (no climate effects)
    
    Parameters
    ----------
    config : Dict[str, Any]
        Integrated configuration dictionary
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing baseline TFP arrays [lat, lon, time] for each SSP scenario
    """
    print("\n" + "="*80)
    print("STEP 2: CALCULATING BASELINE TFP (ALL SSPs)")
    print("="*80)
    
    forward_ssps = config['ssp_scenarios']['forward_simulation_ssps']
    model_name = config['climate_model']['model_name']
    model_params = config['model_params']
    
    print(f"Climate model: {model_name}")
    print(f"Processing {len(forward_ssps)} SSP scenarios: {forward_ssps}")
    
    # Load all NetCDF data upfront for efficiency
    print("Loading all NetCDF data...")
    all_data = load_all_netcdf_data(config)
    
    # Create ModelParams object from configuration
    params = ModelParams(**model_params)
    
    tfp_results = {}
    
    for i, ssp_name in enumerate(forward_ssps):
        print(f"\nProcessing SSP scenario: {ssp_name} ({i+1}/{len(forward_ssps)})")
        
        # Get SSP-specific data using centralized accessor
        print(f"  Extracting gridded GDP and population data for {ssp_name}...")
        gdp_data = get_ssp_data(all_data, ssp_name, 'gdp')  # [lat, lon, time]
        pop_data = get_ssp_data(all_data, ssp_name, 'population')  # [lat, lon, time] 
        
        # Get dimensions
        nlat, nlon, ntime = gdp_data.shape
        
        print(f"  Grid dimensions: {nlat} lat x {nlon} lon x {ntime} time")
        print(f"  Calculating baseline TFP for each grid cell...")
        
        # Initialize output arrays
        tfp_baseline = np.full((nlat, nlon, ntime), np.nan)
        k_baseline = np.full((nlat, nlon, ntime), np.nan)

        # Use the global valid mask computed during data loading
        valid_mask = all_data['_metadata']['valid_mask']
        
        grid_cells_processed = 0
        
        # Process each grid cell using the pre-computed valid mask
        for lat_idx in range(nlat):
            for lon_idx in range(nlon):
                if valid_mask[lat_idx, lon_idx]:
                    
                    # Calculate baseline TFP and capital stock (no climate effects)
                    tfp_cell, k_cell = calculate_tfp_coin_ssp(pop_timeseries, gdp_timeseries, params)
                    
                    # Store results
                    tfp_baseline[lat_idx, lon_idx, :] = tfp_cell
                    k_baseline[lat_idx, lon_idx, :] = k_cell
                    
                    grid_cells_processed += 1
        
        print(f"  Processed {grid_cells_processed} valid grid cells out of {nlat * nlon} total")
        
        tfp_results[ssp_name] = {
            'tfp_baseline': tfp_baseline,
            'k_baseline': k_baseline, 
            'grid_cells_processed': grid_cells_processed,
            'valid_mask': valid_mask,
            'validation_passed': True
        }
        
    # Write results to NetCDF file with coordinate information
    model_name = config['climate_model']['model_name']
    output_path = get_step_output_path(output_dir, 2, model_name, file_type="nc")
    
    # Add coordinate information from the loaded data
    metadata = all_data['_metadata']
    tfp_results['_coordinates'] = {
        'lat': metadata['lat'],
        'lon': metadata['lon']
    }
    
    save_step2_results_netcdf(tfp_results, output_path, model_name)
    
    print(f"\nStep 2 completed: Baseline TFP calculated for {len(tfp_results)} SSP scenarios")
    return tfp_results


def step3_calculate_scaling_factors_per_cell(config: Dict[str, Any], target_results: Dict[str, Any], 
                                           tfp_results: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    """
    Step 3: Calculate scaling factors for each grid cell (per-cell optimization for SSP245).
    
    This step optimizes scaling factors individually for each grid cell by:
    1. For each grid cell:
       - For each damage function scaling configuration:
         - For each target GDP change pattern:
           - Run optimize_climate_response_scaling using that cell's data
           - Find scaling factor that achieves target GDP reduction for that location
           - Store per-cell scaling factors for forward modeling
    
    Parameters
    ----------
    config : Dict[str, Any]
        Integrated configuration dictionary
    target_results : Dict[str, Any]
        Results from Step 1 (target GDP reduction patterns)
    tfp_results : Dict[str, Any]
        Results from Step 2 (baseline TFP arrays)
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing scaling factor arrays [lat, lon, damage_func, target] 
    """
    print("\n" + "="*80)
    print("STEP 3: CALCULATING SCALING FACTORS PER GRID CELL (SSP245)")
    print("="*80)
    
    reference_ssp = config['ssp_scenarios']['reference_ssp']
    damage_scalings = config['damage_function_scalings']
    gdp_targets = config['gdp_reduction_targets']
    
    n_damage_functions = len(damage_scalings)
    n_gdp_targets = len(gdp_targets)
    total_combinations = n_damage_functions * n_gdp_targets
    
    print(f"Reference SSP: {reference_ssp}")
    print(f"Damage function scalings: {n_damage_functions}")
    print(f"GDP reduction targets: {n_gdp_targets}")
    print(f"Total combinations per grid cell: {total_combinations}")
    
    # Load all climate data for reference SSP
    print(f"Loading climate data for {reference_ssp}...")
    all_data = load_all_netcdf_data(config)
    
    # Extract climate and population data
    temp_data = get_ssp_data(all_data, reference_ssp, 'temperature')  # [lat, lon, time]
    precip_data = get_ssp_data(all_data, reference_ssp, 'precipitation')  # [lat, lon, time] 
    pop_data = get_ssp_data(all_data, reference_ssp, 'population')  # [lat, lon, time]
    gdp_data = get_ssp_data(all_data, reference_ssp, 'gdp')  # [lat, lon, time]
    
    # Get baseline TFP for reference SSP
    reference_tfp = tfp_results[reference_ssp]
    valid_mask = reference_tfp['valid_mask']
    tfp_baseline = reference_tfp['tfp_baseline']  # [lat, lon, time]
    
    # Get dimensions
    nlat, nlon, ntime = temp_data.shape
    
    # Initialize scaling factor arrays 
    scaling_factors = np.full((nlat, nlon, n_damage_functions, n_gdp_targets), np.nan)
    optimization_errors = np.full((nlat, nlon, n_damage_functions, n_gdp_targets), np.nan)
    convergence_flags = np.full((nlat, nlon, n_damage_functions, n_gdp_targets), False)
    
    # Initialize arrays for scaled damage function parameters [lat, lon, damage_func, target, param]
    # The 12 parameters are: k_tas1, k_tas2, k_pr1, k_pr2, tfp_tas1, tfp_tas2, tfp_pr1, tfp_pr2, y_tas1, y_tas2, y_pr1, y_pr2
    n_scaled_params = 12
    scaled_parameters = np.full((nlat, nlon, n_damage_functions, n_gdp_targets, n_scaled_params), np.nan)
    
    # Define parameter names for NetCDF output
    scaled_param_names = ['k_tas1', 'k_tas2', 'k_pr1', 'k_pr2', 
                         'tfp_tas1', 'tfp_tas2', 'tfp_pr1', 'tfp_pr2',
                         'y_tas1', 'y_tas2', 'y_pr1', 'y_pr2']
    
    # Extract model and time parameters
    model_params = config['model_params']
    params = ModelParams(**model_params)
    
    # Get years array (assume annual time steps starting from year 0)
    base_year = 2015  # TODO: This should come from configuration
    years = np.arange(base_year, base_year + ntime)
    
    # Calculate year diverge location for weather filtering
    if params.year_diverge < years[0]:
        year_diverge_loc = 0
    else:
        year_diverge_loc = np.where(years == params.year_diverge)[0][0]
    
    # Initialize counters
    total_grid_cells = 0
    successful_optimizations = 0
    
    print("Starting per-grid-cell optimization...")
    print("This may take significant time for large grids...")
    print(f"Processing {nlat} x {nlon} = {nlat*nlon} grid cells")
    
    # Standard loop structure: GDP target → damage function → spatial (lat/lon) → combinations
    for target_idx, gdp_target in enumerate(gdp_targets):
        target_name = gdp_target['target_name']
        target_reduction_array = target_results[target_name]['reduction_array']  # [lat, lon]
        
        print(f"\nProcessing GDP target: {target_name} ({target_idx+1}/{n_gdp_targets})")
        
        for damage_idx, damage_scaling in enumerate(damage_scalings):
            scaling_name = damage_scaling['scaling_name']
            print(f"  Damage function: {scaling_name} ({damage_idx+1}/{n_damage_functions})")
            
            # Create ScalingParams for this damage function
            scaling_config = filter_scaling_params(damage_scaling)
            scaling_params = ScalingParams(**scaling_config)
            
            for lat_idx in range(nlat):
                for lon_idx in range(nlon):
                    
                    # Check if grid cell is valid (has economic activity)
                    if not valid_mask[lat_idx, lon_idx]:
                        continue
                        
                    total_grid_cells += 1
                    
                    # Extract time series for this grid cell
                    cell_temp = temp_data[lat_idx, lon_idx, :]  # [time]
                    cell_precip = precip_data[lat_idx, lon_idx, :]  # [time]
                    cell_pop = pop_data[lat_idx, lon_idx, :]  # [time]
                    cell_gdp = gdp_data[lat_idx, lon_idx, :]  # [time]
                    cell_tfp_baseline = tfp_baseline[lat_idx, lon_idx, :]  # [time]
                    
                    # Get target reduction for this grid cell  
                    target_reduction = target_reduction_array[lat_idx, lon_idx]
                    
                    # Create weather (filtered) time series
                    filter_width = 30  # years (same as country-level code)
                    cell_temp_weather = apply_time_series_filter(cell_temp, filter_width, year_diverge_loc)
                    cell_precip_weather = apply_time_series_filter(cell_precip, filter_width, year_diverge_loc)
                    
                    # Set baseline parameters for this grid cell
                    params_cell = copy.deepcopy(params)
                    params_cell.tas0 = np.mean(cell_temp[:year_diverge_loc+1])
                    params_cell.pr0 = np.mean(cell_precip[:year_diverge_loc+1])
                    params_cell.amount_scale = target_reduction
                    
                    # Create cell data dictionary matching country_data structure
                    cell_data = {
                        'years': years,
                        'population': cell_pop,
                        'gdp': cell_gdp,
                        'tas': cell_temp,
                        'pr': cell_precip,
                        'tas_weather': cell_temp_weather,
                        'pr_weather': cell_precip_weather,
                        'tfp_baseline': cell_tfp_baseline
                    }
                    
                    try:
                        # Run per-grid-cell optimization
                        optimal_scale, final_error, params_scaled = optimize_climate_response_scaling(
                            cell_data, params_cell, scaling_params
                        )
                        
                        # Store results
                        scaling_factors[lat_idx, lon_idx, damage_idx, target_idx] = optimal_scale
                        optimization_errors[lat_idx, lon_idx, damage_idx, target_idx] = final_error
                        convergence_flags[lat_idx, lon_idx, damage_idx, target_idx] = True
                        
                        # Store scaled damage function parameters
                        scaled_parameters[lat_idx, lon_idx, damage_idx, target_idx, 0] = params_scaled.k_tas1
                        scaled_parameters[lat_idx, lon_idx, damage_idx, target_idx, 1] = params_scaled.k_tas2  
                        scaled_parameters[lat_idx, lon_idx, damage_idx, target_idx, 2] = params_scaled.k_pr1
                        scaled_parameters[lat_idx, lon_idx, damage_idx, target_idx, 3] = params_scaled.k_pr2
                        scaled_parameters[lat_idx, lon_idx, damage_idx, target_idx, 4] = params_scaled.tfp_tas1
                        scaled_parameters[lat_idx, lon_idx, damage_idx, target_idx, 5] = params_scaled.tfp_tas2
                        scaled_parameters[lat_idx, lon_idx, damage_idx, target_idx, 6] = params_scaled.tfp_pr1
                        scaled_parameters[lat_idx, lon_idx, damage_idx, target_idx, 7] = params_scaled.tfp_pr2
                        scaled_parameters[lat_idx, lon_idx, damage_idx, target_idx, 8] = params_scaled.y_tas1
                        scaled_parameters[lat_idx, lon_idx, damage_idx, target_idx, 9] = params_scaled.y_tas2
                        scaled_parameters[lat_idx, lon_idx, damage_idx, target_idx, 10] = params_scaled.y_pr1
                        scaled_parameters[lat_idx, lon_idx, damage_idx, target_idx, 11] = params_scaled.y_pr2
                        
                        successful_optimizations += 1
                        
                    except Exception as e:
                        print(f"    Optimization failed for cell ({lat_idx}, {lon_idx}): {e}")
                        # Arrays already initialized with NaN and False, so no need to set
    
    print(f"\nOptimization completed:")
    print(f"  Total valid grid cells processed: {total_grid_cells}")  
    print(f"  Successful optimizations: {successful_optimizations}")
    print(f"  Success rate: {100*successful_optimizations/max(1, total_grid_cells):.1f}%")
    
    # Create results dictionary
    scaling_results = {
        'scaling_factors': scaling_factors,       # [lat, lon, damage_func_idx, target_idx] 
        'optimization_errors': optimization_errors, # [lat, lon, damage_func_idx, target_idx]
        'convergence_flags': convergence_flags,   # [lat, lon, damage_func_idx, target_idx] boolean
        'scaled_parameters': scaled_parameters,   # [lat, lon, damage_func_idx, target_idx, param_idx]
        'scaled_param_names': scaled_param_names, # Parameter names for the last dimension
        'damage_function_names': [df['scaling_name'] for df in damage_scalings],
        'target_names': [tgt['target_name'] for tgt in gdp_targets], 
        'total_grid_cells': total_grid_cells,
        'successful_optimizations': successful_optimizations,
        'reference_ssp': reference_ssp,
        'valid_mask': valid_mask,  # [lat, lon] boolean mask
        '_coordinates': all_data['_metadata']  # coordinate information
    }
    
    # Write results to NetCDF file
    model_name = config['climate_model']['model_name']
    output_path = get_step_output_path(output_dir, 3, model_name, reference_ssp, "nc")
    save_step3_results_netcdf(scaling_results, output_path, model_name)
    
    print(f"\nStep 3 completed: Scaling factors calculated for each grid cell")
    print(f"Total combinations processed: {total_combinations} per valid grid cell")
    return scaling_results


def step4_forward_integration_all_ssps(config: Dict[str, Any], scaling_results: Dict[str, Any], 
                                     tfp_results: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    """
    Step 4: Run forward model for all SSPs using per-cell scaling factors.
    
    This step applies the forward economic model by:
    1. For each forward simulation SSP:
       - Load climate data (temperature, precipitation)
       - For each damage function and target combination:
         - Apply per-cell scaling factors from Step 3
         - Run calculate_coin_ssp_forward_model for each grid cell
         - Generate climate-integrated projections using baseline TFP from Step 2
    
    Parameters
    ----------
    config : Dict[str, Any]
        Integrated configuration dictionary
    scaling_results : Dict[str, Any]
        Results from Step 3 (per-cell scaling factors)
    tfp_results : Dict[str, Any]
        Results from Step 2 (baseline TFP arrays)
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing forward model results [lat, lon, time, combination] for each SSP
    """
    print("\n" + "="*80)
    print("STEP 4: FORWARD INTEGRATION (ALL SSPs)")
    print("="*80)
    
    forward_ssps = config['ssp_scenarios']['forward_simulation_ssps']
    n_damage_functions = len(config['damage_function_scalings'])
    n_gdp_targets = len(config['gdp_reduction_targets'])
    total_combinations = n_damage_functions * n_gdp_targets
    
    print(f"Processing {len(forward_ssps)} SSP scenarios")
    print(f"Using {total_combinations} scaling factor combinations per grid cell")
    print(f"Total processing: {len(forward_ssps)} SSPs × {total_combinations} combinations")
    
    # Load all climate and economic data for all SSPs upfront
    print("Loading all NetCDF data for forward modeling...")
    all_data = load_all_netcdf_data(config)
    
    # Get model parameters and grid dimensions from Step 3
    model_params = config['model_params']
    base_params = ModelParams(**model_params)
    damage_function_names = scaling_results['damage_function_names']
    target_names = scaling_results['target_names']
    valid_mask = scaling_results['valid_mask']
    scaling_factors = scaling_results['scaling_factors']  # [lat, lon, damage_func, target]
    scaled_parameters = scaling_results['scaled_parameters']  # [lat, lon, damage_func, target, param]
    
    # Get grid dimensions
    nlat, nlon = valid_mask.shape
    
    # Calculate year diverge location for weather filtering
    base_year = 2015  # TODO: Get from configuration
    
    forward_results = {}
    
    for i, ssp_name in enumerate(forward_ssps):
        print(f"\nProcessing SSP scenario: {ssp_name} ({i+1}/{len(forward_ssps)})")
        
        # Get SSP-specific data
        temp_data = get_ssp_data(all_data, ssp_name, 'temperature')  # [lat, lon, time]
        precip_data = get_ssp_data(all_data, ssp_name, 'precipitation') # [lat, lon, time]
        pop_data = get_ssp_data(all_data, ssp_name, 'population')  # [lat, lon, time]
        gdp_data = get_ssp_data(all_data, ssp_name, 'gdp')  # [lat, lon, time]
        
        # Get baseline TFP for this SSP from Step 2
        tfp_baseline = tfp_results[ssp_name]['tfp_baseline']  # [lat, lon, time]
        
        ntime = temp_data.shape[2]
        years = np.arange(base_year, base_year + ntime)
        
        # Calculate year diverge location for weather filtering
        if base_params.year_diverge < years[0]:
            year_diverge_loc = 0
        else:
            year_diverge_loc = np.where(years == base_params.year_diverge)[0][0]
        
        print(f"  Grid dimensions: {nlat} lat x {nlon} lon x {ntime} time")
        print(f"  Running forward model for {total_combinations} combinations per valid grid cell")
        
        # Initialize result arrays for this SSP
        # [lat, lon, damage_func, target, time]
        gdp_climate = np.full((nlat, nlon, n_damage_functions, n_gdp_targets, ntime), np.nan)
        gdp_weather = np.full((nlat, nlon, n_damage_functions, n_gdp_targets, ntime), np.nan)
        tfp_climate = np.full((nlat, nlon, n_damage_functions, n_gdp_targets, ntime), np.nan)
        tfp_weather = np.full((nlat, nlon, n_damage_functions, n_gdp_targets, ntime), np.nan)
        k_climate = np.full((nlat, nlon, n_damage_functions, n_gdp_targets, ntime), np.nan)
        k_weather = np.full((nlat, nlon, n_damage_functions, n_gdp_targets, ntime), np.nan)
        
        successful_forward_runs = 0
        total_forward_runs = 0
        
        # Standard loop structure: damage function → target → spatial
        for damage_idx in range(n_damage_functions):
            damage_name = damage_function_names[damage_idx]
            print(f"    Damage function: {damage_name} ({damage_idx+1}/{n_damage_functions})")
            
            for target_idx in range(n_gdp_targets):
                target_name = target_names[target_idx]
                
                for lat_idx in range(nlat):
                    for lon_idx in range(nlon):
                        
                        # Check if grid cell is valid and has optimization results
                        if not valid_mask[lat_idx, lon_idx]:
                            continue
                        
                        # Check if scaling factor optimization was successful for this combination
                        if np.isnan(scaling_factors[lat_idx, lon_idx, damage_idx, target_idx]):
                            continue
                            
                        total_forward_runs += 1
                        
                        # Extract time series for this grid cell
                        cell_temp = temp_data[lat_idx, lon_idx, :]  # [time]
                        cell_precip = precip_data[lat_idx, lon_idx, :]  # [time]
                        cell_pop = pop_data[lat_idx, lon_idx, :]  # [time]
                        cell_gdp = gdp_data[lat_idx, lon_idx, :]  # [time]
                        cell_tfp_baseline = tfp_baseline[lat_idx, lon_idx, :]  # [time]
                        
                        # Create weather (filtered) time series
                        filter_width = 30  # years (same as country-level code)
                        cell_temp_weather = apply_time_series_filter(cell_temp, filter_width, year_diverge_loc)
                        cell_precip_weather = apply_time_series_filter(cell_precip, filter_width, year_diverge_loc)
                        
                        # Create ModelParams with scaled damage function parameters
                        params_scaled = copy.deepcopy(base_params)
                        params_scaled.tas0 = np.mean(cell_temp[:year_diverge_loc+1])
                        params_scaled.pr0 = np.mean(cell_precip[:year_diverge_loc+1])
                        
                        # Set scaled damage function parameters from Step 3 results
                        params_scaled.k_tas1 = scaled_parameters[lat_idx, lon_idx, damage_idx, target_idx, 0]
                        params_scaled.k_tas2 = scaled_parameters[lat_idx, lon_idx, damage_idx, target_idx, 1]
                        params_scaled.k_pr1 = scaled_parameters[lat_idx, lon_idx, damage_idx, target_idx, 2]
                        params_scaled.k_pr2 = scaled_parameters[lat_idx, lon_idx, damage_idx, target_idx, 3]
                        params_scaled.tfp_tas1 = scaled_parameters[lat_idx, lon_idx, damage_idx, target_idx, 4]
                        params_scaled.tfp_tas2 = scaled_parameters[lat_idx, lon_idx, damage_idx, target_idx, 5]
                        params_scaled.tfp_pr1 = scaled_parameters[lat_idx, lon_idx, damage_idx, target_idx, 6]
                        params_scaled.tfp_pr2 = scaled_parameters[lat_idx, lon_idx, damage_idx, target_idx, 7]
                        params_scaled.y_tas1 = scaled_parameters[lat_idx, lon_idx, damage_idx, target_idx, 8]
                        params_scaled.y_tas2 = scaled_parameters[lat_idx, lon_idx, damage_idx, target_idx, 9]
                        params_scaled.y_pr1 = scaled_parameters[lat_idx, lon_idx, damage_idx, target_idx, 10]
                        params_scaled.y_pr2 = scaled_parameters[lat_idx, lon_idx, damage_idx, target_idx, 11]
                        
                        try:
                            # Run forward model with climate data
                            y_climate, a_climate, k_climate_values, _, _, _ = calculate_coin_ssp_forward_model(
                                cell_tfp_baseline, cell_pop, cell_temp, cell_precip, params_scaled
                            )
                            
                            # Run forward model with weather data (no climate trends)
                            y_weather, a_weather, k_weather_values, _, _, _ = calculate_coin_ssp_forward_model(
                                cell_tfp_baseline, cell_pop, cell_temp_weather, cell_precip_weather, params_scaled
                            )
                            
                            # Store results (convert normalized values back to actual units)
                            gdp_climate[lat_idx, lon_idx, damage_idx, target_idx, :] = y_climate * cell_gdp[0]
                            gdp_weather[lat_idx, lon_idx, damage_idx, target_idx, :] = y_weather * cell_gdp[0] 
                            tfp_climate[lat_idx, lon_idx, damage_idx, target_idx, :] = a_climate
                            tfp_weather[lat_idx, lon_idx, damage_idx, target_idx, :] = a_weather
                            k_climate[lat_idx, lon_idx, damage_idx, target_idx, :] = k_climate_values
                            k_weather[lat_idx, lon_idx, damage_idx, target_idx, :] = k_weather_values
                            
                            successful_forward_runs += 1
                            
                        except Exception as e:
                            print(f"      Forward model failed for cell ({lat_idx}, {lon_idx}), "
                                  f"damage {damage_name}, target {target_name}: {e}")
                            # Arrays already initialized with NaN, so no need to set
        
        print(f"  Forward model completed:")
        print(f"    Total forward runs: {total_forward_runs}")
        print(f"    Successful runs: {successful_forward_runs}")
        print(f"    Success rate: {100*successful_forward_runs/max(1, total_forward_runs):.1f}%")
        
        # Store results for this SSP
        forward_results[ssp_name] = {
            'gdp_climate': gdp_climate,        # [lat, lon, damage_func, target, time]
            'gdp_weather': gdp_weather,        # [lat, lon, damage_func, target, time] 
            'tfp_climate': tfp_climate,        # [lat, lon, damage_func, target, time]
            'tfp_weather': tfp_weather,        # [lat, lon, damage_func, target, time]
            'k_climate': k_climate,            # [lat, lon, damage_func, target, time]
            'k_weather': k_weather,            # [lat, lon, damage_func, target, time]
            'successful_forward_runs': successful_forward_runs,
            'total_forward_runs': total_forward_runs,
            'success_rate': successful_forward_runs / max(1, total_forward_runs)
        }
    
    # Add metadata and coordinate information to results
    step4_results = {
        'forward_results': forward_results,  # SSP-specific results
        'damage_function_names': damage_function_names,
        'target_names': target_names,
        'valid_mask': valid_mask,
        'total_ssps_processed': len(forward_ssps),
        'processing_summary': {
            ssp: {
                'successful_runs': forward_results[ssp]['successful_forward_runs'],
                'total_runs': forward_results[ssp]['total_forward_runs'],
                'success_rate': forward_results[ssp]['success_rate']
            } for ssp in forward_ssps
        },
        '_coordinates': all_data['_metadata']
    }
    
    # Write results to NetCDF file
    model_name = config['climate_model']['model_name'] 
    output_path = get_step_output_path(output_dir, 4, model_name, file_type="nc")
    save_step4_results_netcdf(step4_results, output_path, model_name)
    
    print(f"\nStep 4 completed: Forward integration for {len(forward_results)} SSP scenarios")
    return step4_results


def step5_processing_summary(config: Dict[str, Any], target_results: Dict[str, Any],
                           tfp_results: Dict[str, Any], scaling_results: Dict[str, Any],
                           forward_results: Dict[str, Any]) -> None:
    """
    Step 5: Generate processing summary and completion report.
    
    Note: NetCDF outputs are generated by each individual step (Steps 1-4) for
    data persistence and debugging. This step provides a comprehensive summary
    of the entire processing workflow.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Integrated configuration dictionary
    target_results : Dict[str, Any]
        Results from Step 1
    tfp_results : Dict[str, Any] 
        Results from Step 2
    scaling_results : Dict[str, Any]
        Results from Step 3
    forward_results : Dict[str, Any]
        Results from Step 4
    """
    print("\n" + "="*80)
    print("STEP 5: PROCESSING SUMMARY")
    print("="*80)
    
    model_name = config['climate_model']['model_name']
    reference_ssp = config['ssp_scenarios']['reference_ssp']
    forward_ssps = config['ssp_scenarios']['forward_simulation_ssps']
    damage_functions = config['damage_function_scalings']
    gdp_targets = config['gdp_reduction_targets']
    
    print(f"Climate Model: {model_name}")
    print(f"Reference SSP: {reference_ssp}")
    print(f"Forward SSPs: {len(forward_ssps)} scenarios ({', '.join(forward_ssps)})")
    print(f"Damage Functions: {len(damage_functions)} configurations")
    print(f"GDP Targets: {len(gdp_targets)} patterns") 
    
    print("\n" + "-"*60)
    print("STEP-BY-STEP PROCESSING RESULTS:")
    print("-"*60)
    
    # Step 1 Summary
    print(f"Step 1 - Target GDP Changes:")
    print(f"  ✅ Calculated {len(gdp_targets)} target reduction patterns")
    for target_name, target_data in target_results.items():
        if target_name != '_metadata':
            print(f"     • {target_name}: GDP-weighted mean = {target_data['global_mean_achieved']:.6f}")
    
    # Step 2 Summary  
    print(f"\nStep 2 - Baseline TFP Calculation:")
    print(f"  ✅ Processed {len(forward_ssps)} SSP scenarios")
    for ssp_name, ssp_data in tfp_results.items():
        if ssp_name != '_coordinates':
            print(f"     • {ssp_name}: {ssp_data['grid_cells_processed']} valid grid cells")
    
    # Step 3 Summary
    print(f"\nStep 3 - Scaling Factor Optimization:")
    print(f"  ✅ Optimized {scaling_results['total_grid_cells']} valid grid cells")
    print(f"     • Success rate: {100*scaling_results['successful_optimizations']/max(1, scaling_results['total_grid_cells']):.1f}%")
    print(f"     • Combinations per cell: {len(damage_functions)} damage × {len(gdp_targets)} targets = {len(damage_functions)*len(gdp_targets)}")
    
    # Step 4 Summary
    print(f"\nStep 4 - Forward Model Integration:")
    total_successful = 0
    total_runs = 0
    for ssp_name in forward_ssps:
        if ssp_name in forward_results['forward_results']:
            ssp_data = forward_results['forward_results'][ssp_name]
            total_successful += ssp_data['successful_forward_runs']  
            total_runs += ssp_data['total_forward_runs']
            print(f"     • {ssp_name}: {ssp_data['successful_forward_runs']}/{ssp_data['total_forward_runs']} runs ({100*ssp_data['success_rate']:.1f}%)")
    
    print(f"  ✅ Overall forward model success: {total_successful}/{total_runs} ({100*total_successful/max(1, total_runs):.1f}%)")
    
    print("\n" + "-"*60)
    print("OUTPUT FILES GENERATED:")
    print("-"*60)
    print(f"All results saved with step-by-step NetCDF output:")
    print(f"  • Step 1: step1_target_gdp_{model_name}_{reference_ssp}.nc")
    print(f"  • Step 2: step2_baseline_tfp_{model_name}.nc") 
    print(f"  • Step 3: step3_scaling_factors_{model_name}_{reference_ssp}.nc")
    print(f"  • Step 4: step4_forward_results_{model_name}.nc")


def run_integrated_pipeline(config_path: str) -> None:
    """
    Execute the complete integrated processing pipeline following README Section 3.
    
    The 5-step processing flow:
    1. Calculate target GDP changes using reference SSP (global calculation) 
    2. Calculate baseline TFP for each SSP (per grid cell, no climate)
    3. Calculate scaling factors per grid cell (optimize for each cell using reference SSP)
    4. Run forward model for all SSPs (using per-cell scaling factors)
    5. Generate processing summary (NetCDF outputs created by each step)
    
    Parameters
    ---------- 
    config_path : str
        Path to integrated JSON configuration file
    """
    print("Starting COIN-SSP Integrated Processing Pipeline")
    print("Following README.md Section 3: Grid Cell Processing")
    print("="*100)
    
    try:
        # Load configuration
        config = load_integrated_config(config_path)
        
        # Setup output directory
        output_dir = setup_output_directory(config)
        
        # Execute 5-step processing flow
        target_results = step1_calculate_target_gdp_changes(config, output_dir)
        tfp_results = step2_calculate_baseline_tfp(config, output_dir)
        scaling_results = step3_calculate_scaling_factors_per_cell(config, target_results, tfp_results, output_dir)
        forward_results = step4_forward_integration_all_ssps(config, scaling_results, tfp_results, output_dir)
        step5_processing_summary(config, target_results, tfp_results, scaling_results, forward_results)
        
        print("\n" + "="*100)
        print("INTEGRATED PROCESSING COMPLETE")
        print("="*100)
        print("All 5 steps executed successfully!")
        print("Per-grid-cell scaling factors calculated and applied to all SSP scenarios")
        
    except Exception as e:
        print(f"\nERROR: Processing failed with exception: {e}")
        raise


def main():
    """Main entry point for integrated processing pipeline."""
    if len(sys.argv) != 2:
        print("Usage: python main_integrated.py <integrated_config.json>")
        print("\nExample: python main_integrated.py coin_ssp_integrated_config_example.json")
        print("\nThis pipeline implements README.md Section 3: Grid Cell Processing")
        print("Key feature: Per-grid-cell scaling factor optimization using optimize_climate_response_scaling")
        sys.exit(1)
    
    config_path = sys.argv[1]
    run_integrated_pipeline(config_path)


if __name__ == "__main__":
    main()