#!/usr/bin/env python3
"""
Integrated COIN-SSP Processing Pipeline

This module implements the complete processing pipeline for gridded climate-economic 
modeling using unified JSON configuration files. The pipeline processes one climate 
model at a time with configurable SSP scenarios, response functions, and GDP targets.

Processing Flow (per README.md Section 3):
1. Develop target GDP changes using SSP245 scenario (global calculation)
2. Calculate baseline TFP for each SSP scenario (per grid cell, no climate)
3. Calculate scaling factors for each grid cell (per cell optimization for SSP245)
4. Run forward model for all SSPs (using per-cell scaling factors)
5. Generate summary data products (global and regional aggregates)
"""

import argparse
import copy
import json
import os
import pandas as pd
import shutil
import sys
import time
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple
from coin_ssp_utils import filter_scaling_params

from coin_ssp_models import ScalingParams, ModelParams

from coin_ssp_utils import (
    calculate_time_means, calculate_global_mean,
    calculate_all_target_reductions, load_all_data, get_ssp_data, get_grid_metadata,
    calculate_tfp_coin_ssp, save_step1_results_netcdf, save_step2_results_netcdf,
    apply_time_series_filter, save_step3_results_netcdf, save_step4_results_netcdf_split,
    create_target_gdp_visualization, create_baseline_tfp_visualization, create_scaling_factors_visualization,
    create_objective_function_visualization,
    create_forward_model_visualization, create_forward_model_maps_visualization,
    create_forward_model_ratio_visualization, load_step3_results_from_netcdf,
    calculate_area_weights, calculate_weather_vars
)
from coin_ssp_core import (
    optimize_climate_response_scaling, calculate_coin_ssp_forward_model,
    calculate_reference_gdp_climate_variability, apply_variability_target_scaling, process_response_target_optimization
)
from model_params_factory import ModelParamsFactory


def build_filename(prefix: str, json_id: str, model_name: str, ext: str, ssp_name: str = None, var_name: str = None) -> str:
    """
    Build standardized filename with format: {prefix}_{json_id}_{model_name}_{ssp_name}_{var_name}.{ext}
    Missing fields (None) are omitted along with their preceding underscore.

    Parameters
    ----------
    prefix : str
        File prefix (e.g., "step1", "step2")
    json_id : str
        JSON configuration ID (e.g., "0007")
    model_name : str, optional
        Climate model name (e.g., "CanESM5")
    ssp_name : str, optional
        SSP scenario name (e.g., "ssp245")
    var_name : str, optional
        Variable name (e.g., "target_gdp", "baseline_tfp")
    ext : str
        File extension (default: "nc")

    Returns
    -------
    str
        Standardized filename
    """
    parts = [prefix, json_id]
    if model_name:
        parts.append(model_name)
    if ssp_name:
        parts.append(ssp_name)
    if var_name:
        parts.append(var_name)

    return "_".join(parts) + f".{ext}"


def setup_output_directory(config: Dict[str, Any]) -> str:
    """
    Create output directory structure for integrated processing results.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Integrated configuration dictionary (must contain run_metadata.json_id)

    Returns
    -------
    str
        Path to the created output directory
    """
    # Get json_id from run_metadata
    json_id = config['run_metadata']['json_id']

    model_name = config['climate_model']['model_name']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = Path("./data/output")
    output_dir = base_output_dir / f"output_{json_id}_{model_name}_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created output directory: {output_dir}")

    return str(output_dir)


def get_step_output_path(output_dir: str, step_num: int, config: Dict[str, Any], ssp_name: str = None,
                        var_name: str = None, file_type: str = "nc") -> str:
    """
    Generate standardized output file path for processing steps.

    Parameters
    ----------
    output_dir : str
        Base output directory path
    step_num : int
        Processing step number (1-5)
    config : Dict[str, Any]
        Configuration dictionary containing run_metadata.json_id and climate_model.model_name
    ssp_name : str, optional
        SSP scenario name (if step-specific)
    var_name : str, optional
        Variable name (if step-specific)
    file_type : str
        File extension type (default: "nc")

    Returns
    -------
    str
        Complete output file path
    """
    # Extract configuration values
    json_id = config['run_metadata']['json_id']
    model_name = config['climate_model']['model_name']

    # Create step-specific prefix
    prefix = f"step{step_num}"
    filename = build_filename(prefix, json_id, model_name, file_type, ssp_name, var_name)

    return os.path.join(output_dir, filename)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load and validate JSON configuration file.
    
    Parameters
    ----------
    config_path : str
        Path to the JSON configuration file
        
    Returns
    -------
    Dict[str, Any]
        Parsed and validated configuration dictionary
    """
    print(f"Loading configuration from: {config_path}")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Create ModelParams factory for clean parameter management
    config['model_params_factory'] = ModelParamsFactory(config['model_params'])

    # Basic validation
    required_sections = ['climate_model', 'ssp_scenarios', 'time_periods',
                        'model_params', 'response_function_scalings', 'gdp_targets']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Required configuration section missing: {section}")
    
    n_response_functions = len(config['response_function_scalings'])
    n_gdp_targets = len(config['gdp_targets'])
    n_combinations = n_response_functions * n_gdp_targets
    
    print(f"Configuration loaded successfully:")
    print(f"  Climate model: {config['climate_model']['model_name']}")
    print(f"  Reference SSP: {config['ssp_scenarios']['reference_ssp']}")
    print(f"  Forward SSPs: {config['ssp_scenarios']['forward_simulation_ssps']}")
    print(f"  Response function scalings: {n_response_functions}")
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
        Type of data ('tas', 'pr', 'gdp', 'pop', 'target_reductions')
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


def step1_calculate_target_gdp_changes(config: Dict[str, Any], output_dir: str, json_id: str, all_data: Dict[str, Any]) -> Dict[str, Any]:
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
    gdp_targets = config['gdp_targets']
    time_periods = config['time_periods']

    print(f"Using reference SSP: {reference_ssp}")
    print(f"Climate model: {model_name}")
    print(f"Processing {len(gdp_targets)} GDP reduction targets")

    prediction_start = time_periods['prediction_period']['start_year']
    print(f"Prediction period start year: {prediction_start}")

    # Convert from all_data format to Step 1's expected format
    print(f"Using pre-loaded NetCDF data for {reference_ssp}...")
    data = {
        'tas': all_data[reference_ssp]['tas'],
        'gdp': all_data[reference_ssp]['gdp'],
        'lat': all_data['_metadata']['lat'],
        'lon': all_data['_metadata']['lon'],
        'years': all_data['years']
          }
    
    # Calculate temporal means
    print("Calculating temporal means...")
    tas_ref = calculate_time_means(data['tas'], data['years'], 
                                   time_periods['reference_period']['start_year'],
                                   time_periods['reference_period']['end_year'])
    
    gdp_target = calculate_time_means(data['gdp'], data['years'],
                                     time_periods['target_period']['start_year'], 
                                     time_periods['target_period']['end_year'])
    
    # Get valid mask from metadata
    valid_mask = all_data['_metadata']['valid_mask']

    # Prepare gridded data for target calculation functions
    gridded_data = {
        'tas_ref': tas_ref,
        'gdp_target': gdp_target,
        'lat': data['lat'],
        'valid_mask': valid_mask
    }
    
    # Calculate global means for verification using valid mask
    global_tas_ref = calculate_global_mean(tas_ref, data['lat'], valid_mask)
    global_gdp_target = calculate_global_mean(gdp_target, data['lat'], valid_mask)
    
    print(f"Global mean reference temperature: {global_tas_ref:.2f}°C")
    print(f"Global mean target GDP: {global_gdp_target:.2e}")
    
    # Calculate all target reductions using extracted functions
    print("Calculating target GDP reductions...")
    calculation_results = calculate_all_target_reductions(gdp_targets, gridded_data)
    
    # Process results for Step 1 output format
    target_results = {}
    
    for target_name, calc_result in calculation_results.items():
        target_shape = calc_result['target_shape']
        reduction_array = calc_result['reduction_array']
        
        # Calculate achieved GDP-weighted global mean
        if calc_result['constraint_verification'] and 'global_mean_constraint' in calc_result['constraint_verification']:
            global_mean_achieved = calc_result['constraint_verification']['global_mean_constraint']['achieved']
        elif 'global_statistics' in calc_result:
            global_mean_achieved = calc_result['global_statistics']['gdp_weighted_mean']
        else:
            # For constant case, calculate directly
            global_mean_achieved = calculate_global_mean(gdp_target * (1 + reduction_array), data['lat'], valid_mask) / global_gdp_target - 1
        
        target_results[target_name] = {
            'target_shape': target_shape,
            'reduction_array': reduction_array,
            'global_mean_achieved': float(global_mean_achieved),
            'constraint_satisfied': True,
            'economic_bounds_valid': True,
            'coefficients': calc_result['coefficients'],
            'constraint_verification': calc_result['constraint_verification']
        }
        
        print(f"  {target_name} ({target_shape}): GDP-weighted mean = {global_mean_achieved:.6f}")
    
    # Store coordinate and metadata information
    target_results['_metadata'] = {
        'tas_ref': tas_ref,
        'gdp_target': gdp_target,
        'lat': data['lat'],
        'lon': data['lon'],
        'global_tas_ref': float(global_tas_ref),
        'global_gdp_target': float(global_gdp_target),
        'reference_ssp': reference_ssp,
        'time_periods': time_periods
    }
    
    # Write results to NetCDF file
    model_name = config['climate_model']['model_name']
    reference_ssp = config['ssp_scenarios']['reference_ssp']
    output_path = get_step_output_path(output_dir, 1, config, reference_ssp, "target_gdp", "nc")
    save_step1_results_netcdf(target_results, output_path, config)

    # Create visualization
    print("Generating target GDP visualization...")
    visualization_path = create_target_gdp_visualization(target_results, config, output_dir,
                                                        reference_ssp, valid_mask)
    print(f"✅ Visualization saved: {visualization_path}")

    print(f"\nStep 1 completed: {len(gdp_targets)} target GDP change patterns calculated")
    print("Target reductions ready for per-grid-cell scaling factor optimization")
    return target_results


def step2_calculate_baseline_tfp(config: Dict[str, Any], output_dir: str, all_data: Dict[str, Any]) -> Dict[str, Any]:
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
    
    # Use pre-loaded data
    print("Using pre-loaded NetCDF data...")
    all_data = all_data
    
    # Create ModelParams instance using factory
    params = config['model_params_factory'].create_base()
    
    tfp_results = {}
    
    for i, ssp_name in enumerate(forward_ssps):
        print(f"\nProcessing SSP scenario: {ssp_name} ({i+1}/{len(forward_ssps)})")
        
        # Get SSP-specific data using centralized accessor
        print(f"  Extracting gridded GDP and population data for {ssp_name}...")
        gdp_data = get_ssp_data(all_data, ssp_name, 'gdp')  # [time, lat, lon]
        pop_data = get_ssp_data(all_data, ssp_name, 'pop')  # [time, lat, lon] 
        
        # Get dimensions
        # GDP data is [time, lat, lon] = (137, 64, 128)
        ntime, nlat, nlon = gdp_data.shape

        print(f"  Grid dimensions: {ntime} time x {nlat} lat x {nlon} lon")
        print(f"  Calculating baseline TFP for each grid cell...")

        # Initialize output arrays [time, lat, lon] for results storage
        tfp_baseline = np.full((ntime, nlat, nlon), np.nan)
        k_baseline = np.full((ntime, nlat, nlon), np.nan)

        # Use the global valid mask computed during data loading
        valid_mask = all_data['_metadata']['valid_mask']
        
        grid_cells_processed = 0
        
        # Process each grid cell using the pre-computed valid mask
        for lat_idx in range(nlat):
            for lon_idx in range(nlon):
                if valid_mask[lat_idx, lon_idx]:
                    # Extract time series for this valid grid cell (data is [time, lat, lon])
                    pop_timeseries = pop_data[:, lat_idx, lon_idx]
                    gdp_timeseries = gdp_data[:, lat_idx, lon_idx]

                    # Calculate baseline TFP and capital stock (no climate effects)
                    tfp_cell, k_cell = calculate_tfp_coin_ssp(pop_timeseries, gdp_timeseries, params)

                    # Additional NaN check immediately after TFP calculation
                    if np.any(np.isnan(tfp_cell)) or np.any(np.isnan(k_cell)):
                        print(f"\n{'='*80}")
                        print(f"NaN DETECTED AFTER TFP CALCULATION - GRID CELL DIAGNOSIS")
                        print(f"{'='*80}")
                        print(f"Grid cell location: lat_idx={lat_idx}, lon_idx={lon_idx}")
                        print(f"SSP scenario: {ssp_name}")
                        print(f"Grid cells processed so far: {grid_cells_processed}")

                        print(f"INPUT TIME SERIES FOR THIS GRID CELL:")
                        print(f"  Population: {pop_timeseries}")
                        print(f"  GDP: {gdp_timeseries}")

                        print(f"TFP CALCULATION RESULTS:")
                        print(f"  TFP: {tfp_cell}")
                        print(f"  Capital: {k_cell}")
                        print(f"  TFP contains NaN: {np.any(np.isnan(tfp_cell))}")
                        print(f"  Capital contains NaN: {np.any(np.isnan(k_cell))}")

                        if np.any(np.isnan(tfp_cell)):
                            print(f"  TFP NaN indices: {np.where(np.isnan(tfp_cell))[0].tolist()}")
                        if np.any(np.isnan(k_cell)):
                            print(f"  Capital NaN indices: {np.where(np.isnan(k_cell))[0].tolist()}")

                        print(f"{'='*80}")
                        raise RuntimeError(f"NaN detected in TFP results at grid cell ({lat_idx}, {lon_idx})")

                    # Store results (output arrays are [time, lat, lon])
                    tfp_baseline[:, lat_idx, lon_idx] = tfp_cell
                    k_baseline[:, lat_idx, lon_idx] = k_cell
                    
                    grid_cells_processed += 1
        
        # grid_cells_processed should equal the pre-computed valid count
        valid_count = all_data['_metadata']['valid_count']
        print(f"  Processed {grid_cells_processed} valid grid cells out of {nlat * nlon} total")
        assert grid_cells_processed == valid_count, f"Processed {grid_cells_processed} != valid count {valid_count}"
        
        tfp_results[ssp_name] = {
            'tfp_baseline': tfp_baseline,
            'k_baseline': k_baseline, 
            'grid_cells_processed': grid_cells_processed,
            'valid_mask': valid_mask,
            'validation_passed': True
        }
        
    # Write results to NetCDF file with coordinate information
    model_name = config['climate_model']['model_name']
    reference_ssp = config['ssp_scenarios']['reference_ssp']
    output_path = get_step_output_path(output_dir, 2, config, reference_ssp, "baseline_tfp", "nc")
    
    # Add metadata for visualization (coordinates, years, and reference data for valid cell identification)
    metadata = all_data['_metadata']
    reference_ssp = config['ssp_scenarios']['reference_ssp']
   
    # Get years array from pre-computed metadata
    years = all_data['years']


    tfp_results['_metadata'] = {
        'lat': metadata['lat'],
        'lon': metadata['lon'],
        'years': years,
        'gdp_data': all_data[reference_ssp]['gdp'][0],  # Reference period GDP for valid cell mask
        'pop_data': all_data[reference_ssp]['pop'][0]  # Reference period population
    }

    # Keep the coordinates for NetCDF compatibility
    tfp_results['_coordinates'] = {
        'lat': metadata['lat'],
        'lon': metadata['lon']
    }
    
    save_step2_results_netcdf(tfp_results, output_path, config)

    # Generate TFP visualization
    print("Generating baseline TFP visualization...")
    visualization_path = create_baseline_tfp_visualization(tfp_results, config, output_dir, all_data)
    print(f"✅ Visualization saved: {visualization_path}")

    print(f"\nStep 2 completed: Baseline TFP calculated for {len(tfp_results)} SSP scenarios")
    return tfp_results


def step3_calculate_scaling_factors_per_cell(config: Dict[str, Any], target_results: Dict[str, Any],
                                           tfp_results: Dict[str, Any], output_dir: str,
                                           all_data: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Step 3: Calculate scaling factors for each grid cell (per-cell optimization for SSP245).
    
    This step optimizes scaling factors individually for each grid cell by:
    1. For each grid cell:
       - For each response function scaling configuration:
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
        Dictionary containing scaling factor arrays [lat, lon, response_func, target] 
    """
    print("\n" + "="*80)
    print("STEP 3: CALCULATING SCALING FACTORS PER GRID CELL (SSP245)")
    print("="*80)
    
    reference_ssp = config['ssp_scenarios']['reference_ssp']
    response_scalings = config['response_function_scalings']
    gdp_targets = config['gdp_targets']
    
    n_response_functions = len(response_scalings)
    n_gdp_targets = len(gdp_targets)
    total_combinations = n_response_functions * n_gdp_targets
    
    print(f"Reference SSP: {reference_ssp}")
    print(f"Response function scalings: {n_response_functions}")
    print(f"GDP reduction targets: {n_gdp_targets}")
    print(f"Total combinations per grid cell: {total_combinations}")
    
    # Use pre-loaded NetCDF data
    print(f"Using pre-loaded NetCDF data for {reference_ssp}...")
    all_data = all_data
    
    # Extract climate and population data
    tas_data = get_ssp_data(all_data, reference_ssp, 'tas')  # [lat, lon, time]
    pr_data = get_ssp_data(all_data, reference_ssp, 'pr')  # [lat, lon, time]
    pop_data = get_ssp_data(all_data, reference_ssp, 'pop')  # [lat, lon, time]
    gdp_data = get_ssp_data(all_data, reference_ssp, 'gdp')  # [lat, lon, time]

    # Get pre-computed weather variables for reference SSP
    tas_weather_data = all_data[reference_ssp]['tas_weather']  # [time, lat, lon]
    pr_weather_data = all_data[reference_ssp]['pr_weather']    # [time, lat, lon]
    
    # Get baseline TFP for reference SSP
    reference_tfp = tfp_results[reference_ssp]
    valid_mask = reference_tfp['valid_mask']
    tfp_baseline = reference_tfp['tfp_baseline']  # [time, lat, lon]
    
    # Get dimensions (tas_data is [time, lat, lon])
    ntime, nlat, nlon = tas_data.shape
    
    # Initialize scaling factor arrays 
    scaling_factors = np.full((nlat, nlon, n_response_functions, n_gdp_targets), np.nan)
    optimization_errors = np.full((nlat, nlon, n_response_functions, n_gdp_targets), np.nan)
    convergence_flags = np.full((nlat, nlon, n_response_functions, n_gdp_targets), False)
    
    # Initialize arrays for scaled response function parameters [lat, lon, response_func, target, param]
    # The 12 parameters are: k_tas1, k_tas2, k_pr1, k_pr2, tfp_tas1, tfp_tas2, tfp_pr1, tfp_pr2, y_tas1, y_tas2, y_pr1, y_pr2
    n_scaled_params = 12
    scaled_parameters = np.full((nlat, nlon, n_response_functions, n_gdp_targets, n_scaled_params), np.nan)
    
    # Define parameter names for NetCDF output
    scaled_param_names = ['k_tas1', 'k_tas2', 'k_pr1', 'k_pr2', 
                         'tfp_tas1', 'tfp_tas2', 'tfp_pr1', 'tfp_pr2',
                         'y_tas1', 'y_tas2', 'y_pr1', 'y_pr2']
    
    # Create base ModelParams instance using factory
    params = config['model_params_factory'].create_base()
    
    # Get years array from pre-computed metadata
    years = all_data['years']

    # Calculate reference period indices for tas0/pr0 baseline calculation
    time_periods = config['time_periods']
    ref_start_year = time_periods['reference_period']['start_year']
    ref_end_year = time_periods['reference_period']['end_year']

    ref_start_idx = np.where(years == ref_start_year)[0][0]
    ref_end_idx = np.where(years == ref_end_year)[0][0]
    
    # Initialize counters
    total_grid_cells = 0
    successful_optimizations = 0
    
    print("Starting per-grid-cell optimization...")
    print("This may take significant time for large grids...")
    print(f"Processing {nlat} x {nlon} = {nlat*nlon} grid cells")

    # Get reference climate baselines from all_data (computed by calculate_weather_vars)
    tas0_2d = all_data['tas0_2d']
    pr0_2d = all_data['pr0_2d']

    # Check if any targets are of type 'variability'
    has_variability_targets = any(target.get('target_type', 'damage') == 'variability' for target in gdp_targets)
    if has_variability_targets:
        print("Detected 'variability' type GDP targets. Preparing variability reference scaling...")
        # EXPENSIVE: Compute reference relationship once
        variability_reference_scaling = calculate_reference_gdp_climate_variability(
                    all_data, config, reference_tfp, response_scalings
                )

    # Process each GDP target with conditional logic based on target type
    for target_idx, gdp_target in enumerate(gdp_targets):
        target_type = gdp_target.get('target_type', 'damage')  # Default to 'damage' for backward compatibility

        if target_type == 'variability':

            if variability_reference_scaling is None:
                # This should never happen if has_variability_targets check worked
                raise RuntimeError("Variability reference scaling not computed but variability target encountered")

            # CHEAP: Apply target-specific scaling
            print(f"\nApplying variability target: {gdp_target['target_name']} ({target_idx+1}/{n_gdp_targets})")
            results = apply_variability_target_scaling(
                variability_reference_scaling, gdp_target, tas_data, pr_data,
                tas0_2d, pr0_2d, target_idx, response_scalings,
                scaling_factors, optimization_errors, convergence_flags, scaled_parameters
            )

        else:  # target_type == 'damage'

            # EXPENSIVE: Separate optimization for each damage target
            results = process_response_target_optimization(
                target_idx, gdp_target, target_results, response_scalings,
                tas_data, pr_data, pop_data, gdp_data,
                reference_tfp, valid_mask, tfp_baseline, years, config,
                scaling_factors, optimization_errors, convergence_flags, scaled_parameters,
                total_grid_cells, successful_optimizations,
                tas_weather_data, pr_weather_data
            )

            # Update counters from damage optimization
            total_grid_cells = results['total_grid_cells']
            successful_optimizations = results['successful_optimizations']

    print(f"\nOptimization completed:")
    print(f"  Total valid grid cells processed: {total_grid_cells}")  
    print(f"  Successful optimizations: {successful_optimizations}")
    print(f"  Success rate: {100*successful_optimizations/max(1, total_grid_cells):.1f}%")
    
    # Create results dictionary
    scaling_results = {
        'scaling_factors': scaling_factors,       # [lat, lon, response_func_idx, target_idx] 
        'optimization_errors': optimization_errors, # [lat, lon, response_func_idx, target_idx]
        'convergence_flags': convergence_flags,   # [lat, lon, response_func_idx, target_idx] boolean
        'scaled_parameters': scaled_parameters,   # [lat, lon, response_func_idx, target_idx, param_idx]
        'scaled_param_names': scaled_param_names, # Parameter names for the last dimension
        'response_function_names': [df['scaling_name'] for df in response_scalings],
        'target_names': [tgt['target_name'] for tgt in gdp_targets], 
        'total_grid_cells': total_grid_cells,
        'successful_optimizations': successful_optimizations,
        'reference_ssp': reference_ssp,
        'valid_mask': valid_mask,  # [lat, lon] boolean mask
        '_coordinates': all_data['_metadata']  # coordinate information
    }
    
    # Write results to NetCDF file
    output_path = get_step_output_path(output_dir, 3, config, reference_ssp, "scaling_factors", "nc")
    save_step3_results_netcdf(scaling_results, output_path, config)

    # Generate scaling factors visualization
    print("Generating scaling factors visualization...")
    visualization_path = create_scaling_factors_visualization(scaling_results, config, output_dir)
    print(f"✅ Scaling factors visualization saved: {visualization_path}")

    # Generate objective function visualization
    print("Generating objective function visualization...")
    obj_func_path = create_objective_function_visualization(scaling_results, config, output_dir)
    print(f"✅ Objective function visualization saved: {obj_func_path}")

    # Display GDP-weighted scaling factor summary
    print_gdp_weighted_scaling_summary(scaling_results, config, all_data, output_dir)

    print(f"\nStep 3 completed: Scaling factors calculated for each grid cell")
    print(f"Total combinations processed: {total_combinations} per valid grid cell")
    return scaling_results


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

        # Also display table in terminal with updated columns
        print("Summary Table (also saved to CSV):")
        print(f"{'Target':<35} {'Response Function':<20} {'GDP-Wtd Mean':<13} {'GDP-Wtd Median':<15} {'Max':<10} {'Min':<10} {'ObjFunc Max':<11} {'ObjFunc Mean':<12} {'ObjFunc Std':<11} {'ObjFunc Min':<11}")
        print("-" * 158)

        for row in csv_data:
            print(f"{row['target_name']:<35} {row['response_function']:<20} "
                  f"{row['gdp_weighted_mean']:<13.6f} {row['gdp_weighted_median']:<15.6f} "
                  f"{row['scaling_max']:<10.6f} {row['scaling_min']:<10.6f} "
                  f"{row['obj_func_max']:<11.6f} {row['obj_func_mean']:<12.6f} "
                  f"{row['obj_func_std']:<11.6f} {row['obj_func_min']:<11.6f}")

        print("-" * 158)
    else:
        # Fallback: print to terminal only (old format)
        print(f"{'Target':<35} {'Response Function':<20} {'GDP-Wtd Mean':<13} {'GDP-Wtd Median':<15} {'Max':<10} {'Min':<10} {'ObjFunc Max':<11} {'ObjFunc Mean':<12} {'ObjFunc Std':<11} {'ObjFunc Min':<11}")
        print("-" * 158)

        for row in csv_data:
            print(f"{row['target_name']:<35} {row['response_function']:<20} "
                  f"{row['gdp_weighted_mean']:<13.6f} {row['gdp_weighted_median']:<15.6f} "
                  f"{row['scaling_max']:<10.6f} {row['scaling_min']:<10.6f} "
                  f"{row['obj_func_max']:<11.6f} {row['obj_func_mean']:<12.6f} "
                  f"{row['obj_func_std']:<11.6f} {row['obj_func_min']:<11.6f}")

        print("-" * 158)

    print()


def step4_forward_integration_all_ssps(config: Dict[str, Any], scaling_results: Dict[str, Any],
                                     tfp_results: Dict[str, Any], output_dir: str, all_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Step 4: Run forward model for all SSPs using per-cell scaling factors.
    
    This step applies the forward economic model by:
    1. For each forward simulation SSP:
       - Load climate data (temperature, precipitation)
       - For each response function and target combination:
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
    n_response_functions = len(config['response_function_scalings'])
    n_gdp_targets = len(config['gdp_targets'])
    total_combinations = n_response_functions * n_gdp_targets
    
    print(f"Processing {len(forward_ssps)} SSP scenarios")
    print(f"Using {total_combinations} scaling factor combinations per grid cell")
    print(f"Total processing: {len(forward_ssps)} SSPs × {total_combinations} combinations")
    
    # Use pre-loaded climate and economic data
    print("Using pre-loaded NetCDF data for forward modeling...")
    all_data = all_data
    
    # Create base ModelParams instance using factory
    base_params = config['model_params_factory'].create_base()
    response_function_names = scaling_results['response_function_names']
    target_names = scaling_results['target_names']
    valid_mask = scaling_results['valid_mask']
    scaling_factors = scaling_results['scaling_factors']  # [lat, lon, response_func, target]
    scaled_parameters = scaling_results['scaled_parameters']  # [lat, lon, response_func, target, param]
    
    # Get grid dimensions
    nlat, nlon = valid_mask.shape
    
    years = all_data['years']
    time_periods = config['time_periods']
    
    forward_results = {}
    
    for i, ssp_name in enumerate(forward_ssps):
        print(f"\nProcessing SSP scenario: {ssp_name} ({i+1}/{len(forward_ssps)})")
        
        # Get SSP-specific data
        tas_data = get_ssp_data(all_data, ssp_name, 'tas')  # [time, lat, lon]
        pr_data = get_ssp_data(all_data, ssp_name, 'pr') # [time, lat, lon]
        pop_data = get_ssp_data(all_data, ssp_name, 'pop')  # [time, lat, lon]
        gdp_data = get_ssp_data(all_data, ssp_name, 'gdp')  # [time, lat, lon]

        # Get pre-computed weather variables
        tas_weather_data = all_data[ssp_name]['tas_weather']  # [time, lat, lon]
        pr_weather_data = all_data[ssp_name]['pr_weather']    # [time, lat, lon]

        # Get baseline TFP for this SSP from Step 2
        tfp_baseline = tfp_results[ssp_name]['tfp_baseline']  # [time, lat, lon]

        ntime = tas_data.shape[0]  # Time is first dimension
        years = all_data['years']

        # Calculate reference period indices for tas0/pr0 baseline calculation
        ref_start_year = time_periods['reference_period']['start_year']
        ref_end_year = time_periods['reference_period']['end_year']
        ref_start_idx = np.where(years == ref_start_year)[0][0]
        ref_end_idx = np.where(years == ref_end_year)[0][0]

        
        print(f"  Grid dimensions: {ntime} time x {nlat} lat x {nlon} lon")
        print(f"  Running forward model for {total_combinations} combinations per valid grid cell")
        
        # Initialize result arrays for this SSP
        # [lat, lon, response_func, target, time]
        gdp_climate = np.full((nlat, nlon, n_response_functions, n_gdp_targets, ntime), np.nan)
        gdp_weather = np.full((nlat, nlon, n_response_functions, n_gdp_targets, ntime), np.nan)
        tfp_climate = np.full((nlat, nlon, n_response_functions, n_gdp_targets, ntime), np.nan)
        tfp_weather = np.full((nlat, nlon, n_response_functions, n_gdp_targets, ntime), np.nan)
        k_climate = np.full((nlat, nlon, n_response_functions, n_gdp_targets, ntime), np.nan)
        k_weather = np.full((nlat, nlon, n_response_functions, n_gdp_targets, ntime), np.nan)
        
        successful_forward_runs = 0
        total_forward_runs = 0
        
        # Standard computational loop structure: target → damage → spatial
        for target_idx in range(n_gdp_targets):
            target_name = target_names[target_idx]
            print(f"    GDP reduction target: {target_name} ({target_idx+1}/{n_gdp_targets})")

            for response_idx in range(n_response_functions):
                response_name = response_function_names[response_idx]
                
                for lat_idx in range(nlat):
                    for lon_idx in range(nlon):
                        
                        # Check if grid cell is valid and has optimization results
                        if not valid_mask[lat_idx, lon_idx]:
                            continue
                        
                        # Check if scaling factor optimization was successful for this combination
                        if np.isnan(scaling_factors[lat_idx, lon_idx, response_idx, target_idx]):
                            continue
                            
                        total_forward_runs += 1
                        
                        # Extract time series for this grid cell (climate data is [time, lat, lon])
                        cell_tas = tas_data[:, lat_idx, lon_idx]  # [time]
                        cell_pr = pr_data[:, lat_idx, lon_idx]  # [time]
                        cell_pop = pop_data[:, lat_idx, lon_idx]  # [time]
                        cell_gdp = gdp_data[:, lat_idx, lon_idx]  # [time]
                        cell_tfp_baseline = tfp_baseline[:, lat_idx, lon_idx]  # [time] (data is [time, lat, lon])
                        
                        # Get pre-computed weather (filtered) time series
                        cell_tas_weather = tas_weather_data[:, lat_idx, lon_idx]  # [time]
                        cell_pr_weather = pr_weather_data[:, lat_idx, lon_idx]    # [time]
                        
                        # Create ModelParams with scaled response function parameters
                        params_scaled = copy.deepcopy(base_params)
                        params_scaled.tas0 = np.mean(cell_tas[ref_start_idx:ref_end_idx+1])
                        params_scaled.pr0 = np.mean(cell_pr[ref_start_idx:ref_end_idx+1])
                        
                        # Set scaled response function parameters from Step 3 results
                        params_scaled.k_tas1 = scaled_parameters[lat_idx, lon_idx, response_idx, target_idx, 0]
                        params_scaled.k_tas2 = scaled_parameters[lat_idx, lon_idx, response_idx, target_idx, 1]
                        params_scaled.k_pr1 = scaled_parameters[lat_idx, lon_idx, response_idx, target_idx, 2]
                        params_scaled.k_pr2 = scaled_parameters[lat_idx, lon_idx, response_idx, target_idx, 3]
                        params_scaled.tfp_tas1 = scaled_parameters[lat_idx, lon_idx, response_idx, target_idx, 4]
                        params_scaled.tfp_tas2 = scaled_parameters[lat_idx, lon_idx, response_idx, target_idx, 5]
                        params_scaled.tfp_pr1 = scaled_parameters[lat_idx, lon_idx, response_idx, target_idx, 6]
                        params_scaled.tfp_pr2 = scaled_parameters[lat_idx, lon_idx, response_idx, target_idx, 7]
                        params_scaled.y_tas1 = scaled_parameters[lat_idx, lon_idx, response_idx, target_idx, 8]
                        params_scaled.y_tas2 = scaled_parameters[lat_idx, lon_idx, response_idx, target_idx, 9]
                        params_scaled.y_pr1 = scaled_parameters[lat_idx, lon_idx, response_idx, target_idx, 10]
                        params_scaled.y_pr2 = scaled_parameters[lat_idx, lon_idx, response_idx, target_idx, 11]
                        
                        # Run forward model with climate data
                        y_climate, a_climate, k_climate_values, _, _, _ = calculate_coin_ssp_forward_model(
                            cell_tfp_baseline, cell_pop, cell_tas, cell_pr, params_scaled
                        )

                        # Run forward model with weather data (no climate trends)
                        y_weather, a_weather, k_weather_values, _, _, _ = calculate_coin_ssp_forward_model(
                            cell_tfp_baseline, cell_pop, cell_tas_weather, cell_pr_weather, params_scaled
                        )

                        # Store results (convert normalized values back to actual units)
                        gdp_climate[lat_idx, lon_idx, response_idx, target_idx, :] = y_climate * cell_gdp[0]
                        gdp_weather[lat_idx, lon_idx, response_idx, target_idx, :] = y_weather * cell_gdp[0]
                        tfp_climate[lat_idx, lon_idx, response_idx, target_idx, :] = a_climate
                        tfp_weather[lat_idx, lon_idx, response_idx, target_idx, :] = a_weather
                        k_climate[lat_idx, lon_idx, response_idx, target_idx, :] = k_climate_values
                        k_weather[lat_idx, lon_idx, response_idx, target_idx, :] = k_weather_values

                        successful_forward_runs += 1
        
        print(f"  Forward model completed:")
        print(f"    Total forward runs: {total_forward_runs}")
        print(f"    Successful runs: {successful_forward_runs}")
        print(f"    Success rate: {100*successful_forward_runs/max(1, total_forward_runs):.1f}%")
        
        # Store results for this SSP
        forward_results[ssp_name] = {
            'gdp_climate': gdp_climate,        # [lat, lon, response_func, target, time]
            'gdp_weather': gdp_weather,        # [lat, lon, response_func, target, time] 
            'tfp_climate': tfp_climate,        # [lat, lon, response_func, target, time]
            'tfp_weather': tfp_weather,        # [lat, lon, response_func, target, time]
            'k_climate': k_climate,            # [lat, lon, response_func, target, time]
            'k_weather': k_weather,            # [lat, lon, response_func, target, time]
            'successful_forward_runs': successful_forward_runs,
            'total_forward_runs': total_forward_runs,
            'success_rate': successful_forward_runs / max(1, total_forward_runs)
        }
    
    # Add metadata and coordinate information to results
    step4_results = {
        'forward_results': forward_results,  # SSP-specific results
        'response_function_names': response_function_names,
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
        '_coordinates': {
            **all_data['_metadata'],
            'years': all_data['years']
        }
    }
    
    # Write results to separate NetCDF files per SSP/variable
    model_name = config['climate_model']['model_name']
    saved_files = save_step4_results_netcdf_split(step4_results, output_dir, config)
    print(f"Step 4 NetCDF files saved: {len(saved_files)} files")

    # Create PDF visualizations
    print("Creating Step 4 PDF visualizations...")

    # Line plots visualization
    pdf_path = create_forward_model_visualization(step4_results, config, output_dir, all_data)
    print(f"Step 4 line plots saved to: {pdf_path}")

    # Ratio plots visualization
    ratio_pdf_path = create_forward_model_ratio_visualization(step4_results, config, output_dir, all_data)
    print(f"Step 4 ratio plots saved to: {ratio_pdf_path}")

    # Maps visualization (generates both linear and log10 scale PDFs)
    linear_maps_path, log10_maps_path = create_forward_model_maps_visualization(step4_results, config, output_dir, all_data)

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
    response_functions = config['response_function_scalings']
    gdp_targets = config['gdp_targets']
    
    print(f"Climate Model: {model_name}")
    print(f"Reference SSP: {reference_ssp}")
    print(f"Forward SSPs: {len(forward_ssps)} scenarios ({', '.join(forward_ssps)})")
    print(f"Damage Functions: {len(response_functions)} configurations")
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
        if ssp_name not in ['_coordinates', '_metadata']:
            print(f"     • {ssp_name}: {ssp_data['grid_cells_processed']} valid grid cells")
    
    # Step 3 Summary
    print(f"\nStep 3 - Scaling Factor Optimization:")
    print(f"  ✅ Optimized {scaling_results['total_grid_cells']} valid grid cells")
    print(f"     • Success rate: {100*scaling_results['successful_optimizations']/max(1, scaling_results['total_grid_cells']):.1f}%")
    print(f"     • Combinations per cell: {len(response_functions)} damage × {len(gdp_targets)} targets = {len(response_functions)*len(gdp_targets)}")
    
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


def run_pipeline(config_path: str, step3_file: str = None) -> None:
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
        Path to JSON configuration file
    """
    print("Starting COIN-SSP Integrated Processing Pipeline")
    print("Following README.md Section 3: Grid Cell Processing")
    print("="*100)
    
    # Start overall timing
    pipeline_start_time = time.time()
    step_times = {}

    try:
        # Load configuration
        config = load_config(config_path)

        # Extract json_id from configuration
        json_id = config['run_metadata']['json_id']

        # Setup output directory
        output_dir = setup_output_directory(config)

        # Copy config file to output directory for reproducibility
        config_filename = os.path.basename(config_path)
        config_copy_path = os.path.join(output_dir, config_filename)
        shutil.copy2(config_path, config_copy_path)
        print(f"Configuration file copied to: {config_copy_path}")

        # Load all NetCDF data once for efficiency (major optimization)
        data_start = time.time()
        print("Loading all NetCDF data for entire pipeline...")
        all_data = load_all_data(config, output_dir)
        print("✅ All NetCDF data loaded - will be reused across all processing steps")

        # Compute weather variables for all SSPs
        all_data = calculate_weather_vars(all_data, config)

        step_times['Data Loading'] = time.time() - data_start

        # Execute 5-step processing flow
        step1_start = time.time()
        target_results = step1_calculate_target_gdp_changes(config, output_dir, json_id, all_data)
        step_times['Step 1 - Target GDP'] = time.time() - step1_start

        step2_start = time.time()
        tfp_results = step2_calculate_baseline_tfp(config, output_dir, all_data)
        step_times['Step 2 - Baseline TFP'] = time.time() - step2_start

        # Step 3: Load from file or compute scaling factors
        step3_start = time.time()
        if step3_file:
            print(f"\n🔄 LOADING STEP 3 FROM FILE: {step3_file}")
            scaling_results = load_step3_results_from_netcdf(step3_file)

            # Create Step 3 visualizations even when loaded from file
            print("Creating Step 3 PDF visualizations from loaded data...")
            model_name = config['climate_model']['model_name']

            # Scaling factors visualization
            pdf_path = create_scaling_factors_visualization(scaling_results, config, output_dir)
            print(f"✅ Scaling factors visualization saved to: {pdf_path}")

            # Objective function visualization
            obj_func_path = create_objective_function_visualization(scaling_results, config, output_dir)
            print(f"✅ Objective function visualization saved to: {obj_func_path}")
            step_times['Step 3 - Scaling Factors (Loaded)'] = time.time() - step3_start
        else:
            scaling_results = step3_calculate_scaling_factors_per_cell(config, target_results, tfp_results, output_dir, all_data)
            step_times['Step 3 - Scaling Factors (Computed)'] = time.time() - step3_start

        step4_start = time.time()
        forward_results = step4_forward_integration_all_ssps(config, scaling_results, tfp_results, output_dir, all_data)
        step_times['Step 4 - Forward Integration'] = time.time() - step4_start

        step5_start = time.time()
        step5_processing_summary(config, target_results, tfp_results, scaling_results, forward_results)
        step_times['Step 5 - Processing Summary'] = time.time() - step5_start

        # Calculate total time and print timing report
        total_time = time.time() - pipeline_start_time

        print("\n" + "="*100)
        print("PIPELINE TIMING REPORT")
        print("="*100)
        for step_name, duration in step_times.items():
            print(f"{step_name:<35}: {duration:8.2f} seconds")
        print("-" * 50)
        print(f"{'TOTAL PIPELINE TIME':<35}: {total_time:8.2f} seconds")
        print("="*100)

        print("\n" + "="*100)
        print("INTEGRATED PROCESSING COMPLETE")
        print("="*100)
        print("All 5 steps executed successfully!")
        print("Per-grid-cell scaling factors calculated and applied to all SSP scenarios")

    except Exception as e:
        # Print partial timing report even on failure
        total_time = time.time() - pipeline_start_time
        print(f"\n" + "="*60)
        print("PARTIAL TIMING REPORT (before failure)")
        print("="*60)
        for step_name, duration in step_times.items():
            print(f"{step_name:<35}: {duration:8.2f} seconds")
        print("-" * 50)
        print(f"{'TOTAL TIME TO FAILURE':<35}: {total_time:8.2f} seconds")
        print("="*60)
        print(f"\nERROR: Processing failed with exception: {e}")
        raise


def main():
    """Main entry point for integrated processing pipeline."""

    parser = argparse.ArgumentParser(
        description="COIN-SSP Integrated Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py coin_ssp_config_0007.json
  python main.py config.json --step3-file data/output/previous_run/step3_scaling_factors_CanESM5.nc

This pipeline implements README.md Section 3: Grid Cell Processing
Key feature: Per-grid-cell scaling factor optimization using optimize_climate_response_scaling
        """
    )

    parser.add_argument('config', help='Path to configuration JSON file')
    parser.add_argument('--step3-file', help='Path to existing Step 3 NetCDF file to load instead of running optimization')

    args = parser.parse_args()

    run_pipeline(args.config, step3_file=args.step3_file)


if __name__ == "__main__":
    main()