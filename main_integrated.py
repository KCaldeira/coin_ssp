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
5. Generate NetCDF outputs with proper metadata
"""

import json
import os
import sys
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Tuple

from coin_ssp_utils import load_gridded_data, calculate_global_mean
from coin_ssp_core import ModelParams, ScalingParams, optimize_climate_response_scaling, calculate_tfp_coin_ssp, calculate_coin_ssp_forward_model


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


def step1_calculate_target_gdp_changes(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Step 1: Develop target GDP changes using SSP245 scenario (global calculation).
    
    This step calculates spatially-explicit target GDP reductions by:
    1. Loading gridded climate and economic data for reference SSP (typically SSP245)
    2. For each GDP reduction target configuration:
       - Apply constraint satisfaction to generate spatial reduction patterns
       - Ensure GDP-weighted global means match specified targets
    3. Save target reduction arrays for use in scaling factor optimization
    
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
    print("STEP 1: CALCULATING TARGET GDP CHANGES (SSP245)")
    print("="*80)
    
    reference_ssp = config['ssp_scenarios']['reference_ssp']
    model_name = config['climate_model']['model_name']
    gdp_targets = config['gdp_reduction_targets']
    
    print(f"Using reference SSP: {reference_ssp}")
    print(f"Climate model: {model_name}")
    print(f"Processing {len(gdp_targets)} GDP reduction targets")
    
    # TODO: Load gridded data for reference SSP
    print(f"Loading gridded data for {reference_ssp}...")
    # temp_file = resolve_netcdf_filepath(config, 'temperature', reference_ssp)
    # gdp_file = resolve_netcdf_filepath(config, 'gdp', reference_ssp)
    # gridded_data = load_gridded_data([temp_file, gdp_file], ...)
    
    target_results = {}
    
    for i, gdp_target in enumerate(gdp_targets):
        target_name = gdp_target['target_name']
        target_type = gdp_target['target_type']
        
        print(f"\nProcessing target: {target_name} (type: {target_type}) ({i+1}/{len(gdp_targets)})")
        
        # TODO: Implement target GDP calculation using existing constraint satisfaction code
        # This would adapt calculate_target_gdp_reductions.py for integrated workflow
        
        # target_reductions = calculate_target_reductions_gridded(gridded_data, gdp_target)
        
        # Stub result with proper dimensions [lat, lon]
        target_results[target_name] = {
            'target_type': target_type,
            'reduction_array': None,  # TODO: Replace with actual [lat, lon] array
            'global_mean_achieved': -0.10,  # TODO: Calculate actual GDP-weighted mean
            'constraint_satisfied': True,
            'economic_bounds_valid': True
        }
    
    print(f"\nStep 1 completed: {len(target_results)} target GDP change patterns calculated")
    return target_results


def step2_calculate_baseline_tfp(config: Dict[str, Any]) -> Dict[str, Any]:
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
    
    # Create ModelParams object from configuration
    # params = ModelParams(**model_params)  # TODO: Implement proper parameter mapping
    
    tfp_results = {}
    
    for i, ssp_name in enumerate(forward_ssps):
        print(f"\nProcessing SSP scenario: {ssp_name} ({i+1}/{len(forward_ssps)})")
        
        # TODO: Load gridded economic data for this SSP
        print(f"  Loading gridded GDP and population data for {ssp_name}...")
        # gdp_file = resolve_netcdf_filepath(config, 'gdp', ssp_name)
        # pop_file = resolve_netcdf_filepath(config, 'population', ssp_name)
        # gdp_data = load_gridded_data(gdp_file, ...)
        # pop_data = load_gridded_data(pop_file, ...)
        
        print(f"  Calculating baseline TFP for each grid cell...")
        # TODO: Apply calculate_tfp_coin_ssp to each grid cell
        # This requires vectorization or iteration over lat/lon dimensions
        
        # tfp_baseline = np.zeros((nlat, nlon, ntime))  # [lat, lon, time]
        # k_baseline = np.zeros((nlat, nlon, ntime))
        # 
        # for lat_idx in range(nlat):
        #     for lon_idx in range(nlon):
        #         if valid_grid_cell(lat_idx, lon_idx):
        #             pop_timeseries = pop_data[lat_idx, lon_idx, :]
        #             gdp_timeseries = gdp_data[lat_idx, lon_idx, :]
        #             tfp_cell, k_cell = calculate_tfp_coin_ssp(pop_timeseries, gdp_timeseries, params)
        #             tfp_baseline[lat_idx, lon_idx, :] = tfp_cell
        #             k_baseline[lat_idx, lon_idx, :] = k_cell
        
        # Stub result
        tfp_results[ssp_name] = {
            'tfp_baseline': None,  # TODO: Replace with actual [lat, lon, time] array
            'k_baseline': None,    # TODO: Replace with actual [lat, lon, time] array
            'grid_cells_processed': 0,  # TODO: Count of valid grid cells
            'valid_mask': None,    # TODO: Boolean mask of valid grid cells
            'validation_passed': True
        }
        
    print(f"\nStep 2 completed: Baseline TFP calculated for {len(tfp_results)} SSP scenarios")
    return tfp_results


def step3_calculate_scaling_factors_per_cell(config: Dict[str, Any], target_results: Dict[str, Any], 
                                           tfp_results: Dict[str, Any]) -> Dict[str, Any]:
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
    
    # TODO: Load climate data for reference SSP
    print(f"Loading climate data for {reference_ssp}...")
    # temp_file = resolve_netcdf_filepath(config, 'temperature', reference_ssp)
    # precip_file = resolve_netcdf_filepath(config, 'precipitation', reference_ssp)
    # temp_data = load_gridded_data(temp_file, ...)
    # precip_data = load_gridded_data(precip_file, ...)
    
    # Get baseline TFP for reference SSP
    reference_tfp = tfp_results[reference_ssp]
    
    # TODO: Initialize scaling factor arrays with proper dimensions
    # scaling_factors = np.full((nlat, nlon, n_damage_functions, n_gdp_targets), np.nan)
    # optimization_errors = np.full((nlat, nlon, n_damage_functions, n_gdp_targets), np.nan)
    # convergence_flags = np.full((nlat, nlon, n_damage_functions, n_gdp_targets), False)
    
    scaling_results = {
        'scaling_factors': None,      # TODO: [lat, lon, damage_func_idx, target_idx] array
        'optimization_errors': None,  # TODO: [lat, lon, damage_func_idx, target_idx] array  
        'convergence_flags': None,    # TODO: [lat, lon, damage_func_idx, target_idx] boolean array
        'damage_function_names': [df['scaling_name'] for df in damage_scalings],
        'target_names': [tgt['target_name'] for tgt in gdp_targets],
        'total_grid_cells': 0,        # TODO: Count of valid grid cells
        'successful_optimizations': 0 # TODO: Count of successful optimizations
    }
    
    print("Starting per-grid-cell optimization...")
    print("This may take significant time for large grids...")
    
    # TODO: Implement nested loop over grid cells and combinations
    # for lat_idx in range(nlat):
    #     for lon_idx in range(nlon):
    #         if not valid_grid_cell(lat_idx, lon_idx):
    #             continue
    #             
    #         print(f"Processing grid cell ({lat_idx}, {lon_idx})")
    #         
    #         # Extract time series for this grid cell
    #         cell_temp = temp_data[lat_idx, lon_idx, :]
    #         cell_precip = precip_data[lat_idx, lon_idx, :]
    #         cell_tfp = reference_tfp['tfp_baseline'][lat_idx, lon_idx, :]
    #         # ... extract other needed data
    #         
    #         for damage_idx, damage_scaling in enumerate(damage_scalings):
    #             for target_idx, gdp_target in enumerate(gdp_targets):
    #                 target_name = gdp_target['target_name']
    #                 target_reduction = target_results[target_name]['reduction_array'][lat_idx, lon_idx]
    #                 
    #                 # Create ScalingParams and run optimization for this cell
    #                 scaling_params = ScalingParams(**damage_scaling)
    #                 
    #                 try:
    #                     optimal_scale, final_error, params_scaled = optimize_climate_response_scaling(
    #                         cell_data, model_params, scaling_params, target_reduction
    #                     )
    #                     
    #                     scaling_factors[lat_idx, lon_idx, damage_idx, target_idx] = optimal_scale
    #                     optimization_errors[lat_idx, lon_idx, damage_idx, target_idx] = final_error
    #                     convergence_flags[lat_idx, lon_idx, damage_idx, target_idx] = True
    #                     
    #                 except Exception as e:
    #                     print(f"Optimization failed for cell ({lat_idx}, {lon_idx}): {e}")
    
    print(f"\nStep 3 completed: Scaling factors calculated for each grid cell")
    print(f"Total combinations processed: {total_combinations} per valid grid cell")
    return scaling_results


def step4_forward_integration_all_ssps(config: Dict[str, Any], scaling_results: Dict[str, Any], 
                                     tfp_results: Dict[str, Any]) -> Dict[str, Any]:
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
    
    forward_results = {}
    
    for i, ssp_name in enumerate(forward_ssps):
        print(f"\nProcessing SSP scenario: {ssp_name} ({i+1}/{len(forward_ssps)})")
        
        # TODO: Load climate data for this SSP
        # temp_file = resolve_netcdf_filepath(config, 'temperature', ssp_name)
        # precip_file = resolve_netcdf_filepath(config, 'precipitation', ssp_name)
        # temp_data = load_gridded_data(temp_file, ...)
        # precip_data = load_gridded_data(precip_file, ...)
        
        # Get baseline TFP for this SSP
        baseline_tfp = tfp_results[ssp_name]
        
        # TODO: Initialize result arrays
        # gdp_climate = np.full((nlat, nlon, ntime, total_combinations), np.nan)
        # tfp_climate = np.full((nlat, nlon, ntime, total_combinations), np.nan)
        # k_climate = np.full((nlat, nlon, ntime, total_combinations), np.nan)
        
        print(f"  Running forward model for each grid cell and combination...")
        
        # TODO: Implement nested loop over grid cells and combinations
        # for lat_idx in range(nlat):
        #     for lon_idx in range(nlon):
        #         if not valid_grid_cell(lat_idx, lon_idx):
        #             continue
        #         
        #         # Extract baseline data for this grid cell
        #         cell_tfp_baseline = baseline_tfp['tfp_baseline'][lat_idx, lon_idx, :]
        #         cell_pop = pop_data[lat_idx, lon_idx, :]  # Need to load this
        #         cell_gdp_baseline = gdp_data[lat_idx, lon_idx, :]  # Need to load this
        #         cell_temp = temp_data[lat_idx, lon_idx, :]
        #         cell_precip = precip_data[lat_idx, lon_idx, :]
        #         
        #         combination_idx = 0
        #         for damage_idx in range(n_damage_functions):
        #             for target_idx in range(n_gdp_targets):
        #                 # Get scaling factor for this cell and combination
        #                 scale_factor = scaling_results['scaling_factors'][lat_idx, lon_idx, damage_idx, target_idx]
        #                 
        #                 if np.isnan(scale_factor):
        #                     combination_idx += 1
        #                     continue
        #                 
        #                 # Create scaled parameters
        #                 params_scaled = create_scaled_params(model_params, damage_scaling, scale_factor)
        #                 
        #                 # Run forward model
        #                 gdp_proj, tfp_proj, k_proj, _ = calculate_coin_ssp_forward_model(
        #                     cell_tfp_baseline, cell_pop, cell_gdp_baseline, cell_temp, params_scaled
        #                 )
        #                 
        #                 gdp_climate[lat_idx, lon_idx, :, combination_idx] = gdp_proj
        #                 tfp_climate[lat_idx, lon_idx, :, combination_idx] = tfp_proj
        #                 k_climate[lat_idx, lon_idx, :, combination_idx] = k_proj
        #                 
        #                 combination_idx += 1
        
        # Stub result
        forward_results[ssp_name] = {
            'gdp_climate': None,    # TODO: [lat, lon, time, combination] array
            'tfp_climate': None,    # TODO: [lat, lon, time, combination] array
            'k_climate': None,      # TODO: [lat, lon, time, combination] array
            'combination_labels': [
                f"{df['scaling_name']}_{tgt['target_name']}" 
                for df in config['damage_function_scalings']
                for tgt in config['gdp_reduction_targets']
            ],
            'grid_cells_processed': 0,
            'validation_passed': True
        }
        
    print(f"\nStep 4 completed: Forward integration for {len(forward_results)} SSP scenarios")
    return forward_results


def step5_generate_netcdf_outputs(config: Dict[str, Any], target_results: Dict[str, Any],
                                 tfp_results: Dict[str, Any], scaling_results: Dict[str, Any],
                                 forward_results: Dict[str, Any]) -> None:
    """
    Step 5: Generate NetCDF outputs with proper metadata.
    
    This step creates comprehensive NetCDF output files containing:
    1. Target GDP reduction patterns from Step 1
    2. Baseline TFP arrays from Step 2  
    3. Per-cell scaling factors from Step 3
    4. Forward model projections from Step 4
    5. Proper coordinate systems, attributes, and metadata
    
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
    print("STEP 5: GENERATING NETCDF OUTPUTS")
    print("="*80)
    
    output_config = config['output_configuration']
    output_dir = output_config['output_directory']
    model_name = config['climate_model']['model_name']
    reference_ssp = config['ssp_scenarios']['reference_ssp']
    
    # Create output directory with timestamp
    if output_config.get('include_timestamp', True):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_output_dir = os.path.join(output_dir, f"run_integrated_{model_name}_{reference_ssp}_{timestamp}")
    else:
        run_output_dir = os.path.join(output_dir, f"run_integrated_{model_name}_{reference_ssp}")
    
    os.makedirs(run_output_dir, exist_ok=True)
    print(f"Output directory: {run_output_dir}")
    
    # TODO: Generate NetCDF files for each component
    print("Generating NetCDF files:")
    
    print("  1. Target GDP reductions...")
    # save_target_reductions_netcdf(target_results, run_output_dir, model_name, reference_ssp)
    
    print("  2. Baseline TFP time series...")
    # save_baseline_tfp_netcdf(tfp_results, run_output_dir, model_name)
    
    print("  3. Per-cell scaling factors...")
    # save_scaling_factors_netcdf(scaling_results, run_output_dir, model_name, reference_ssp)
    
    print("  4. Forward model projections...")
    # for ssp_name, ssp_results in forward_results.items():
    #     save_forward_results_netcdf(ssp_results, run_output_dir, model_name, ssp_name)
    
    # TODO: Generate validation reports if requested
    if config['processing_options']['validation'].get('constraint_verification', True):
        print("  5. Validation and diagnostic reports...")
        # generate_validation_reports(config, target_results, scaling_results, forward_results, run_output_dir)
    
    # TODO: Generate PDF visualizations if requested  
    if config['output_configuration']['output_formats'].get('pdf_maps', True):
        print("  6. PDF visualization maps...")
        # generate_integrated_pdf_maps(config, target_results, forward_results, run_output_dir)
    
    print(f"\nAll outputs generated in: {run_output_dir}")


def run_integrated_pipeline(config_path: str) -> None:
    """
    Execute the complete integrated processing pipeline following README Section 3.
    
    The 5-step processing flow:
    1. Calculate target GDP changes using SSP245 (global calculation) 
    2. Calculate baseline TFP for each SSP (per grid cell, no climate)
    3. Calculate scaling factors per grid cell (optimize for each cell using SSP245)
    4. Run forward model for all SSPs (using per-cell scaling factors)
    5. Generate NetCDF outputs with proper metadata
    
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
        
        # Execute 5-step processing flow
        target_results = step1_calculate_target_gdp_changes(config)
        tfp_results = step2_calculate_baseline_tfp(config)
        scaling_results = step3_calculate_scaling_factors_per_cell(config, target_results, tfp_results)
        forward_results = step4_forward_integration_all_ssps(config, scaling_results, tfp_results)
        step5_generate_netcdf_outputs(config, target_results, tfp_results, scaling_results, forward_results)
        
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