import copy
import json
import os
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime
from typing import Dict, Any, List


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

    # Map data types to config keys
    prefix_mapping = {
        'tas': 'tas_file_prefix',
        'pr': 'pr_file_prefix',
        'gdp': 'gdp_file_prefix',
        'pop': 'pop_file_prefix',
        'target_reductions': 'target_reductions_file_prefix'
    }

    prefix_key = prefix_mapping[data_type]
    prefix = climate_model['file_prefixes'][prefix_key]

    filename = f"{prefix}_{model_name}_{ssp_name}.nc"
    filepath = os.path.join(input_dir, filename)

    return filepath


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

    print(f"Loading Step 3 results from: {netcdf_path}")

    # Load NetCDF file
    ds = xr.open_dataset(netcdf_path)

    # Extract arrays
    scaling_factors = ds.scaling_factors.values
    optimization_errors = ds.optimization_errors.values
    convergence_flags = ds.convergence_flags.values
    scaled_parameters = ds.scaled_parameters.values
    valid_mask = ds.valid_mask.values

    # Extract coordinate labels
    response_function_names = [str(name) for name in ds.response_func.values]

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
            'tas_ref': (['lat', 'lon'], metadata['tas_ref']),
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

    ds.tas_ref.attrs = {
        'long_name': 'Reference period temperature',
        'units': '°C'
    }

    ds.gdp_target.attrs = {
        'long_name': 'Target period GDP',
        'units': 'economic units'
    }

    # Add global attributes
    serializable_config = create_serializable_config(config)
    ds.attrs = {
        'title': 'COIN-SSP Target GDP Reductions - Step 1 Results',
        'reference_ssp': metadata['reference_ssp'],
        'reference_period': f"{metadata['time_periods']['reference_period']['start_year']}-{metadata['time_periods']['reference_period']['end_year']}",
        'target_period': f"{metadata['time_periods']['target_period']['start_year']}-{metadata['time_periods']['target_period']['end_year']}",
        'global_tas_ref': metadata['global_tas_ref'],
        'global_gdp_target': metadata['global_gdp_target'],
        'creation_date': datetime.now().isoformat(),
        'configuration_json': json.dumps(serializable_config, indent=2)
    }

    # Save to file
    ds.to_netcdf(output_path)
    print(f"Step 1 results saved to {output_path}")
    return output_path


def save_step2_results_netcdf(tfp_results: Dict[str, Any], output_path: str, config: Dict[str, Any], all_data: Dict[str, Any]) -> str:
    """
    Save Step 2 baseline TFP results to NetCDF file.

    Parameters
    ----------
    tfp_results : Dict[str, Any]
        Results from step2_calculate_baseline_tfp()
    output_path : str
        Complete output file path
    config : Dict[str, Any]
        Full configuration dictionary to embed in NetCDF file

    Returns
    -------
    str
        Path to saved NetCDF file
    """

    # Extract model name from config
    model_name = config['climate_model']['model_name']

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
        valid_masks[i] = all_data['_metadata']['valid_mask']

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


def save_step3_results_netcdf(scaling_results: Dict[str, Any], output_path: str, config: Dict[str, Any], all_data: Dict[str, Any]) -> str:
    """
    Save Step 3 scaling factor results to NetCDF file.

    Parameters
    ----------
    scaling_results : Dict[str, Any]
        Results from step3_calculate_scaling_factors_per_cell()
    output_path : str
        Complete output file path
    config : Dict[str, Any]
        Full configuration dictionary to embed in NetCDF file

    Returns
    -------
    str
        Path to saved NetCDF file
    """

    # Extract model name from config
    model_name = config['climate_model']['model_name']

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Extract data arrays and metadata
    scaling_factors = scaling_results['scaling_factors']  # [response_func, target, lat, lon]
    optimization_errors = scaling_results['optimization_errors']  # [response_func, target, lat, lon]
    convergence_flags = scaling_results['convergence_flags']  # [response_func, target, lat, lon]
    scaled_parameters = scaling_results['scaled_parameters']  # [response_func, target, param, lat, lon]
    valid_mask = all_data['_metadata']['valid_mask']  # [lat, lon]

    # Get dimensions and coordinate info
    n_response_func, n_target, nlat, nlon = scaling_factors.shape
    n_scaled_params = scaled_parameters.shape[2]
    response_function_names = scaling_results['response_function_names']
    target_names = scaling_results['target_names']
    scaled_param_names = scaling_results['scaled_param_names']
    coordinates = scaling_results['_coordinates']

    # Arrays are already in the correct format [response_func, target, lat, lon]
    scaling_factors_t = scaling_factors  # [response_func, target, lat, lon]
    optimization_errors_t = optimization_errors  # [response_func, target, lat, lon]
    convergence_flags_t = convergence_flags  # [response_func, target, lat, lon]
    scaled_parameters_t = scaled_parameters  # [response_func, target, param, lat, lon]

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
        'description': 'Optimized scaling factors for climate response functions per grid cell'
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
        'long_name': 'Scaled climate response function parameters',
        'units': 'various',
        'description': 'Climate response function parameters (scaling_factor × base_parameter) for each grid cell and combination',
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


def save_step4_results_netcdf_split(step4_results: Dict[str, Any], output_dir: str, config: Dict[str, Any], all_data: Dict[str, Any]) -> List[str]:
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

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Extract model name from config
    model_name = config['climate_model']['model_name']

    # Extract metadata and structure
    forward_results = step4_results['forward_results']
    response_function_names = step4_results['response_function_names']
    target_names = step4_results['target_names']
    valid_mask = all_data['_metadata']['valid_mask']
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
            # Reorder coordinates from [response_func, target, time, lat, lon]
            # to [target, response_func, time, lat, lon]
            climate_data = ssp_result[climate_var].transpose(1, 0, 2, 3, 4)  # [target, response_func, time, lat, lon]
            weather_data = ssp_result[weather_var].transpose(1, 0, 2, 3, 4)  # [target, response_func, time, lat, lon]

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

            # Extract configuration values for standardized naming
            json_id = config['run_metadata']['json_id']
            model_name = config['climate_model']['model_name']

            # Generate filename using standardized pattern
            filename = f"step4_{json_id}_{model_name}_{ssp_name}_forward_{var_base}.nc"
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


def write_all_loaded_data_netcdf(all_data: Dict[str, Any], config: Dict[str, Any], output_dir: str) -> str:
    """
    Write all loaded NetCDF data to a single output file for reference and validation.

    Parameters
    ----------
    all_data : Dict[str, Any]
        All loaded NetCDF data from load_all_data()
    config : Dict[str, Any]
        Configuration dictionary
    output_dir : str
        Output directory path

    Returns
    -------
    str
        Path to written NetCDF file
    """

    # Extract metadata
    metadata = all_data['_metadata']
    lat = metadata['lat']
    lon = metadata['lon']
    years = all_data['years']  # Years stored at top level, not in metadata
    valid_mask = all_data['_metadata']['valid_mask']
    model_name = config['climate_model']['model_name']

    # Get SSP list (excluding metadata and years)
    ssp_names = [key for key in all_data.keys() if key not in ['_metadata', 'years']]

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
    tas_all = np.full((n_ssp, n_time, n_lat, n_lon), np.nan)
    pr_all = np.full((n_ssp, n_time, n_lat, n_lon), np.nan)
    gdp_all = np.full((n_ssp, n_time, n_lat, n_lon), np.nan)
    pop_all = np.full((n_ssp, n_time, n_lat, n_lon), np.nan)

    # Fill arrays
    for i, ssp_name in enumerate(ssp_names):
        ssp_data = all_data[ssp_name]
        tas_all[i] = ssp_data['tas']      # [time, lat, lon]
        pr_all[i] = ssp_data['pr']  # [time, lat, lon]
        gdp_all[i] = ssp_data['gdp']                      # [time, lat, lon]
        pop_all[i] = ssp_data['pop']        # [time, lat, lon]

    # Create xarray Dataset
    ds = xr.Dataset(
        {
            'tas': (['ssp', 'time', 'lat', 'lon'], tas_all),
            'pr': (['ssp', 'time', 'lat', 'lon'], pr_all),
            'gdp': (['ssp', 'time', 'lat', 'lon'], gdp_all),
            'pop': (['ssp', 'time', 'lat', 'lon'], pop_all),
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
    ds.tas.attrs = {
        'long_name': 'Surface Air Temperature',
        'units': 'degrees_celsius',
        'description': 'Annual surface air temperature, converted from Kelvin'
    }

    ds.pr.attrs = {
        'long_name': 'Precipitation Rate',
        'units': 'mm/day',
        'description': 'Annual precipitation rate'
    }

    ds.gdp.attrs = {
        'long_name': 'GDP Density',
        'units': 'economic_units',
        'description': 'GDP density with exponential growth applied before prediction year'
    }

    ds.pop.attrs = {
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