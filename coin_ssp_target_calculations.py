import numpy as np
import xarray as xr
from typing import Dict, Any

from coin_ssp_math_utils import calculate_gdp_weighted_mean


def calculate_constant_target_reduction(gdp_amount_value, tas_ref_template):
    """
    Calculate constant GDP reduction across all grid cells.

    Parameters
    ----------
    gdp_amount_value : float
        Constant reduction value (e.g., -0.10 for 10% reduction)
    tas_ref_template : xr.DataArray
        Template array with target (lat, lon) coordinates

    Returns
    -------
    xr.DataArray
        Constant reduction array with same coordinates as template
    """
    return xr.full_like(tas_ref_template, gdp_amount_value, dtype=np.float64)


def calculate_linear_target_reduction(linear_config, valid_mask, all_data, reference_ssp, period_start, period_end):
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
    valid_mask : xr.DataArray
        Boolean mask for valid economic grid cells [lat, lon]
    all_data : dict
        Combined data structure containing xarray DataArrays
    reference_ssp : str
        Reference SSP scenario name
    period_start : int
        Start year of constraint period
    period_end : int
        End year of constraint period

    Returns
    -------
    dict
        Dictionary containing:
        - 'reduction_array': Linear reduction DataArray [lat, lon]
        - 'coefficients': {'a0': intercept, 'a1': slope}
        - 'constraint_verification': Verification of constraint satisfaction
    """
    # Extract configuration parameters
    global_mean_linear = linear_config['global_mean_amount']
    T_ref_linear = linear_config['reference_temperature']
    value_at_ref_linear = linear_config['amount_at_reference_temp']

    # Get time series data as xarray DataArrays
    tas_series = all_data[reference_ssp]['tas']
    gdp_series = all_data[reference_ssp]['gdp']

    # Calculate time-averaged temperature for constraint period
    from coin_ssp_math_utils import calculate_time_means
    tas_period = calculate_time_means(tas_series, period_start, period_end)

    # Select constraint period data using coordinate-based slicing
    tas_period_series = tas_series.sel(time=slice(period_start, period_end))
    gdp_period_series = gdp_series.sel(time=slice(period_start, period_end))

    # Apply valid mask (broadcasts automatically across time)
    tas_masked = tas_period_series.where(valid_mask)
    gdp_masked = gdp_period_series.where(valid_mask)

    # Calculate GDP-weighted temperature for each time step (vectorized)
    gdp_weighted_temps = (
        (gdp_masked * tas_masked).sum(dim=['lat', 'lon']) /
        gdp_masked.sum(dim=['lat', 'lon'])
    )

    # Mean over time of GDP-weighted temperature
    mean_gdp_weighted_temp = float(gdp_weighted_temps.mean(dim='time').values)

    # Solve for coefficients
    a1_linear = (global_mean_linear - value_at_ref_linear) / (mean_gdp_weighted_temp - T_ref_linear)
    a0_linear = value_at_ref_linear - a1_linear * T_ref_linear

    # Calculate linear reduction array (automatic broadcasting)
    linear_reduction = a0_linear + a1_linear * tas_period

    # Verify constraint satisfaction
    constraint1_check = a0_linear + a1_linear * T_ref_linear

    # Verify global mean constraint
    reduction_broadcast = linear_reduction.broadcast_like(tas_series)
    constraint2_check = calculate_gdp_weighted_mean(
        1 + reduction_broadcast,
        gdp_series, valid_mask, period_start, period_end
    ) - 1

    return {
        'reduction_array': linear_reduction.astype(np.float64),
        'coefficients': {
            'a0': float(a0_linear),
            'a1': float(a1_linear),
            'a2': 0.0
        },
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
        'gdp_weighted_tas_mean': float(mean_gdp_weighted_temp)
    }


def calculate_quadratic_target_reduction(quadratic_config, valid_mask, all_data, reference_ssp, period_start, period_end):
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
    valid_mask : xr.DataArray
        Boolean mask for valid economic grid cells [lat, lon]
    all_data : dict
        Combined data structure containing xarray DataArrays
    reference_ssp : str
        Reference SSP scenario name
    period_start : int
        Start year of constraint period
    period_end : int
        End year of constraint period

    Returns
    -------
    dict
        Dictionary containing:
        - 'reduction_array': Quadratic reduction DataArray [lat, lon]
        - 'coefficients': {'a0': constant, 'a1': linear, 'a2': quadratic}
        - 'constraint_verification': Verification of constraint satisfaction
    """
    # Extract configuration parameters
    global_mean_quad = quadratic_config['global_mean_amount']
    T0 = quadratic_config['zero_amount_temperature']
    derivative_at_T0 = quadratic_config['derivative_at_zero_amount_temperature']

    # Get time series data as xarray DataArrays
    tas_series = all_data[reference_ssp]['tas']
    gdp_series = all_data[reference_ssp]['gdp']

    # Calculate time-averaged temperature for constraint period
    from coin_ssp_math_utils import calculate_time_means
    tas_period = calculate_time_means(tas_series, period_start, period_end)

    # Select constraint period data using coordinate-based slicing
    tas_period_series = tas_series.sel(time=slice(period_start, period_end))
    gdp_period_series = gdp_series.sel(time=slice(period_start, period_end))

    # Apply valid mask (broadcasts automatically across time)
    tas_masked = tas_period_series.where(valid_mask)
    gdp_masked = gdp_period_series.where(valid_mask)

    # Calculate GDP-weighted T and T² for each time step (vectorized)
    tas2_masked = tas_masked ** 2

    gdp_weighted_temps = (
        (gdp_masked * tas_masked).sum(dim=['lat', 'lon']) /
        gdp_masked.sum(dim=['lat', 'lon'])
    )

    gdp_weighted_temps2 = (
        (gdp_masked * tas2_masked).sum(dim=['lat', 'lon']) /
        gdp_masked.sum(dim=['lat', 'lon'])
    )

    # Mean over time
    mean_gdp_weighted_temp = float(gdp_weighted_temps.mean(dim='time').values)
    mean_gdp_weighted_temp2 = float(gdp_weighted_temps2.mean(dim='time').values)

    # Solve for coefficients
    denominator = T0**2 - 2*T0*mean_gdp_weighted_temp + mean_gdp_weighted_temp2
    a2_quad = (global_mean_quad - derivative_at_T0*(mean_gdp_weighted_temp - T0)) / denominator
    a1_quad = derivative_at_T0 - 2*a2_quad*T0
    a0_quad = -derivative_at_T0*T0 + a2_quad*T0**2

    # Calculate quadratic reduction array (automatic broadcasting)
    quadratic_reduction = a0_quad + a1_quad * tas_period + a2_quad * tas_period**2

    # Verify constraint satisfaction
    constraint1_check = a0_quad + a1_quad * T0 + a2_quad * T0**2
    constraint2_check = a1_quad + 2 * a2_quad * T0

    # Verify global mean constraint
    reduction_broadcast = quadratic_reduction.broadcast_like(tas_series)
    constraint3_check = calculate_gdp_weighted_mean(
        1 + reduction_broadcast,
        gdp_series, valid_mask, period_start, period_end
    ) - 1

    return {
        'reduction_array': quadratic_reduction.astype(np.float64),
        'coefficients': {
            'a0': float(a0_quad),
            'a1': float(a1_quad),
            'a2': float(a2_quad)
        },
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
        'gdp_weighted_tas_mean': float(mean_gdp_weighted_temp),
        'gdp_weighted_tas2_mean': float(mean_gdp_weighted_temp2),
        'derivative_at_zero_tas': float(derivative_at_T0),
        'zero_amount_temperature': float(T0)
    }


def calculate_all_target_reductions(target_configs, gridded_data, all_data, reference_ssp, config):
    """
    Calculate all configured target GDP reductions using gridded data.

    This function processes multiple target configurations and automatically
    determines the reduction type from available parameters.

    Parameters
    ----------
    target_configs : list
        List of target configuration dictionaries
    gridded_data : dict
        Dictionary containing gridded xarray DataArrays:
        - 'tas_ref': Reference period temperature DataArray [lat, lon]
    all_data : dict
        Combined data structure containing xarray DataArrays
    reference_ssp : str
        Reference SSP scenario name
    config : dict
        Configuration containing time period definitions

    Returns
    -------
    dict
        Dictionary with target_name keys, each containing:
        - 'reduction_array': Calculated reduction DataArray [lat, lon]
        - 'coefficients': Function coefficients
        - 'constraint_verification': Constraint satisfaction results
    """
    results = {}

    valid_mask = all_data['_metadata']['valid_mask']

    # Get time period bounds
    target_period_start = config['time_periods']['target_period']['start_year']
    target_period_end = config['time_periods']['target_period']['end_year']
    historical_period_start = config['time_periods']['historical_period']['start_year']
    historical_period_end = config['time_periods']['historical_period']['end_year']

    # Get template for constant targets
    tas_ref = gridded_data['tas_ref']

    for target_config in target_configs:
        target_name = target_config['target_name']
        target_shape = target_config['target_shape']
        target_type = target_config.get('target_type', 'damage')

        # Determine constraint period based on target type
        if target_type == 'variability':
            period_start = historical_period_start
            period_end = historical_period_end
        else:  # damage
            period_start = target_period_start
            period_end = target_period_end

        if target_shape == 'constant':
            # Constant reduction
            reduction_array = calculate_constant_target_reduction(
                target_config['global_mean_amount'], tas_ref
            )
            result = {
                'target_shape': target_shape,
                'reduction_array': reduction_array,
                'coefficients': {
                    'a0': float(target_config['global_mean_amount']),
                    'a1': 0.0,
                    'a2': 0.0
                },
                'constraint_verification': None,
                'global_statistics': {
                    'gdp_weighted_mean': target_config['global_mean_amount']
                }
            }

        elif target_shape == 'quadratic':
            # Quadratic reduction
            result = calculate_quadratic_target_reduction(target_config, valid_mask,
                                                         all_data, reference_ssp, period_start, period_end)
            result['target_shape'] = target_shape

        elif target_shape == 'linear':
            # Linear reduction
            result = calculate_linear_target_reduction(target_config, valid_mask,
                                                      all_data, reference_ssp, period_start, period_end)
            result['target_shape'] = target_shape

        else:
            raise ValueError(f"Unknown target_shape '{target_shape}' for target '{target_name}'")

        results[target_name] = result

    return results