import numpy as np
from typing import Dict, Any

from coin_ssp_math_utils import calculate_global_mean, calculate_gdp_weighted_mean


def calculate_constant_target_reduction(gdp_amount_value, tas_ref_shape):
    """
    Calculate constant GDP reduction across all grid cells.

    Parameters
    ----------
    gdp_amount_value : float
        Constant reduction value (e.g., -0.10 for 10% reduction)
    tas_ref_shape : tuple
        Shape of temperature reference array for output sizing

    Returns
    -------
    np.ndarray
        Constant reduction array with shape tas_ref_shape
    """
    return np.full(tas_ref_shape, gdp_amount_value, dtype=np.float64)


def calculate_linear_target_reduction(linear_config, lat, valid_mask, all_data, reference_ssp, period_start, period_end):
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
    lat : np.ndarray
        Latitude coordinate array for area weighting
    valid_mask : np.ndarray
        Boolean mask for valid economic grid cells [lat, lon]
    all_data : dict
        Combined data structure containing time series
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
        - 'reduction_array': Linear reduction array [lat, lon]
        - 'coefficients': {'a0': intercept, 'a1': slope}
        - 'constraint_verification': Verification of constraint satisfaction
    """
    # Extract configuration parameters
    global_mean_linear = linear_config['global_mean_amount']
    T_ref_linear = linear_config['reference_temperature']
    value_at_ref_linear = linear_config['amount_at_reference_temp']

    # Get time series data
    years = all_data['years']
    tas_series = all_data[reference_ssp]['tas']
    gdp_series = all_data[reference_ssp]['gdp']

    # Calculate time-averaged temperature for constraint period (for creating reduction spatial pattern)
    from coin_ssp_math_utils import calculate_time_means
    tas_period = calculate_time_means(tas_series, years, period_start, period_end)

    # Get time indices for constraint period
    period_mask = (years >= period_start) & (years <= period_end)
    tas_period_series = tas_series[period_mask]
    gdp_period_series = gdp_series[period_mask]

    # Calculate sums over time and space using valid mask
    # The reduction at time t is: reduction[t,lat,lon] = a0 + a1 * T[t,lat,lon]
    # Constraint: mean_over_time[ sum(GDP[t] * (1 + a0 + a1*T[t])) / sum(GDP[t]) ] = 1 + global_mean
    #
    # Expanding: mean_over_time[ (sum(GDP[t]) * (1+a0) + a1*sum(GDP[t]*T[t])) / sum(GDP[t]) ]
    #          = mean_over_time[ (1+a0) + a1*sum(GDP[t]*T[t])/sum(GDP[t]) ]
    #          = (1+a0) + a1*mean_over_time[sum(GDP[t]*T[t])/sum(GDP[t])]
    #
    # So: a0 + a1*mean_over_time[GDP-weighted T] = global_mean
    #
    # Also: a0 + a1*T_ref = value_at_ref
    #
    # Solving: a1 = (global_mean - value_at_ref) / (mean_over_time[GDP-weighted T] - T_ref)
    #          a0 = value_at_ref - a1*T_ref

    # Calculate GDP-weighted temperature for each time step
    gdp_weighted_temps = []
    for t in range(len(tas_period_series)):
        sum_gdp = np.sum(gdp_period_series[t][valid_mask])
        sum_gdp_tas = np.sum((gdp_period_series[t] * tas_period_series[t])[valid_mask])
        gdp_weighted_temps.append(sum_gdp_tas / sum_gdp)

    # Mean over time of GDP-weighted temperature
    mean_gdp_weighted_temp = np.mean(gdp_weighted_temps)

    # Solve for coefficients
    a1_linear = (global_mean_linear - value_at_ref_linear) / (mean_gdp_weighted_temp - T_ref_linear)
    a0_linear = value_at_ref_linear - a1_linear * T_ref_linear

    # Calculate linear reduction array using constraint period temperature
    linear_reduction = a0_linear + a1_linear * tas_period

    # Verify constraint satisfaction
    constraint1_check = a0_linear + a1_linear * T_ref_linear  # Should equal value_at_ref_linear

    # Verify global mean constraint using time series (should be exact)
    constraint2_check = calculate_gdp_weighted_mean(
        np.ones_like(tas_series) * (1 + linear_reduction)[np.newaxis, :, :],  # Broadcast reduction to time dimension
        gdp_series, years, lat, valid_mask, period_start, period_end
    ) - 1

    # Store GDP-weighted temperature for reporting
    gdp_weighted_tas_mean = mean_gdp_weighted_temp

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
        'gdp_weighted_tas_mean': float(gdp_weighted_tas_mean)
    }


def calculate_quadratic_target_reduction(quadratic_config, lat, valid_mask, all_data, reference_ssp, period_start, period_end):
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
    tas_ref : np.ndarray
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

    # Get time series data
    years = all_data['years']
    tas_series = all_data[reference_ssp]['tas']
    gdp_series = all_data[reference_ssp]['gdp']

    # Calculate time-averaged temperature for constraint period (for creating reduction spatial pattern)
    from coin_ssp_math_utils import calculate_time_means
    tas_period = calculate_time_means(tas_series, years, period_start, period_end)

    # Get time indices for constraint period
    period_mask = (years >= period_start) & (years <= period_end)
    tas_period_series = tas_series[period_mask]
    gdp_period_series = gdp_series[period_mask]

    # Calculate GDP-weighted temperature and T² for each time step
    # The reduction at time t is: reduction[t,lat,lon] = a + b*T[t,lat,lon] + c*T[t,lat,lon]²
    # Constraint: mean_over_time[ sum(GDP[t] * (1 + a + b*T[t] + c*T[t]²)) / sum(GDP[t]) ] = 1 + global_mean
    #
    # Expanding: a + b*mean_over_time[GDP-weighted T] + c*mean_over_time[GDP-weighted T²] = global_mean
    #
    # Also: a + b*T0 + c*T0² = 0  (zero at T0)
    #       b + 2*c*T0 = derivative_at_T0  (slope at T0)
    #
    # From the last two: b = derivative_at_T0 - 2*c*T0, a = -derivative_at_T0*T0 + c*T0²
    #
    # Substituting into first: (-derivative_at_T0*T0 + c*T0²) + (derivative_at_T0 - 2*c*T0)*mean[GDP-wtd T] + c*mean[GDP-wtd T²] = global_mean
    # Solving for c: c = (global_mean + derivative_at_T0*(T0 - mean[GDP-wtd T])) / (T0² - 2*T0*mean[GDP-wtd T] + mean[GDP-wtd T²])

    gdp_weighted_temps = []
    gdp_weighted_temps2 = []
    for t in range(len(tas_period_series)):
        sum_gdp = np.sum(gdp_period_series[t][valid_mask])
        sum_gdp_tas = np.sum((gdp_period_series[t] * tas_period_series[t])[valid_mask])
        sum_gdp_tas2 = np.sum((gdp_period_series[t] * tas_period_series[t]**2)[valid_mask])
        gdp_weighted_temps.append(sum_gdp_tas / sum_gdp)
        gdp_weighted_temps2.append(sum_gdp_tas2 / sum_gdp)

    # Mean over time of GDP-weighted T and T²
    mean_gdp_weighted_temp = np.mean(gdp_weighted_temps)
    mean_gdp_weighted_temp2 = np.mean(gdp_weighted_temps2)

    # Solve for coefficients
    denominator = T0**2 - 2*T0*mean_gdp_weighted_temp + mean_gdp_weighted_temp2
    c_quad = (global_mean_quad - derivative_at_T0*(mean_gdp_weighted_temp - T0)) / denominator
    b_quad = derivative_at_T0 - 2*c_quad*T0
    a_quad = -derivative_at_T0*T0 + c_quad*T0**2

    # Store for reporting
    gdp_weighted_tas_mean = mean_gdp_weighted_temp
    gdp_weighted_tas2_mean = mean_gdp_weighted_temp2

    # Calculate quadratic reduction array using constraint period temperature
    quadratic_reduction = a_quad + b_quad * tas_period + c_quad * tas_period**2

    # Verify constraint satisfaction
    constraint1_check = a_quad + b_quad * T0 + c_quad * T0**2  # Should be 0 at T0
    constraint2_check = b_quad + 2 * c_quad * T0  # Derivative at T0: should equal derivative_at_T0

    # Verify global mean constraint using time series (should be exact)
    constraint3_check = calculate_gdp_weighted_mean(
        np.ones_like(tas_series) * (1 + quadratic_reduction)[np.newaxis, :, :],  # Broadcast reduction to time dimension
        gdp_series, years, lat, valid_mask, period_start, period_end
    ) - 1

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
        'gdp_weighted_tas_mean': float(gdp_weighted_tas_mean),
        'gdp_weighted_tas2_mean': float(gdp_weighted_tas2_mean),
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
        List of target configuration dictionaries, each containing:
        - 'target_name': Unique identifier
        - Type-specific parameters (determines calculation method):
          * Constant: 'global_mean_amount'
          * Linear: 'global_mean_amount' (without zero point)
          * Quadratic: 'zero_amount_temperature'
    gridded_data : dict
        Dictionary containing gridded data arrays:
        - 'tas_ref': Reference period temperature [lat, lon]
        - 'gdp_target': Target period GDP [lat, lon]
        - 'lat': Latitude coordinates
    all_data : dict
        Combined data structure containing time series
    reference_ssp : str
        Reference SSP scenario name
    target_period_start : int
        Start year of target period
    target_period_end : int
        End year of target period

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

    lat = gridded_data['lat']
    valid_mask = all_data['_metadata']['valid_mask']

    # Get time period bounds
    target_period_start = config['time_periods']['target_period']['start_year']
    target_period_end = config['time_periods']['target_period']['end_year']
    historical_period_start = config['time_periods']['historical_period']['start_year']
    historical_period_end = config['time_periods']['historical_period']['end_year']

    # Get shape for constant targets
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
                target_config['global_mean_amount'], tas_ref.shape
            )
            result = {
                'target_shape': target_shape,
                'reduction_array': reduction_array,
                'coefficients': None,
                'constraint_verification': None,
                'global_statistics': {
                    'gdp_weighted_mean': target_config['global_mean_amount']
                }
            }

        elif target_shape == 'quadratic':
            # Quadratic reduction (has zero point)
            result = calculate_quadratic_target_reduction(target_config, lat, valid_mask,
                                                         all_data, reference_ssp, period_start, period_end)
            result['target_shape'] = target_shape

        elif target_shape == 'linear':
            # Linear reduction (has global mean constraint)
            result = calculate_linear_target_reduction(target_config, lat, valid_mask,
                                                      all_data, reference_ssp, period_start, period_end)
            result['target_shape'] = target_shape

        else:
            raise ValueError(f"Unknown target_shape '{target_shape}' for target '{target_name}'. "
                           f"Must be 'constant', 'linear', or 'quadratic'.")

        results[target_name] = result

    return results