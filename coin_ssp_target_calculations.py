import numpy as np
from typing import Dict, Any

from coin_ssp_math_utils import calculate_global_mean


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


def calculate_linear_target_reduction(linear_config, tas_ref, gdp_target, lat, valid_mask):
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
    gdp_weighted_tas_mean = np.float64(calculate_global_mean(gdp_target * tas_ref, lat, valid_mask) / global_gdp_target)

    # Set up weighted least squares system for exact constraint satisfaction
    X = np.array([
        [1.0, T_ref_linear],                    # Point constraint equation
        [1.0, gdp_weighted_tas_mean]           # GDP-weighted global mean equation
    ], dtype=np.float64)

    y = np.array([
        value_at_ref_linear,                    # Target at reference temperature
        global_mean_linear                      # Target GDP-weighted global mean
    ], dtype=np.float64)

    # Solve for coefficients: [a0, a1]
    coefficients = np.linalg.solve(X, y)
    a0_linear, a1_linear = coefficients

    # Calculate linear reduction array
    linear_reduction = a0_linear + a1_linear * tas_ref

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
        'gdp_weighted_tas_mean': float(gdp_weighted_tas_mean)
    }


def calculate_quadratic_target_reduction(quadratic_config, tas_ref, gdp_target, lat, valid_mask):
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

    # Calculate GDP-weighted global means
    global_gdp_target = calculate_global_mean(gdp_target, lat, valid_mask)
    gdp_weighted_tas_mean = np.float64(calculate_global_mean(gdp_target * tas_ref, lat, valid_mask) / global_gdp_target)
    gdp_weighted_tas2_mean = np.float64(calculate_global_mean(gdp_target * tas_ref**2, lat, valid_mask) / global_gdp_target)

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

    denominator = T0**2 - 2*T0*gdp_weighted_tas_mean + gdp_weighted_tas2_mean
    c_quad = (global_mean_quad - derivative_at_T0*(gdp_weighted_tas_mean - T0)) / denominator
    b_quad = derivative_at_T0 - 2*c_quad*T0
    a_quad = -derivative_at_T0*T0 + c_quad*T0**2

    # Calculate quadratic reduction array using absolute temperature
    quadratic_reduction = a_quad + b_quad * tas_ref + c_quad * tas_ref**2

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
        'gdp_weighted_tas_mean': float(gdp_weighted_tas_mean),
        'gdp_weighted_tas2_mean': float(gdp_weighted_tas2_mean),
        'derivative_at_zero_tas': float(derivative_at_T0),
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
        - 'tas_ref': Reference period temperature [lat, lon]
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

    tas_ref = gridded_data['tas_ref']
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
                target_config['gdp_amount'], tas_ref.shape
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
            result = calculate_quadratic_target_reduction(target_config, tas_ref, gdp_target, lat, valid_mask)
            result['target_shape'] = target_shape

        elif target_shape == 'linear':
            # Linear reduction (has global mean constraint)
            result = calculate_linear_target_reduction(target_config, tas_ref, gdp_target, lat, valid_mask)
            result['target_shape'] = target_shape

        else:
            raise ValueError(f"Unknown target_shape '{target_shape}' for target '{target_name}'. "
                           f"Must be 'constant', 'linear', or 'quadratic'.")

        results[target_name] = result

    return results