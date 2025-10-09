import numpy as np
import xarray as xr
from typing import Tuple, Sequence, Optional, Dict, Any
from coin_ssp_math_utils import calculate_time_means

def _check_dims(t: xr.DataArray, g: xr.DataArray, a: xr.DataArray,
                dims: Sequence[str]) -> None:
    for d in dims:
        if d not in t.dims or d not in g.dims:
            raise ValueError(f"tas and gdp must both have dim '{d}'.")
    # area must cover the spatial dims; broadcasting across time is fine
    for d in dims[1:]:
        if d not in a.dims:
            raise ValueError(f"area must have spatial dim '{d}'.")

def _tw_moments(t: xr.DataArray, w: xr.DataArray, max_k: int):
    """Return T1, T2[, T3] with a common finite mask; each is a 0-D DataArray."""
    mask = np.isfinite(t) & np.isfinite(w)
    t = t.where(mask).astype("float64")
    w = w.where(mask).astype("float64")

    dims = (t * w).dims  # union of dims; xarray will broadcast automatically

    T1 = (t * w).sum(dims, skipna=True)
    outs = [T1]
    if max_k >= 2:
        T2 = ((t * t) * w).sum(dims, skipna=True); outs.append(T2)
    if max_k >= 3:
        T3 = ((t * t * t) * w).sum(dims, skipna=True); outs.append(T3)

    return tuple(outs)

def fit_linear_A_xr(
    t: xr.DataArray, w: xr.DataArray, t0: float, response: float, valid_mask: xr.DataArray, eps: float = 1e-12
) -> Tuple[float, float]:
    """
    Find a0, a1 for A(t)=a0+a1*t subject to:
      A(t0)=1  and  sum(A*t*w) / sum(t*w) = 1 + response

    valid_mask is applied to weights to exclude invalid grid cells.
    """
    w_masked = w.where(valid_mask, 0.0)
    T1, T2 = _tw_moments(t, w_masked, max_k=2)
    T1, T2 = T1.item(), T2.item()

    denom = T2 - t0 * T1
    if abs(denom) < eps:
        # Consistent only if response*T1 ~ 0; choose simplest solution then.
        if not np.isclose(response * T1, 0.0, rtol=1e-12, atol=1e-12):
            raise ValueError("Inconsistent linear constraints (denominator ~ 0 but response*T1 ≠ 0).")
        a1 = 0.0
        a0 = 0.0
    else:
        a1 = (response * T1) / denom
        a0 =  - a1 * t0

    return float(a0), float(a1)

def fit_quadratic_A_xr(
    t: xr.DataArray, w: xr.DataArray, t0: float, td: float, response: float, valid_mask: xr.DataArray, eps: float = 1e-12
) -> Tuple[float, float, float]:
    """
    Find a0, a1, a2 for A(t)=a0+a1*t+a2*t**2 subject to:
      A(t0)=1,  dA/dt|_{t=td}=0,  and  sum(A*t*w) / sum(t*w) = 1 + response

    valid_mask is applied to weights to exclude invalid grid cells.
    """
    w_masked = w.where(valid_mask, 0.0)
    T1, T2, T3 = _tw_moments(t, w_masked, max_k=3)
    T1, T2, T3 = T1.item(), T2.item(), T3.item()

    denom = T3 - 2.0 * td * T2 + (2.0 * td * t0 - t0 * t0) * T1
    if abs(denom) < eps:
        if not np.isclose(response * T1, 0.0, rtol=1e-12, atol=1e-12):
            raise ValueError("Inconsistent quadratic constraints (denominator ~ 0 but response*T1 ≠ 0).")
        # Simplest valid member: constant A(t)=0 (a2=a1=0) satisfies all constraints when response*T1=0
        a2 = 0.0
        a1 = 0.0
        a0 = 0.0
    else:
        a2 = (response * T1) / denom
        a1 = -2.0 * a2 * td
        a0 = - a2 * (t0 * t0 - 2.0 * td * t0)

    return float(a0), float(a1), float(a2)


def calculate_constant_target_response(gdp_amount_value, tas_ref_template):
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


def calculate_linear_target_response(linear_config, valid_mask, all_data, reference_ssp, period_start, period_end):
    """
    Calculate linear temperature-dependent GDP reduction using constraint satisfaction.

    Implements the mathematical framework:
    reduction(T) = a0 + a1 * T

    Subject to two constraints:
    1. Zero anchor: reduction(T_zero) = 0
    2. GDP-weighted global mean: ∑[w_i * gdp_i * (1 + reduction(T_i))] / ∑[w_i * gdp_i] = target_mean

    Parameters
    ----------
    linear_config : dict
        Configuration containing:
        - 'global_mean_amount': Target GDP-weighted global mean (e.g., -0.10)
        - 'zero_amount_temperature': Temperature with zero reduction (e.g., 13.5)
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
        - 'diagnostics': Diagnostic information from fitting function
    """
    print("\n" + "="*80)
    print("DEBUG: calculate_linear_target_response")
    print("="*80)

    # Extract configuration parameters
    global_mean_amount = linear_config['global_mean_amount']
    zero_amount_temperature = linear_config['zero_amount_temperature']

    print(f"DEBUG: Config parameters:")
    print(f"  global_mean_amount = {global_mean_amount}")
    print(f"  zero_amount_temperature = {zero_amount_temperature}")
    print(f"  period_start = {period_start}, period_end = {period_end}")

    # Get time series data as xarray DataArrays
    tas_series = all_data[reference_ssp]['tas']
    gdp_series = all_data[reference_ssp]['gdp']
    area_weights = all_data['_metadata']['area_weights']

    print(f"DEBUG: Input data shapes:")
    print(f"  tas_series.shape = {tas_series.shape}")
    print(f"  gdp_series.shape = {gdp_series.shape}")
    print(f"  area_weights.shape = {area_weights.shape}")
    print(f"  valid_mask.shape = {valid_mask.shape}, sum = {np.sum(valid_mask)}")

    # Calculate time-averaged temperature for constraint period
    tas_period = calculate_time_means(tas_series, period_start, period_end)

    print(f"DEBUG: Time-averaged temperature:")
    print(f"  tas_period.shape = {tas_period.shape}")
    print(f"  tas_period range = [{float(tas_period.min()):.2f}, {float(tas_period.max()):.2f}]")
    print(f"  tas_period mean = {float(tas_period.mean()):.2f}")

    # Select constraint period data using coordinate-based slicing
    tas_period_series = tas_series.sel(time=slice(period_start, period_end))
    gdp_period_series = gdp_series.sel(time=slice(period_start, period_end))

    print(f"DEBUG: Constraint period data:")
    print(f"  tas_period_series.shape = {tas_period_series.shape}")
    print(f"  gdp_period_series.shape = {gdp_period_series.shape}")

    print(f"DEBUG: Calling fit_linear_A_xr with:")
    print(f"  tas_zero = {zero_amount_temperature}")
    print(f"  response = {global_mean_amount}")

    # Use fit_linear_A_xr with valid_mask as extra_weight
    # The function solves for A(T) = a0 + a1*T where:
    # - A(tas_zero) = 1 (zero anchor constraint at zero_amount_temperature)
    # - sum(A(T)*area*gdp) / sum(area*gdp) = 1 + response
    # def fit_linear_A_xr(
    # t: xr.DataArray, w: xr.DataArray, t0: float, response: float, eps: float = 1e-12) -> Tuple[float, float]:
    a0, a1 = fit_linear_A_xr(
        tas_period_series,
        gdp_period_series * area_weights,
        zero_amount_temperature,
        global_mean_amount,
        valid_mask
    )

    print(f"DEBUG: Fitting results:")
    print(f"  a0 = {a0}")
    print(f"  a1 = {a1}")

    # The function gives us A(T) = a0 + a1*T which directly represents reduction(T)

    # Calculate linear reduction array (automatic broadcasting)
    linear_response = a0 + a1 * tas_period

    print(f"DEBUG: Final reduction array:")
    print(f"  linear_response.shape = {linear_response.shape}")
    print(f"  linear_response range = [{float(linear_response.min()):.6f}, {float(linear_response.max()):.6f}]")
    print(f"  linear_response mean (all) = {float(linear_response.mean()):.6f}")
    valid_reductions = linear_response.values[valid_mask]
    print(f"  linear_response mean (valid cells) = {float(np.mean(valid_reductions)):.6f}")
    print(f"  Check at zero_temp: a0 + a1*{zero_amount_temperature} = {a0 + a1*zero_amount_temperature:.6f}")
    print("="*80 + "\n")

    return {
        'reduction_array': linear_response.astype(np.float64),
        'coefficients': {
            'a0': float(a0),
            'a1': float(a1),
            'a2': 0.0
        }
    }


def calculate_quadratic_target_response(quadratic_config, valid_mask, all_data, reference_ssp, period_start, period_end):
    """
    Calculate quadratic temperature-dependent GDP reduction using derivative constraint.

    Implements the mathematical framework:
    reduction(T) = a + b*T + c*T²

    Subject to three constraints:
    1. Zero point: reduction(T₀) = 0
    2. Derivative at zero: reduction'(T₀) = zero_derivative_temperature
    3. GDP-weighted global mean: ∑[w_i * gdp_i * (1 + reduction(T_i))] / ∑[w_i * gdp_i] = target_mean

    Parameters
    ----------
    quadratic_config : dict
        Configuration containing:
        - 'global_mean_amount': Target GDP-weighted global mean (e.g., -0.10)
        - 'zero_amount_temperature': Temperature with zero reduction (e.g., 13.5)
        - 'zero_derivative_temperature': Slope at T₀ (e.g., -0.01)
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
    zero_amount_temperature = quadratic_config['zero_amount_temperature']
    zero_derivative_temperature = quadratic_config['zero_derivative_temperature']

    # Get time series data as xarray DataArrays
    tas_series = all_data[reference_ssp]['tas']
    gdp_series = all_data[reference_ssp]['gdp']
    area_weights = all_data['_metadata']['area_weights']

    # Calculate time-averaged temperature for constraint period
    tas_period = calculate_time_means(tas_series, period_start, period_end)

    # Select constraint period data using coordinate-based slicing
    tas_period_series = tas_series.sel(time=slice(period_start, period_end))
    gdp_period_series = gdp_series.sel(time=slice(period_start, period_end))

    # Use fit_quadratic_A_xr with valid_mask as extra_weight
    # The function solves for A(T) = a0 + a1*T + a2*T² where:
    # - A(tas_zero) = 0 (zero anchor constraint at zero_amount_temperature)
    # - dA/dt(tas_zero_deriv) = 0 (derivative is zero at zero_derivative_temperature)
    # - sum(A(T)*area*gdp) / sum(area*gdp) = 1 + response
    # def fit_quadratic_A_xr(
    # t: xr.DataArray, w: xr.DataArray, t0: float, td: float, response: float, eps: float = 1e-12 ) -> Tuple[float, float, float]:
    a0, a1, a2 = fit_quadratic_A_xr(
        tas_period_series,
        area_weights * gdp_period_series,
        zero_amount_temperature,
        zero_derivative_temperature,
        global_mean_quad,
        valid_mask
    )

    print(f"DEBUG: QUadratic fitting results:")
    print(f"  a0 = {a0}")
    print(f"  a1 = {a1}")
    print(f"  a2 = {a2}")
    # The function gives us A(T) = a0 + a1*T + a2*T² which directly represents reduction(T)

    # Calculate quadratic reduction array (automatic broadcasting)
    quadratic_response = a0 + a1 * tas_period + a2 * tas_period**2

    return {
        'reduction_array': quadratic_response.astype(np.float64),
        'coefficients': {
            'a0': float(a0),
            'a1': float(a1),
            'a2': float(a2)
        }
    }


def calculate_all_target_responses(target_configs, gridded_data, all_data, reference_ssp, config):
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
            reduction_array = calculate_constant_target_response(
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
            result = calculate_quadratic_target_response(target_config, valid_mask,
                                                         all_data, reference_ssp, period_start, period_end)
            result['target_shape'] = target_shape

        elif target_shape == 'linear':
            # Linear reduction
            result = calculate_linear_target_response(target_config, valid_mask,
                                                      all_data, reference_ssp, period_start, period_end)
            result['target_shape'] = target_shape

        else:
            raise ValueError(f"Unknown target_shape '{target_shape}' for target '{target_name}'")

        results[target_name] = result

    return results