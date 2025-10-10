import numpy as np
import xarray as xr
from typing import Tuple, Sequence, Optional, Dict, Any
from coin_ssp_math_utils import calculate_time_means
import numpy as np
import xarray as xr
from typing import Tuple

def _moments_w(t: xr.DataArray, w: xr.DataArray, order: int):
    """
    Compute S0=∑w, S1=∑t w, S2=∑t^2 w with a common finite mask.
    Returns the requested scalars as floats.
    """
    mask = np.isfinite(t) & np.isfinite(w)
    t = t.where(mask).astype("float64")
    w = w.where(mask).astype("float64")

    # after broadcasting, product holds all dims to reduce over
    dims = (t * w).dims

    S0 = w.sum(dims, skipna=True).item()
    S1 = (t * w).sum(dims, skipna=True).item()
    if order == 1:
        return (S0, S1)
    S2 = ((t * t) * w).sum(dims, skipna=True).item()
    return (S0, S1, S2)

def fit_linear_gdp_pattern(
    t: xr.DataArray, w: xr.DataArray, t0: float, response: float, valid_mask: xr.DataArray, eps: float = 1e-12
) -> Tuple[float, float]:
    """
    Solve a0, a1 for A(t)=a0+a1*t with:
      A(t0)=0  and  sum(A*w)/sum(w) = response
    """
    w = w.where(valid_mask, 0.0)
    S0, S1 = _moments_w(t, w, order=1)

    if abs(S0) < eps:
        raise ValueError("sum(w) ≈ 0; mean constraint undefined.")

    denom = S1 - t0 * S0
    rhs = response * S0

    if abs(denom) < eps:
        if not np.isclose(rhs, 0.0, rtol=1e-12, atol=1e-12):
            raise ValueError("Inconsistent linear constraints (denominator ≈ 0 but response ≠ 0).")
        a1 = 0.0
        a0 = 0.0
    else:
        a1 = rhs / denom
        a0 = -a1 * t0

    return float(a0), float(a1)

def fit_quadratic_gdp_pattern(
    t: xr.DataArray, w: xr.DataArray, t0: float, td: float, response: float, valid_mask: xr.DataArray, eps: float = 1e-12
) -> Tuple[float, float, float]:
    """
    Solve a0, a1, a2 for A(t)=a0+a1*t+a2*t**2 with:
      A(t0)=0,  A'(td)=0,  and  sum(A*w)/sum(w) = response
    """
    w = w.where(valid_mask, 0.0)
    S0, S1, S2 = _moments_w(t, w, order=2)

    if abs(S0) < eps:
        raise ValueError("sum(w) ≈ 0; mean constraint undefined.")

    denom = (2.0 * td * t0 - t0 * t0) * S0 - 2.0 * td * S1 + S2
    rhs = response * S0

    if abs(denom) < eps:
        if not np.isclose(rhs, 0.0, rtol=1e-12, atol=1e-12):
            raise ValueError("Inconsistent quadratic constraints (denominator ≈ 0 but response ≠ 0).")
        a2 = 0.0
        a1 = 0.0
        a0 = 0.0
    else:
        a2 = rhs / denom
        a1 = -2.0 * a2 * td
        a0 = a2 * (2.0 * td * t0 - t0 * t0)

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

    print(f"DEBUG: Calling fit_linear_gdp_pattern with:")
    print(f"  tas_zero = {zero_amount_temperature}")
    print(f"  response = {global_mean_amount}")

    # Use fit_linear_gdp_pattern with valid_mask as extra_weight
    # The function solves for A(T) = a0 + a1*T where:
    # - A(tas_zero) = 1 (zero anchor constraint at zero_amount_temperature)
    # - sum(A(T)*area*gdp) / sum(area*gdp) = 1 + response
    # def fit_linear_gdp_pattern(
    # t: xr.DataArray, w: xr.DataArray, t0: float, response: float, eps: float = 1e-12) -> Tuple[float, float]:
    a0, a1 = fit_linear_gdp_pattern(
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

    # Calculate linear reduction array at each grid cell as GDP-weighted mean over time
    # reduction[lat,lon] = mean_over_time[(a0 + a1*T[t,lat,lon]) * GDP[t,lat,lon]] / mean_over_time[GDP[t,lat,lon]]
    reduction_timeseries = a0 + a1 * tas_period_series  # [time, lat, lon]
    gdp_masked = gdp_period_series.where(valid_mask)
    linear_response = (
        (reduction_timeseries * gdp_masked).sum(dim='time') /
        gdp_masked.sum(dim='time')
    )

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

    # Use fit_quadratic_gdp_pattern with valid_mask as extra_weight
    # The function solves for A(T) = a0 + a1*T + a2*T² where:
    # - A(tas_zero) = 0 (zero anchor constraint at zero_amount_temperature)
    # - dA/dt(tas_zero_deriv) = 0 (derivative is zero at zero_derivative_temperature)
    # - sum(A(T)*area*gdp) / sum(area*gdp) = 1 + response
    # def fit_quadratic_gdp_pattern(
    # t: xr.DataArray, w: xr.DataArray, t0: float, td: float, response: float, eps: float = 1e-12 ) -> Tuple[float, float, float]:
    a0, a1, a2 = fit_quadratic_gdp_pattern(
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

    # Calculate quadratic reduction array at each grid cell as GDP-weighted mean over time
    # reduction[lat,lon] = mean_over_time[(a0 + a1*T[t,lat,lon] + a2*T[t,lat,lon]²) * GDP[t,lat,lon]] / mean_over_time[GDP[t,lat,lon]]
    reduction_timeseries = a0 + a1 * tas_period_series + a2 * tas_period_series**2  # [time, lat, lon]
    gdp_masked = gdp_period_series.where(valid_mask)
    quadratic_response = (
        (reduction_timeseries * gdp_masked).sum(dim='time') /
        gdp_masked.sum(dim='time')
    )

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