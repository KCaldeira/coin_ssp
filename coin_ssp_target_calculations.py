import numpy as np
import xarray as xr
from typing import Dict, Any

from coin_ssp_math_utils import calculate_gdp_weighted_mean

import numpy as np
import xarray as xr
from typing import Tuple, Sequence, Optional, Dict, Any

def _check_dims(t: xr.DataArray, g: xr.DataArray, a: xr.DataArray,
                dims: Sequence[str]) -> None:
    for d in dims:
        if d not in t.dims or d not in g.dims:
            raise ValueError(f"tas and gdp must both have dim '{d}'.")
    # area must cover the spatial dims; broadcasting across time is fine
    for d in dims[1:]:
        if d not in a.dims:
            raise ValueError(f"area must have spatial dim '{d}'.")

def _weighted_sums(
    tas: xr.DataArray,
    gdp: xr.DataArray,
    area: xr.DataArray,
    dims: Sequence[str],
    max_k: int,
    extra_weight: Optional[xr.DataArray] = None,
) -> Tuple[xr.DataArray, ...]:
    """
    Return (S0, S1, ..., Smax_k) where Sk = sum(t^k * w) over 'dims',
    with w = area * gdp * (extra_weight if provided) and a *common mask*
    applied so all Sk use the same sample.
    """
    _check_dims(tas, gdp, area, dims)

    # Common finite mask across tas, gdp, area, (and extra_weight if supplied)
    mask = np.isfinite(tas) & np.isfinite(gdp) & np.isfinite(area)
    if extra_weight is not None:
        mask = mask & np.isfinite(extra_weight)

    # Broadcast weights to 3D and apply mask
    # Multiply gdp first to preserve [time, lat, lon] dimension order
    w = (gdp * area) if extra_weight is None else (gdp * area * extra_weight)
    w = w.where(mask).astype("float64")

    t = tas.where(mask).astype("float64")

    outs = []
    S0 = w.sum(dims, skipna=True)
    outs.append(S0)

    if max_k >= 1:
        S1 = (t * w).sum(dims, skipna=True)
        outs.append(S1)
    if max_k >= 2:
        t2 = t * t
        S2 = (t2 * w).sum(dims, skipna=True)
        outs.append(S2)
    if max_k >= 3:
        t3 = t * t * t
        S3 = (t3 * w).sum(dims, skipna=True)
        outs.append(S3)
    if max_k >= 4:
        t4 = (t * t) * (t * t)
        S4 = (t4 * w).sum(dims, skipna=True)
        outs.append(S4)

    return tuple(outs)  # tuple of 0-D DataArrays


def fit_linear_A_xr(
    tas: xr.DataArray,
    gdp: xr.DataArray,
    area: xr.DataArray,
    tas_zero: float,
    response: float,
    *,
    dims: Sequence[str] = ("time", "lat", "lon"),
    extra_weight: Optional[xr.DataArray] = None,
    eps: float = 1e-12,
    return_diagnostics: bool = False,
) -> Tuple[float, float] | Tuple[float, float, Dict[str, Any]]:
    """
    Solve for (a0, a1) in A(t) = a0 + a1*t under:
      (a) A(tas_zero) = 0
      (b) sum(A(t)*area*gdp) / sum(area*gdp) = 1 + response

    Returns:
      a0, a1  (Python floats)
      If return_diagnostics=True, also returns a dict with simple checks.
    """
    print(f"DEBUG fit_linear_A_xr: Computing weighted sums...")
    S0, S1, S2 = _weighted_sums(tas, gdp, area, dims, max_k=2, extra_weight=extra_weight)
    S0v, S1v, S2v = S0.item(), S1.item(), S2.item()

    print(f"DEBUG fit_linear_A_xr: Weighted sums:")
    print(f"  S0 (sum of weights) = {S0v:.6e}")
    print(f"  S1 (sum of T*weights) = {S1v:.6e}")
    print(f"  S2 (sum of T²*weights) = {S2v:.6e}")

    if abs(S0v) < eps:
        raise ValueError("sum(area*gdp[*(extra_weight)]) ~ 0; weighted mean undefined.")

    t0 = float(tas_zero)
    target = (1.0 + float(response)) * S0v

    print(f"DEBUG fit_linear_A_xr: Constraint setup:")
    print(f"  tas_zero (t0) = {t0}")
    print(f"  response = {response}")
    print(f"  target = (1 + response) * S0 = {target:.6e}")
    print(f"  Constraint: mean(A) = sum(A*T*w)/sum(w) = 1 + response = {1.0 + float(response)}")

    denom = S2v - t0 * S1v
    print(f"DEBUG fit_linear_A_xr: Solving for coefficients:")
    print(f"  denom = S2 - t0*S1 = {S2v:.6e} - {t0}*{S1v:.6e} = {denom:.6e}")

    if abs(denom) < eps:
        if not np.isclose(target, 0.0, rtol=1e-12, atol=1e-12):
            raise ValueError("Constraints inconsistent (S2 - t0*S1 ≈ 0 but target ≠ 0).")
        # Infinite family; choose simplest member
        a1 = 0.0
        a0 = 0.0
        print(f"DEBUG fit_linear_A_xr: Degenerate case, setting a0=a1=0")
    else:
        a1 = target / denom
        a0 = -a1 * t0
        print(f"DEBUG fit_linear_A_xr: Solution:")
        print(f"  a1 = target / denom = {target:.6e} / {denom:.6e} = {a1}")
        print(f"  a0 = -a1 * t0 = -{a1} * {t0} = {a0}")

    if not return_diagnostics:
        return float(a0), float(a1)

    # Diagnostics using existing sums (no extra passes)
    num_check = a0 * S0v + a1 * S1v  # sum(A(T)*w) = sum((a0 + a1*T)*w) = a0*S0 + a1*S1
    print(f"DEBUG fit_linear_A_xr: Verification:")
    print(f"  Weighted mean of A(T) = (a0*S0 + a1*S1)/S0 = {num_check/S0v}")
    print(f"  Should equal 1 + response = {1.0 + float(response)}")
    print(f"  Zero anchor: A(t0) = a0 + a1*t0 = {a0 + a1 * t0} (should be ~0)")

    diagnostics = {
        "zero_anchor_residual": a0 + a1 * t0,  # should be ~ 0
        "mean_target": 1.0 + float(response),
        "mean_actual": num_check / S0v,
        "mean_abs_error": abs(num_check / S0v - (1.0 + float(response))),
        "S0": S0v, "S1": S1v, "S2": S2v,
    }
    return float(a0), float(a1), diagnostics


def fit_quadratic_A_xr(
    tas: xr.DataArray,
    gdp: xr.DataArray,
    area: xr.DataArray,
    tas_zero: float,
    tas_zero_deriv: float,
    response: float,
    *,
    dims: Sequence[str] = ("time", "lat", "lon"),
    extra_weight: Optional[xr.DataArray] = None,
    eps: float = 1e-12,
    return_diagnostics: bool = False,
) -> Tuple[float, float, float] | Tuple[float, float, float, Dict[str, Any]]:
    """
    Solve for (a0, a1, a2) in A(t) = a0 + a1*t + a2*t**2 under:
      (a) A(tas_zero) = 0
      (b) dA/dt(tas_zero_deriv) = 0
      (c) sum(A(t)*t*area*gdp) / sum(area*gdp) = 1 + response

    Returns:
      a0, a1, a2  (Python floats)
      If return_diagnostics=True, also returns a dict with simple checks.
    """
    S0, S1, S2, S3 = _weighted_sums(tas, gdp, area, dims, max_k=3, extra_weight=extra_weight)
    S0v, S1v, S2v, S3v = S0.item(), S1.item(), S2.item(), S3.item()

    if abs(S0v) < eps:
        raise ValueError("sum(area*gdp[*(extra_weight)]) ~ 0; weighted mean undefined.")

    t0 = float(tas_zero)
    td = float(tas_zero_deriv)
    target = (1.0 + float(response)) * S0v

    denom = (2.0 * td * t0 - t0 * t0) * S1v - 2.0 * td * S2v + S3v
    if abs(denom) < eps:
        if not np.isclose(target, 0.0, rtol=1e-12, atol=1e-12):
            raise ValueError("Constraints inconsistent (quadratic denominator ≈ 0 but target ≠ 0).")
        # Infinite family; choose the trivial member
        a2 = 0.0
        a1 = 0.0
        a0 = 0.0
    else:
        a2 = target / denom
        a1 = -2.0 * a2 * td
        a0 = a2 * (2.0 * td * t0 - t0 * t0)

    if not return_diagnostics:
        return float(a0), float(a1), float(a2)

    # Diagnostics using existing sums (no extra passes)
    num_check = a0 * S1v + a1 * S2v + a2 * S3v
    diagnostics = {
        "zero_anchor_residual": a0 + a1 * t0 + a2 * t0 * t0,  # should be ~ 0
        "derivative_anchor_residual": a1 + 2.0 * a2 * td,     # should be ~ 0
        "mean_target": 1.0 + float(response),
        "mean_actual": num_check / S0v,
        "mean_abs_error": abs(num_check / S0v - (1.0 + float(response))),
        "S0": S0v, "S1": S1v, "S2": S2v, "S3": S3v,
    }
    return float(a0), float(a1), float(a2), diagnostics


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
    from coin_ssp_math_utils import calculate_time_means
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
    # - A(tas_zero) = 0 (zero anchor constraint at zero_amount_temperature)
    # - sum(A(T)*area*gdp) / sum(area*gdp) = 1 + response
    a0, a1, diagnostics = fit_linear_A_xr(
        tas_period_series,
        gdp_period_series,
        area_weights,
        tas_zero=zero_amount_temperature,
        response=global_mean_amount,
        extra_weight=valid_mask.astype(float),
        return_diagnostics=True
    )

    print(f"DEBUG: Fitting results:")
    print(f"  a0 = {a0}")
    print(f"  a1 = {a1}")
    print(f"DEBUG: Diagnostics:")
    for key, value in diagnostics.items():
        print(f"  {key} = {value}")

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
        },
        'diagnostics': diagnostics
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
    from coin_ssp_math_utils import calculate_time_means
    tas_period = calculate_time_means(tas_series, period_start, period_end)

    # Select constraint period data using coordinate-based slicing
    tas_period_series = tas_series.sel(time=slice(period_start, period_end))
    gdp_period_series = gdp_series.sel(time=slice(period_start, period_end))

    # Use fit_quadratic_A_xr with valid_mask as extra_weight
    # The function solves for A(T) = a0 + a1*T + a2*T² where:
    # - A(tas_zero) = 0 (zero anchor constraint at zero_amount_temperature)
    # - dA/dt(tas_zero_deriv) = 0 (derivative is zero at zero_derivative_temperature)
    # - sum(A(T)*area*gdp) / sum(area*gdp) = 1 + response
    a0, a1, a2, diagnostics = fit_quadratic_A_xr(
        tas_period_series,
        gdp_period_series,
        area_weights,
        tas_zero=zero_amount_temperature,
        tas_zero_deriv=zero_derivative_temperature,
        response=global_mean_quad,
        extra_weight=valid_mask.astype(float),
        return_diagnostics=True
    )

    # The function gives us A(T) = a0 + a1*T + a2*T² which directly represents reduction(T)

    # Calculate quadratic reduction array (automatic broadcasting)
    quadratic_response = a0 + a1 * tas_period + a2 * tas_period**2

    return {
        'reduction_array': quadratic_response.astype(np.float64),
        'coefficients': {
            'a0': float(a0),
            'a1': float(a1),
            'a2': float(a2)
        },
        'diagnostics': diagnostics
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