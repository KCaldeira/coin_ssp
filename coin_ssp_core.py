# The purpose of this code is to determine a total factor productivity time series based on 
# the following information:
#
# <gdp? a time series of gdp
# <pop> a time series of population
# 
# <params> a dictionary of parameters
# <params[deelta]> depreciation rate
# <alpha> elasticity of output with respect to capital
# 
#  We have as our core equation:
# 
#  Y(t) = A(t) K(t)**alpha  L(t)**(1-alpha)
#
#  d K(t) / d t = s * Y(t) - delta * K(t)
#
# where Y(t) is gross production, A(t) is total factor productivity, K(t) is capital, and
#  L(t) is population.
#
# In dimension units:
#
# Y(t) -- $/yr
# K(t) -- $
# L(t) -- people
# A(t) -- ($/yr) ($)**(-alpha) * (people)**(alpha-1)
#
# Howeever, this can be non-dimensionalized by dividing each year's value by the value at the first
# year, for example:
#
#  y(t) = Y(t)/Y(0)   where 0 is a stand-in for the reference year
#  k(t) = K(t)/K(0)
#  l(t) = L(t)/L(0)
#  a(t) - A(t)/A(0)
#
#  This makes it so that all of the year(0) values are known.
#
import numpy as np
from dataclasses import dataclass
import copy
from scipy.optimize import minimize

# Small epsilon to prevent division by zero in ratio calculations
RATIO_EPSILON = 1e-20

@dataclass
# parameters for the COIN-SSP model
class ModelParams:
    year_diverge: int = 2025 # year at which climate starts to affect the economy (e.g., 2025)
    year_scale: int = 2100 # reference year for when to check climate amount (e.g., 2100)
    amount_scale: float = 0 # target amount of climate impact on gdp in year_scale (e.g., -0.1 for 10% gdp losses in year_scale)
    s: float = 0.3  # savings rate
    alpha: float = 0.3  # elasticity of output with respect to capital
    delta: float = 0.1  # depreciation rate in 1/yr
    tas0: float = 0 # reference temperature for temperature (tas) response
    pr0: float = 0 # reference precipitation for preciptiation (pr) response
    k_tas1: float = 0 # linear temperature sensitivity for capital loss
    k_tas2: float = 0  # quadratic temperature sensitivity for capital loss
    k_pr1: float = 0  # linear precipitation sensitivity for capital loss
    k_pr2: float = 0  # quadratic precipitation sensitivity for capital loss
    tfp_tas1: float = 0  # linear temperature sensitivity for TFP loss
    tfp_tas2: float = 0  # quadratic temperature sensitivity for TFP loss
    tfp_pr1: float = 0  # linear precipitation sensitivity for TFP loss
    tfp_pr2: float = 0  # quadratic precipitation sensitivity for TFP loss
    y_tas1: float = 0  # linear temperature sensitivity for output loss
    y_tas2: float = 0  # quadratic temperature sensitivity for output loss
    y_pr1: float = 0  # linear precipitation sensitivity for output loss
    y_pr2: float = 0  # quadratic precipitation sensitivity for output loss

@dataclass
# parameters for scaling climate impacts for damage optimization runs
class ScalingParams:
    scaling_name: str = "default"  # name for this scaling parameter set
    k_tas1: float = 0 # linear temperature sensitivity for capital loss
    k_tas2: float = 0  # quadratic temperature sensitivity for capital loss
    k_pr1: float = 0  # linear precipitation sensitivity for capital loss
    k_pr2: float = 0  # quadratic precipitation sensitivity for capital loss
    tfp_tas1: float = 0  # linear temperature sensitivity for TFP loss
    tfp_tas2: float = 0  # quadratic temperature sensitivity for TFP loss
    tfp_pr1: float = 0  # linear precipitation sensitivity for TFP loss
    tfp_pr2: float = 0  # quadratic precipitation sensitivity for TFP loss
    y_tas1: float = 0  # linear temperature sensitivity for output loss
    y_tas2: float = 0  # quadratic temperature sensitivity for output loss
    y_pr1: float = 0  # linear precipitation sensitivity for output loss
    y_pr2: float = 0  # quadratic precipitation sensitivity for output loss

def create_scaled_params(params, scaling, scale_factor):
    """
    Create scaled parameters from base parameters, scaling template, and scale factor.
    Computed once, used many times.
    """
    params_scaled = copy.copy(params)
    params_scaled.k_tas1   = scale_factor * scaling.k_tas1
    params_scaled.k_tas2   = scale_factor * scaling.k_tas2
    params_scaled.tfp_tas1 = scale_factor * scaling.tfp_tas1
    params_scaled.tfp_tas2 = scale_factor * scaling.tfp_tas2
    params_scaled.y_tas1   = scale_factor * scaling.y_tas1
    params_scaled.y_tas2   = scale_factor * scaling.y_tas2
    params_scaled.k_pr1    = scale_factor * scaling.k_pr1
    params_scaled.k_pr2    = scale_factor * scaling.k_pr2
    params_scaled.tfp_pr1  = scale_factor * scaling.tfp_pr1
    params_scaled.tfp_pr2  = scale_factor * scaling.tfp_pr2
    params_scaled.y_pr1    = scale_factor * scaling.y_pr1
    params_scaled.y_pr2    = scale_factor * scaling.y_pr2
    return params_scaled

def calculate_tfp_coin_ssp(pop, gdp, params):
    """
    Calculate total factor productivity time series using the Solow-Swan growth model.

    Parameters
    ----------
    pop : array-like
        Time series of population (L) in people
    gdp : array-like
        Time series of gross domestic product (Y) in $/yr
    params : dict
        Model parameters containing:
        - 's': savings rate (dimensionless)
        - 'alpha': elasticity of output with respect to capital (dimensionless)
        - 'delta': depreciation rate in 1/yr

    Returns
    -------
    a : numpy.ndarray
        Total factor productivity time series, normalized to year 0 (A(t)/A(0))
    k : numpy.ndarray
        Capital stock time series, normalized to year 0 (K(t)/K(0))

    Notes
    -----
    Assumes system is in steady-state at year 0 with normalized values of 1.
    Uses discrete time integration with 1-year time steps.
    """

    # Convert to numpy arrays for consistent handling
    pop = np.array(pop)
    gdp = np.array(gdp)

    y = gdp/gdp[0] # output normalized to year 0
    l = pop/pop[0] # population normalized to year 0
    k = np.copy(y) # capital stock normalized to year 0
    a = np.copy(y) # total factor productivity normalized to year 0
    s = params.s # savings rate
    alpha = params.alpha # elasticity of output with respect to capital
    delta = params.delta # depreciation rate in units of 1/yr

    # Let's assume that at year 0, the system is in steady-state, do d k / dt = 0 at year 0, and a[0] = 1.
    # 0 == s * y[0] - delta * k[0]
    k[0] = (s/delta) # everything is non0dimensionalized to 1 at year 0
    # y[0] ==  a[0] * k[0]**alpha * l[0]**(1-alpha)

    a[0] = k[0]**(-alpha) # nondimensionalized Total Factor Productivity is 0 in year 0

    # since we are assuming steady state, the capital stock will be the same at the start of year 1


    for t in range(len(y)-1):
        # I want y(t+1) ==  a(t+1) * k(t+1)**alpha * l(t)**(1-alpha)
        #
        # so this means that a(t+1) = y(t + 1) / (k(t+1)**alpha * l(t+1)**(1-alpha))

        dkdt = s * y[t] - delta *k[t]
        k[t+1] = k[t] + dkdt  # assumed time step is one year

        a[t+1] = y[t+1] / (k[t+1]**alpha * l[t+1]**(1-alpha))

    # Check for NaN values in TFP result and terminate with diagnostic output
    if np.any(np.isnan(a)) or np.any(np.isnan(k)):
        print(f"\n{'='*80}")
        print(f"NaN DETECTED IN TFP CALCULATION - STOPPING FOR DIAGNOSIS")
        print(f"{'='*80}")

        print(f"INPUT DATA:")
        print(f"  Population (pop): shape={pop.shape}, dtype={pop.dtype}")
        print(f"    range: {np.min(pop):.6e} to {np.max(pop):.6e}")
        print(f"    zeros: {np.sum(pop == 0)} / {len(pop)} ({100*np.sum(pop == 0)/len(pop):.1f}%)")
        print(f"    first/last values: {pop[0]:.6e}, {pop[-1]:.6e}")
        print(f"    full values: {pop}")

        print(f"  GDP: shape={gdp.shape}, dtype={gdp.dtype}")
        print(f"    range: {np.min(gdp):.6e} to {np.max(gdp):.6e}")
        print(f"    zeros: {np.sum(gdp == 0)} / {len(gdp)} ({100*np.sum(gdp == 0)/len(gdp):.1f}%)")
        print(f"    first/last values: {gdp[0]:.6e}, {gdp[-1]:.6e}")
        print(f"    full values: {gdp}")

        print(f"MODEL PARAMETERS:")
        print(f"  s (savings rate): {s}")
        print(f"  alpha (capital elasticity): {alpha}")
        print(f"  delta (depreciation rate): {delta}")

        print(f"INTERMEDIATE CALCULATIONS:")
        print(f"  y (normalized GDP): {y}")
        print(f"    contains NaN: {np.any(np.isnan(y))}")
        print(f"  l (normalized pop): {l}")
        print(f"    contains NaN: {np.any(np.isnan(l))}")
        print(f"  k[0] = s/delta = {s}/{delta} = {k[0]}")
        print(f"  a[0] = k[0]**(-alpha) = {k[0]}**(-{alpha}) = {a[0]}")

        print(f"OUTPUT RESULTS:")
        print(f"  TFP (a): {a}")
        print(f"    contains NaN: {np.any(np.isnan(a))}")
        print(f"    NaN indices: {np.where(np.isnan(a))[0].tolist()}")
        print(f"  Capital (k): {k}")
        print(f"    contains NaN: {np.any(np.isnan(k))}")
        print(f"    NaN indices: {np.where(np.isnan(k))[0].tolist()}")

        print(f"{'='*80}")
        raise RuntimeError("NaN detected in TFP calculation. See diagnostic output above.")

    return a, k

def calculate_coin_ssp_forward_model(tfp, pop, tas, pr, params: ModelParams):

    # This function calculates the forward model for the COIN-SSP economic model.
    # It takes in total factor productivity (tfp), population (pop), 
    # temperature (tas), and a set of model parameters (params).
    # The function returns the adjusted total factor productivity (tfp_adj),
    # capital stock (k), and output (y) time series.

    # Extract parameters from the ModelParams dataclass
    s = params.s  # savings rate
    alpha = params.alpha  # elasticity of output with respect to capital
    delta = params.delta  # depreciation rate in 1/yr
    # Note: The following parameters default to 0 if not provided
    tas0 = params.tas0  # reference temperature for temperature (tas) response
    pr0 = params.pr0  # reference precipitation for precipitation (pr) response
    k_tas1 = params.k_tas1  # linear temperature sensitivity for capital loss
    k_tas2 = params.k_tas2  # quadratic temperature sensitivity for capital loss        
    k_pr1 = params.k_pr1  # linear precipitation sensitivity for capital loss
    k_pr2 = params.k_pr2  # quadratic precipitation sensitivity for capital loss
    tfp_tas1 = params.tfp_tas1  # linear temperature sensitivity for TFP loss
    tfp_tas2 = params.tfp_tas2  # quadratic temperature sensitivity for TFP loss
    tfp_pr1 = params.tfp_pr1  # linear precipitation sensitivity for TFP loss
    tfp_pr2 = params.tfp_pr2  # quadratic precipitation sensitivity for TFP loss
    y_tas1 = params.y_tas1  # linear temperature sensitivity for output loss
    y_tas2 = params.y_tas2  # quadratic temperature sensitivity for output loss
    y_pr1 = params.y_pr1  # linear precipitation sensitivity for output loss
    y_pr2 = params.y_pr2  # quadratic precipitation sensitivity for output loss

    # convert TFP into interannual fractional increase in TFP
    tfp_growth = tfp[1:]/tfp[:-1] # note that this is one element shorter than the other vectors

    # non-dimensionalize the input data
    y = np.ones_like(pop)  # output normalized to 1 at year 0
    l = pop/pop[0] # population normalized to year 0
    k = np.copy(y) # capital stock normalized to year 0
    a = np.copy(y) # total factor productivity normalized to year 0

    # assume that at year 0, the system is in steady-state, do d k / dt = 0 at year 0, and a[0] = 1.
    # 0 == s * y[0] - delta * k[0]
    k[0] = (s/delta) # everything is non0dimensionalized to 1 at year 0
    # y[0] ==  a[0] * k[0]**alpha * l[0]**(1-alpha)
    a0 = k[0]**(-alpha) # nondimensionalized Total Factor Productivity in year 0 in steady state without climate impacts

    # compute climate effect on capital stock, tfp growth rate, and output
    #note that these are all defined so a positive number means a positive economic impact
    k_climate = 1.0 + k_tas1 * (tas - tas0) + k_tas2 * (tas - tas0)**2 # units of fraction of capital
    k_climate += k_pr1 * (pr - pr0) + k_pr2 * (pr - pr0)**2  
    tfp_climate = 1.0 + tfp_tas1 * (tas - tas0) + tfp_tas2 * (tas - tas0)**2 # units of fraction of TFP
    tfp_climate += tfp_pr1 * (pr - pr0) + tfp_pr2 * (pr - pr0)**2  
    y_climate = 1.0 + y_tas1 * (tas - tas0) + y_tas2 * (tas - tas0)**2  # units of fraction of output
    y_climate += y_pr1 * (pr - pr0) + y_pr2 * (pr - pr0)**2

    a[0] = a0 * tfp_climate[0] # initial TFP adjusted for climate in year 0

    for t in range(len(y)-1):

        # compute climate responses
        # Note that the climate response is computed at the start of year t, and then applied
        # to the change in capital stock and TFP over year t to year t+1
        
        # in year t, we are assume that the damage to capital stock occurs before production occurs
        # so that production in year t is based on the capital stock after climate damage
        # and before investment occurs
        y[t] = a[t] * np.maximum(0, k[t]*k_climate[t])**alpha * l[t]**(1-alpha) * y_climate[t]

        # capital stock is then updated based on savings, depereciation, and climate damage
        k[t+1] = (k[t] * k_climate[t]) + s * y[t] - delta * k[t] 

        # apply climate effect to tfp growth rate
        a[t+1] = a[t] * tfp_growth[t] * tfp_climate[t+1]  # tfp is during the year t to t+1

    # compute the last year's output
    t = len(y)-1
    y[t] = a[t] * np.maximum(0, k[t]*k_climate[t])**alpha * l[t]**(1-alpha) * y_climate[t]

    return y, a, k, y_climate, tfp_climate, k_climate


def optimize_climate_response_scaling(
        gridcell_data, params: ModelParams, scaling: ScalingParams,
        target_period, target_type, prediction_start_year):
    """
    Optimize the scaling factor with adaptive bounds expansion.

    Performs initial optimization within specified bounds. If the result hits the bounds
    (within tolerance), expands search in the indicated direction by factor of 5 and
    retries optimization. Avoids re-searching the original parameter space.
    Returns the better of the two optimization results.

    Returns (optimal_scale, final_error, scaled_params).

    Parameters
    ----------
    gridcell_data : dict
        Grid cell time series data
    params : ModelParams
        Model parameters
    scaling : ScalingParams
        Scaling parameters
    target_period : dict
        Dictionary with 'start_year' and 'end_year' for target period
    target_type : str
        Type of optimization target: "damage" for GDP damage amount (default),
        "variability" for variability targets (to be implemented)
    prediction_start_year : int
        Year when prediction period starts (e.g., 2015)

    Notes
    -----
    The adaptive bounds expansion uses efficient directional search:
    - Hit lower bound: new search region is [10×lower, old_lower]
    - Hit upper bound: new search region is [old_upper, 10×upper]
    This avoids re-searching the original bounds while expanding in the promising direction.
    """
    # Define optimization parameters
    x0 = -0.001  # starting guess for the scale
    bounds = (-0.1, 0.1)  # initial bounds for optimization (will be expanded if needed)
    maxiter = 500  # maximum iterations per optimization attempt
    tol = 1e-8  # tolerance for optimization

    # Ensure starting guess is inside bounds
    lo, hi = bounds
    x0 = float(np.clip(x0, lo, hi))

    # Precompute target period indices
    start_year = target_period['start_year']
    end_year = target_period['end_year']
    years = gridcell_data['years']

    # Find index of prediction_start_year
    prediction_start_idx = np.where(years >= prediction_start_year)[0][0]
    target_mask = (years >= start_year) & (years <= end_year)
    target_indices = np.where(target_mask)[0]

    if len(target_indices) == 0:
        raise ValueError(f"No years found in target period {start_year}-{end_year}")


    def objective_damage(xarr):
        # xarr is a length-1 array because we're using scipy.optimize.minimize
        scale = float(xarr[0])

        # Create scaled parameters using helper function
        pc = create_scaled_params(params, scaling, scale)

        # Climate run
        y_climate, *_ = calculate_coin_ssp_forward_model(
            gridcell_data['tfp_baseline'], gridcell_data['population'],
            gridcell_data['tas'], gridcell_data['pr'], pc
        )

        # Weather (baseline) run
        y_weather, *_ = calculate_coin_ssp_forward_model(
            gridcell_data['tfp_baseline'], gridcell_data['population'],
            gridcell_data['tas_weather'], gridcell_data['pr_weather'], pc
        )

        # Calculate ratios for all years in target period
        ratios = y_climate[target_indices] / (y_weather[target_indices] + RATIO_EPSILON)
        mean_ratio = np.mean(ratios)

        target = 1.0 + params.amount_scale
        objective_value = (mean_ratio - target) ** 2
        return objective_value

    def objective_variability(xarr):
        # xarr is a length-1 array because we're using scipy.optimize.minimize
        scale = float(xarr[0])

        # Create scaled parameters using helper function
        pc = create_scaled_params(params, scaling, scale)

        # Weather (baseline) run
        y_weather, *_ = calculate_coin_ssp_forward_model(
            gridcell_data['tfp_baseline'], gridcell_data['population'],
            gridcell_data['tas_weather'], gridcell_data['pr_weather'], pc
        )

        # Split time series into historical and prediction periods using pre-computed index
        # Calculate historical period variability (before prediction period)
        stddev_y_weather = np.std(y_weather[:prediction_start_idx])

        objective_value = (hist_var_ratio - pred_var_ratio) ** 2
        return objective_value

    # Choose objective function based on target type
    if target_type == "damage":
        objective_func = objective_damage
    elif target_type == "variability":
        objective_func = objective_variability
    else:
        raise ValueError(f"Unknown target_type '{target_type}'. Must be 'damage' or 'variability'.")

    # Initial optimization with original bounds
    res = minimize(
        objective_func,
        x0=[x0],
        bounds=[bounds],                # keeps search inside [lo, hi]
        method="L-BFGS-B",              # supports bounds + numeric gradient
        options={"maxiter": maxiter},
        tol=tol
    )

    optimal_scale = float(res.x[0])
    final_error = float(res.fun)

    # Check if optimization hit the bounds and retry with expanded bounds if needed
    lo, hi = bounds
    bound_tolerance = 1e-6  # Consider "at bounds" if within this tolerance

    if (optimal_scale <= lo + bound_tolerance) or (optimal_scale >= hi - bound_tolerance):
        # Hit bounds - expand by factor of 10 and retry
        expansion_factor = 10.0

        # Diagnostic: track bounds expansion
        bound_type = "lower" if optimal_scale <= lo + bound_tolerance else "upper"

        if optimal_scale <= lo + bound_tolerance:
            # Hit lower bound - search lower region, avoid re-searching upper region
            new_lo = lo * expansion_factor if lo < 0 else lo / expansion_factor  # Expand downward 5x
            new_hi = lo  # Upper bound becomes the old lower bound
        else:
            # Hit upper bound - search higher region, avoid re-searching lower region
            new_lo = hi  # Lower bound becomes the old upper bound
            new_hi = hi * expansion_factor if hi > 0 else hi / expansion_factor  # Expand upward 5x

        expanded_bounds = (new_lo, new_hi)

        # Retry optimization with expanded bounds
        res_expanded = minimize(
            objective_func,
            x0=[optimal_scale],  # Start from previous result
            bounds=[expanded_bounds],
            method="L-BFGS-B",
            options={"maxiter": maxiter},
            tol=tol
        )

        # Use expanded result if it's better (lower error)
        if res_expanded.success and res_expanded.fun < final_error:
            old_scale = optimal_scale
            old_error = final_error
            optimal_scale = float(res_expanded.x[0])
            final_error = float(res_expanded.fun)

            # Print improvement notice (uncomment for debugging)
            # print(f"    Bounds expansion improved result: {bound_type} bound hit, "
            #       f"scale {old_scale:.6f} → {optimal_scale:.6f}, error {old_error:.6e} → {final_error:.6e}")
        else:
            # Expansion didn't help, keep original result
            pass
            # print(f"    Bounds expansion at {bound_type} bound didn't improve result, keeping original")

        # Optional: Could add another round of expansion if still hitting bounds

    scaled_params = create_scaled_params(params, scaling, optimal_scale)
    return optimal_scale, final_error, scaled_params