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
from scipy import stats
from coin_ssp_utils import apply_time_series_filter, filter_scaling_params, get_ssp_data, get_grid_metadata
from coin_ssp_math_utils import apply_loess_divide

from coin_ssp_models import ScalingParams, ModelParams

# Small epsilon to prevent division by zero in ratio calculations
RATIO_EPSILON = 1e-20

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
    g0 = params.g0  # GDP variability scaling constant term
    g1 = params.g1  # GDP variability scaling linear term
    g2 = params.g2  # GDP variability scaling quadratic term

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

    # Calculate variability scaling factors for current conditions and reference
    # Note: For "damage" targets, g0=1.0, g1=0.0, g2=0.0, so g_scaling = g_ref_scaling = 1.0
    # The g(T) scaling is only used for "variability" targets to scale GDP-weather relationships by temperature
    g_scaling = g0 + g1 * tas + g2 * tas**2
    g_ref_scaling = g0 + g1 * tas0 + g2 * tas0**2

    # Define climate response functions f_y, f_k, f_tfp
    def f_y(T, P):
        return (y_tas1 * T + y_tas2 * T**2) + (y_pr1 * P + y_pr2 * P**2)

    def f_k(T, P):
        return (k_tas1 * T + k_tas2 * T**2) + (k_pr1 * P + k_pr2 * P**2)

    def f_tfp(T, P):
        return (tfp_tas1 * T + tfp_tas2 * T**2) + (tfp_pr1 * P + tfp_pr2 * P**2)

    # Calculate climate response factors using cleaner formulation
    y_climate = 1.0 + g_scaling * f_y(tas, pr) - g_ref_scaling * f_y(tas0, pr0)
    k_climate = 1.0 + g_scaling * f_k(tas, pr) - g_ref_scaling * f_k(tas0, pr0)
    tfp_climate = 1.0 + g_scaling * f_tfp(tas, pr) - g_ref_scaling * f_tfp(tas0, pr0)  


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
        config, gdp_target):
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
    config : dict
        Configuration dictionary containing time_periods and other optimization settings
    gdp_target : dict
        Current GDP target configuration containing target_type and other target-specific settings

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

    # Extract configuration parameters
    target_period = config['time_periods']['target_period']
    target_type = gdp_target['target_type']
    historical_end_year = config['time_periods']['historical_period']['end_year']

    # Precompute target period indices
    start_year = target_period['start_year']
    end_year = target_period['end_year']
    years = gridcell_data['years']

    # Find index of historical period end (for potential future use in variability calculations)
    historical_end_idx = np.where(years <= historical_end_year)[0][-1]
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
            gridcell_data['tfp_baseline'], gridcell_data['pop'],
            gridcell_data['tas'], gridcell_data['pr'], pc
        )

        # Weather (baseline) run
        y_weather, *_ = calculate_coin_ssp_forward_model(
            gridcell_data['tfp_baseline'], gridcell_data['pop'],
            gridcell_data['tas_weather'], gridcell_data['pr_weather'], pc
        )

        # Calculate ratios for all years in target period
        ratios = y_climate[target_indices] / (y_weather[target_indices] + RATIO_EPSILON)
        mean_ratio = np.mean(ratios)

        target = 1.0 + gdp_target['gdp_amount']
        objective_value = (mean_ratio - target) ** 2
        return objective_value


    # Initial optimization with original bounds
    res = minimize(
        objective_damage,
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
            objective_damage,
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

            # Bounds expansion improved result
        else:
            # Expansion didn't help, keep original result
            pass

        # Optional: Could add another round of expansion if still hitting bounds

    scaled_params = create_scaled_params(params, scaling, optimal_scale)
    return optimal_scale, final_error, scaled_params


def process_response_target_optimization(
    target_idx, gdp_target, target_results, response_scalings,
    tas_data, pr_data, pop_data, gdp_data,
    reference_tfp, valid_mask, tfp_baseline, years, config,
    scaling_factors, optimization_errors, convergence_flags, scaled_parameters,
    total_grid_cells, successful_optimizations,
    tas_weather_data, pr_weather_data
):
    """
    Process optimization for a single damage target across all response functions and grid cells.

    This function encapsulates the nested loops and per-grid-cell optimization that was
    previously inline in main.py (lines 622-700). It handles all response functions for
    a single GDP target.

    Parameters
    ----------
    target_idx : int
        Index of current GDP target
    gdp_target : dict
        GDP target configuration
    target_results : dict
        Target GDP results containing reduction arrays
    response_scalings : list
        List of damage scaling configurations
    tas_data, pr_data, pop_data, gdp_data : np.ndarray
        Climate and economic data arrays [time, lat, lon]
    reference_tfp, valid_mask, tfp_baseline : np.ndarray
        TFP reference data
    years : np.ndarray
        Years array
    config : dict
        Configuration dictionary
    scaling_factors, optimization_errors, convergence_flags, scaled_parameters : np.ndarray
        Output arrays to populate (modified in place)
    tas_weather_data, pr_weather_data : np.ndarray
        Pre-computed weather variables [time, lat, lon]
    total_grid_cells, successful_optimizations : int
        Counters (modified in place via list trick)

    Returns
    -------
    dict
        Updated counters: {'total_grid_cells': int, 'successful_optimizations': int}
    """

    target_name = gdp_target['target_name']
    target_reduction_array = target_results[target_name]['reduction_array']  # [lat, lon]

    print(f"\nProcessing GDP target: {target_name} ({target_idx+1}/?)")

    # Calculate reference period indices from config
    time_periods = config['time_periods']
    ref_start_year = time_periods['reference_period']['start_year']
    ref_end_year = time_periods['reference_period']['end_year']
    ref_start_idx = np.where(years == ref_start_year)[0][0]
    ref_end_idx = np.where(years == ref_end_year)[0][0]

    # Get dimensions
    nlat, nlon = valid_mask.shape
    n_response_functions = len(response_scalings)

    for response_idx, response_scaling in enumerate(response_scalings):
        scaling_name = response_scaling['scaling_name']
        print(f"  Response function: {scaling_name} ({response_idx+1}/{n_response_functions})")

        # Create ScalingParams for this response function
        scaling_config = filter_scaling_params(response_scaling)
        scaling_params = ScalingParams(**scaling_config)

        for lat_idx in range(nlat):
            # Progress indicator: print dot for each latitude band
            print(".", end="", flush=True)

            for lon_idx in range(nlon):

                # Check if grid cell is valid (has economic activity)
                if not valid_mask[lat_idx, lon_idx]:
                    continue

                total_grid_cells += 1

                # Extract time series for this grid cell (climate data is [time, lat, lon])
                cell_tas = tas_data[:, lat_idx, lon_idx]  # [time]
                cell_pr = pr_data[:, lat_idx, lon_idx]  # [time]
                cell_pop = pop_data[:, lat_idx, lon_idx]  # [time]
                cell_gdp = gdp_data[:, lat_idx, lon_idx]  # [time]
                cell_tfp_baseline = tfp_baseline[:, lat_idx, lon_idx]  # [time] (data is [time, lat, lon])

                # Get target reduction for this grid cell
                target_reduction = target_reduction_array[lat_idx, lon_idx]

                # Get weather (filtered) time series from pre-computed arrays
                cell_tas_weather = tas_weather_data[:, lat_idx, lon_idx]
                cell_pr_weather = pr_weather_data[:, lat_idx, lon_idx]

                # Create parameters for this grid cell using factory
                params_cell = config['model_params_factory'].create_for_step(
                    "grid_cell_optimization",
                    tas0=np.mean(cell_tas[ref_start_idx:ref_end_idx+1]),
                    pr0=np.mean(cell_pr[ref_start_idx:ref_end_idx+1])
                )

                # Create cell data dictionary matching gridcell_data structure
                cell_data = {
                    'years': years,
                    'pop': cell_pop,
                    'gdp': cell_gdp,
                    'tas': cell_tas,
                    'pr': cell_pr,
                    'tas_weather': cell_tas_weather,
                    'pr_weather': cell_pr_weather,
                    'tfp_baseline': cell_tfp_baseline
                }

                # Run per-grid-cell optimization
                optimal_scale, final_error, params_scaled = optimize_climate_response_scaling(
                    cell_data, params_cell, scaling_params, config, gdp_target
                )

                # Store results
                scaling_factors[lat_idx, lon_idx, response_idx, target_idx] = optimal_scale
                optimization_errors[lat_idx, lon_idx, response_idx, target_idx] = final_error
                convergence_flags[lat_idx, lon_idx, response_idx, target_idx] = True

                # Store scaled response function parameters
                scaled_parameters[lat_idx, lon_idx, response_idx, target_idx, 0] = params_scaled.k_tas1
                scaled_parameters[lat_idx, lon_idx, response_idx, target_idx, 1] = params_scaled.k_tas2
                scaled_parameters[lat_idx, lon_idx, response_idx, target_idx, 2] = params_scaled.k_pr1
                scaled_parameters[lat_idx, lon_idx, response_idx, target_idx, 3] = params_scaled.k_pr2
                scaled_parameters[lat_idx, lon_idx, response_idx, target_idx, 4] = params_scaled.tfp_tas1
                scaled_parameters[lat_idx, lon_idx, response_idx, target_idx, 5] = params_scaled.tfp_tas2
                scaled_parameters[lat_idx, lon_idx, response_idx, target_idx, 6] = params_scaled.tfp_pr1
                scaled_parameters[lat_idx, lon_idx, response_idx, target_idx, 7] = params_scaled.tfp_pr2
                scaled_parameters[lat_idx, lon_idx, response_idx, target_idx, 8] = params_scaled.y_tas1
                scaled_parameters[lat_idx, lon_idx, response_idx, target_idx, 9] = params_scaled.y_tas2
                scaled_parameters[lat_idx, lon_idx, response_idx, target_idx, 10] = params_scaled.y_pr1
                scaled_parameters[lat_idx, lon_idx, response_idx, target_idx, 11] = params_scaled.y_pr2

                successful_optimizations += 1

        # Newline after each response function completes its latitude bands
        print()

    return {
        'total_grid_cells': total_grid_cells,
        'successful_optimizations': successful_optimizations
    }


def calculate_reference_climate_baselines(all_data, config):
    """
    Calculate reference climate baselines (tas0, pr0) as 2D arrays for all grid cells.

    This computes the reference period mean temperature and precipitation for each grid cell,
    following the same approach as the per-grid-cell optimization but computed once for reuse.

    Parameters
    ----------
    all_data : dict
        Complete data structure containing all SSP climate data
    config : dict
        Configuration dictionary containing time_periods and ssp_scenarios

    Returns
    -------
    tas0_2d, pr0_2d : np.ndarray
        Reference baselines as 2D arrays [lat, lon]
    """
    # Extract data from all_data structure
    # Get reference SSP from config
    reference_ssp = config['ssp_scenarios']['reference_ssp']

    tas_data = get_ssp_data(all_data, reference_ssp, 'tas')
    pr_data = get_ssp_data(all_data, reference_ssp, 'pr')
    years = all_data['years']

    # Get reference period indices
    time_periods = config['time_periods']
    ref_start_year = time_periods['reference_period']['start_year']
    ref_end_year = time_periods['reference_period']['end_year']

    ref_start_idx = np.where(years == ref_start_year)[0][0]
    ref_end_idx = np.where(years == ref_end_year)[0][0]

    # Calculate reference period means for all grid cells
    tas0_2d = np.mean(tas_data[ref_start_idx:ref_end_idx+1, :, :], axis=0)  # [lat, lon]
    pr0_2d = np.mean(pr_data[ref_start_idx:ref_end_idx+1, :, :], axis=0)  # [lat, lon]

    return tas0_2d, pr0_2d


def calculate_variability_climate_response_parameters(
    all_data, config, reference_tfp, response_scalings
):
    """
    Calculate climate response parameters for variability targets using a 4-step calibration process.

    This function determines the climate response parameters that will be used for variability
    scaling by establishing the relationship between weather variability and economic impacts.

    ALGORITHM OVERVIEW:
    ==================

    Step 1: OPTIMIZATION FOR UNIFORM 10% GDP LOSS
    ---------------------------------------------
    - Run optimization to find scaling factors that produce uniform 10% GDP loss in target period
    - This establishes the baseline strength of climate-economy relationship needed for target impact
    - Uses dummy target with 10% constant reduction across all grid cells
    - Outputs: scaling factors for each response function parameter

    Step 2: FORWARD MODEL SIMULATIONS WITH SCALED PARAMETERS
    -------------------------------------------------------
    - Take parameters from Step 1 optimization, scaled by the found factors
    - Run forward model simulations using WEATHER COMPONENTS (tas_weather, pr_weather)
    - This isolates weather variability effects from long-term climate trends
    - Generate economic projections over the full time period (historical + future)
    - Outputs: time series of economic variables (GDP, capital, TFP) for each grid cell

    Step 3: WEATHER VARIABILITY REGRESSION ANALYSIS
    -----------------------------------------------
    - For each grid cell, compute regression: (GDP / LOESS_smoothed_GDP) ~ tas_weather over historical period
    - GDP = actual GDP from forward model simulation
    - LOESS_smoothed_GDP = 30-year LOESS smoothed trend of GDP
    - tas_weather = weather component of temperature (detrended, LOESS-filtered climate signal)
    - Regression slope = fractional change in GDP per degree C of weather variability
    - This quantifies the actual historical relationship between weather and economic fluctuations

    Step 4: PARAMETER NORMALIZATION BY REGRESSION SLOPE
    --------------------------------------------------
    - Divide all climate response parameters from Phase 1 by the regression slope from Phase 3
    - This normalizes parameters so they represent the correct strength per degree of variability
    - Final parameters capture both the target impact magnitude AND the observed weather sensitivity
    - Result: climate parameters calibrated for variability target applications

    Parameters
    ----------
    all_data : dict
        Complete data structure containing all SSP climate and economic data
    config : dict
        Configuration dictionary containing ssp_scenarios and time_periods
    reference_tfp : dict
        TFP reference data containing valid_mask and tfp_baseline
    response_scalings : list
        List of damage scaling configurations (for optimization)

    Returns
    -------
    baseline_climate_parameters : np.ndarray
        Climate response parameters [lat, lon, n_params] calibrated for variability targets.
        Parameters are normalized by weather-GDP regression slopes from historical data.
    """

    print("Computing climate response parameters for variability targets...")
    print("Using 4-step calibration process:")
    print("  1. Optimization for uniform 10% GDP loss")
    print("  2. Forward model simulations with scaled parameters")
    print("  3. Weather variability regression analysis")
    print("  4. Parameter normalization by regression slope")

    # =================================================================================
    # STEP 1: OPTIMIZATION FOR UNIFORM 10% GDP LOSS
    # =================================================================================
    print("\n--- Step 1: Optimization for uniform 10% GDP loss ---")

    # Extract data from all_data structure
    reference_ssp = config['ssp_scenarios']['reference_ssp']
    tas_data = get_ssp_data(all_data, reference_ssp, 'tas')
    pr_data = get_ssp_data(all_data, reference_ssp, 'pr')
    pop_data = get_ssp_data(all_data, reference_ssp, 'pop')
    gdp_data = get_ssp_data(all_data, reference_ssp, 'gdp')
    tas_weather_data = all_data[reference_ssp]['tas_weather']
    pr_weather_data = all_data[reference_ssp]['pr_weather']
    years = all_data['years']
    tas0_2d = all_data['tas0_2d']
    pr0_2d = all_data['pr0_2d']

    # Extract TFP data
    valid_mask = reference_tfp['valid_mask']
    tfp_baseline = reference_tfp['tfp_baseline']
    nlat, nlon = valid_mask.shape

    # Create dummy GDP target for optimization (uniform 10% loss)
    dummy_gdp_target = {
        'target_type': 'damage',
        'target_shape': 'constant',
        'gdp_amount': -0.10,
        'target_name': 'variability_reference'
    }

    constant_reduction = np.full((nlat, nlon), -0.10)
    dummy_target_results = {
        'variability_reference': {
            'reduction_array': constant_reduction
        }
    }

    # Initialize arrays for optimization
    n_response_functions = len(response_scalings)
    n_targets = 1
    n_params = 12
    scaling_factors = np.zeros((nlat, nlon, n_response_functions, n_targets))
    optimization_errors = np.zeros((nlat, nlon, n_response_functions, n_targets))
    convergence_flags = np.zeros((nlat, nlon, n_response_functions, n_targets), dtype=bool)
    scaled_parameters = np.zeros((nlat, nlon, n_response_functions, n_targets, n_params))

    # Run optimization to find scaling factors for uniform 10% GDP loss
    # Uses full climate data (tas_data, pr_data) for optimization
    reference_results = process_response_target_optimization(
        0, dummy_gdp_target, dummy_target_results, response_scalings,
        tas_data, pr_data, pop_data, gdp_data, reference_tfp, valid_mask, tfp_baseline,
        years, config, scaling_factors, optimization_errors, convergence_flags,
        scaled_parameters, 0, 0, tas_weather_data, pr_weather_data
    )

    # Extract optimized parameters (Step 1 results)
    step1_parameters = scaled_parameters[:, :, 0, 0, :]  # [lat, lon, n_params]
    step1_success_mask = convergence_flags[:, :, 0, 0]

    valid_cells = np.sum(valid_mask)
    step1_success = np.sum(step1_success_mask)
    print(f"Phase 1 complete: {step1_success}/{valid_cells} successful optimizations")

    # =================================================================================
    # PHASE 2: FORWARD MODEL SIMULATIONS WITH SCALED PARAMETERS
    # =================================================================================
    print("\n--- Phase 2: Forward model simulations with scaled parameters ---")

    # Get time period indices
    time_periods = config['time_periods']
    hist_start_year = time_periods['historical_period']['start_year']
    hist_end_year = time_periods['historical_period']['end_year']
    hist_start_idx = np.where(years >= hist_start_year)[0][0]
    hist_end_idx = np.where(years <= hist_end_year)[0][-1]
    n_hist_years = hist_end_idx - hist_start_idx + 1

    # Initialize arrays for forward model results
    gdp_forward = np.zeros((len(years), nlat, nlon))

    print(f"Running forward model for {np.sum(step1_success_mask)} grid cells...")

    # Run forward model for each successfully optimized grid cell
    for lat_idx in range(nlat):
        for lon_idx in range(nlon):
            if not (valid_mask[lat_idx, lon_idx] and step1_success_mask[lat_idx, lon_idx]):
                continue

            # Extract model parameters for this cell from Step 1
            cell_params = step1_parameters[lat_idx, lon_idx, :]

            # Create ModelParams object with optimized climate response parameters
            model_params = ModelParams(
                s=config['model_params']['s'],
                alpha=config['model_params']['alpha'],
                delta=config['model_params']['delta'],
                tas0=tas0_2d[lat_idx, lon_idx],
                pr0=pr0_2d[lat_idx, lon_idx],
                k_tas1=cell_params[0], k_tas2=cell_params[1],
                k_pr1=cell_params[2], k_pr2=cell_params[3],
                tfp_tas1=cell_params[4], tfp_tas2=cell_params[5],
                tfp_pr1=cell_params[6], tfp_pr2=cell_params[7],
                y_tas1=cell_params[8], y_tas2=cell_params[9],
                y_pr1=cell_params[10], y_pr2=cell_params[11]
            )

            # Extract time series for this grid cell
            # Use WEATHER COMPONENTS for forward simulation to isolate variability effects
            cell_tas_weather = tas_weather_data[:, lat_idx, lon_idx]
            cell_pr_weather = pr_weather_data[:, lat_idx, lon_idx]
            cell_pop = pop_data[:, lat_idx, lon_idx]
            cell_gdp = gdp_data[:, lat_idx, lon_idx]
            cell_tfp_baseline = tfp_baseline[lat_idx, lon_idx]

            # Run forward model with weather components
            results = calculate_coin_ssp_forward_model(
                model_params, cell_tas_weather, cell_pr_weather, cell_pop, cell_gdp,
                cell_tfp_baseline, years
            )

            gdp_forward[:, lat_idx, lon_idx] = results['gdp_timeseries']

    print(f"Phase 2 complete: Forward model simulations generated")

    # =================================================================================
    # PHASE 3: WEATHER VARIABILITY REGRESSION ANALYSIS
    # =================================================================================
    print("\n--- Phase 3: Weather variability regression analysis ---")

    # Initialize regression slope array
    regression_slopes = np.zeros((nlat, nlon))
    regression_success_mask = np.zeros((nlat, nlon), dtype=bool)

    print(f"Computing y_weather ~ tas_weather regression for {np.sum(step1_success_mask)} cells...")

    for lat_idx in range(nlat):
        for lon_idx in range(nlon):
            if not (valid_mask[lat_idx, lon_idx] and step1_success_mask[lat_idx, lon_idx]):
                continue

            # Extract weather variables for historical period
            tas_weather_hist = tas_weather_data[hist_start_idx:hist_end_idx+1, lat_idx, lon_idx]
            gdp_forward_hist = gdp_forward[hist_start_idx:hist_end_idx+1, lat_idx, lon_idx]

            # Compute GDP ratio: actual GDP divided by 30-year LOESS smoothed GDP trend
            # Use the new apply_loess_divide function for clean, mnemonic operation
            gdp_ratio = apply_loess_divide(gdp_forward_hist, 30)

            # Remove any invalid values
            valid_data_mask = np.isfinite(gdp_ratio) & np.isfinite(tas_weather_hist)

            # Compute regression: (GDP / LOESS_smoothed_GDP) ~ tas_weather
            gdp_ratio_valid = gdp_ratio[valid_data_mask]
            tas_weather_valid = tas_weather_hist[valid_data_mask]

            # Linear regression
            if np.std(tas_weather_valid) > 1e-6:  # Check for sufficient variation
                slope, intercept = np.polyfit(tas_weather_valid, gdp_ratio_valid, 1)
                regression_slopes[lat_idx, lon_idx] = slope
                regression_success_mask[lat_idx, lon_idx] = True

    regression_success = np.sum(regression_success_mask)
    print(f"Phase 3 complete: {regression_success} successful regressions")

    # =================================================================================
    # PHASE 4: PARAMETER NORMALIZATION BY REGRESSION SLOPE
    # =================================================================================
    print("\n--- Phase 4: Parameter normalization by regression slope ---")

    # Initialize final normalized parameters
    final_parameters = np.zeros_like(step1_parameters)
    final_success_mask = step1_success_mask & regression_success_mask

    for lat_idx in range(nlat):
        for lon_idx in range(nlon):
            if not final_success_mask[lat_idx, lon_idx]:
                continue

            slope = regression_slopes[lat_idx, lon_idx]

            final_parameters[lat_idx, lon_idx, :] = step1_parameters[lat_idx, lon_idx, :] / slope

    final_success = np.sum(final_success_mask)
    print(f"Phase 4 complete: {final_success} final calibrated parameters")

    print(f"\n4-phase calibration summary:")
    print(f"  Valid grid cells: {valid_cells}")
    print(f"  Phase 1 success: {step1_success}")
    print(f"  Phase 3 success: {regression_success}")
    print(f"  Final success: {final_success}")

    return final_parameters


def calculate_variability_scaling_parameters(
    baseline_climate_parameters, gdp_target, target_idx,
    all_data, config, response_scalings,
    scaling_factors, optimization_errors, convergence_flags, scaled_parameters
):
    """
    Calculate GDP variability scaling parameters (g0, g1, g2) for variability targets.

    Computes the g0, g1, g2 parameters that define how GDP climate sensitivity varies with
    local temperature according to the target shape (constant, linear, quadratic).
    These parameters are used in the forward model as: g(T) = g0 + g1*T + g2*T²

    For different target shapes:
    - constant: g0 = target_amount, g1 = 0, g2 = 0
    - linear: g0, g1 computed from global_mean_amount and amount_at_reference_temp
    - quadratic: g0, g1, g2 computed from global_mean_amount, zero_amount_temperature, etc.

    Parameters
    ----------
    baseline_climate_parameters : np.ndarray
        Baseline climate parameters [lat, lon, n_params] from calculate_variability_climate_response_parameters
    gdp_target : dict
        Target configuration with variability parameters and target_shape
    target_idx : int
        Target index for result storage
    all_data : dict
        Complete data structure containing all SSP climate and economic data
    config : dict
        Configuration dictionary containing time periods and SSP scenarios
    response_scalings : list
        Damage scaling configurations
    scaling_factors, optimization_errors, convergence_flags, scaled_parameters : np.ndarray
        Output arrays to populate (modified in place)

    Returns
    -------
    dict
        Dictionary containing:
        - 'g0_array': np.ndarray [lat, lon] - g0 values for each grid cell
        - 'g1_array': np.ndarray [lat, lon] - g1 values for each grid cell
        - 'g2_array': np.ndarray [lat, lon] - g2 values for each grid cell
        - 'total_grid_cells': int - number of processed grid cells
        - 'successful_optimizations': int - number of successful calculations
    """
    # Extract data from all_data and config
    reference_ssp = config['ssp_scenarios']['reference_ssp']
    time_periods = config['time_periods']

    # Get climate and economic data
    tas_data = get_ssp_data(all_data, reference_ssp, 'tas')
    gdp_data = get_ssp_data(all_data, reference_ssp, 'gdp')
    years = all_data['years']
    tas0_2d = all_data['tas0_2d']

    # Get historical period indices for GDP weighting
    hist_start_year = time_periods['historical_period']['start_year']
    hist_end_year = time_periods['historical_period']['end_year']
    hist_start_idx = np.where(years >= hist_start_year)[0][0]
    hist_end_idx = np.where(years <= hist_end_year)[0][-1]

    nlat, nlon = tas0_2d.shape
    n_response_functions = len(response_scalings)

    target_name = gdp_target['target_name']
    target_shape = gdp_target['target_shape']

    print(f"\nProcessing GDP target: {target_name} ({target_idx+1}/?) - Shape: {target_shape}")

    # Step 1: Calculate mean GDP by grid cell over historical period
    mean_gdp_per_cell = np.mean(gdp_data[hist_start_idx:hist_end_idx+1, :, :], axis=0)  # [lat, lon]

    # Calculate GDP variability scaling parameters (g0, g1, g2) based on target shape
    if target_shape == 'constant':
        # For constant targets: g(T) = g0, g1 = 0, g2 = 0
        g0_value = gdp_target.get('gdp_amount', 1.0)
        print(f"  Constant GDP variability scaling: g0={g0_value}")

        g0_array = np.full((nlat, nlon), g0_value)
        g1_array = np.zeros((nlat, nlon))
        g2_array = np.zeros((nlat, nlon))

    elif target_shape == 'linear':
        # Linear case: g(T) = a0 + a1 * T, so g0 = a0, g1 = a1, g2 = 0
        global_mean_amount = gdp_target['global_mean_amount']
        reference_temperature = gdp_target['reference_temperature']
        amount_at_reference_temp = gdp_target['amount_at_reference_temp']

        print(f"  Linear target: global_mean={global_mean_amount}, ref_temp={reference_temperature}, amount_at_ref={amount_at_reference_temp}")

        # Calculate GDP-weighted mean temperature over historical period
        total_gdp = np.sum(mean_gdp_per_cell[mean_gdp_per_cell > 0])
        gdp_weighted_tas = np.sum(mean_gdp_per_cell * tas0_2d) / total_gdp

        # Calculate coefficients for linear relationship: g(T) = a0 + a1 * T
        a1 = (amount_at_reference_temp - global_mean_amount) / (reference_temperature - gdp_weighted_tas)
        a0 = global_mean_amount - a1 * gdp_weighted_tas

        print(f"  Linear coefficients: g0={a0:.6f}, g1={a1:.6f}")
        print(f"  GDP-weighted mean temperature: {gdp_weighted_tas:.2f}°C")

        g0_array = np.full((nlat, nlon), a0)
        g1_array = np.full((nlat, nlon), a1)
        g2_array = np.zeros((nlat, nlon))

    elif target_shape == 'quadratic':
        # Quadratic case: g(T) = a0 + a1*T + a2*T^2, so g0 = a0, g1 = a1, g2 = a2
        global_mean_amount = gdp_target['global_mean_amount']
        zero_amount_temperature = gdp_target['zero_amount_temperature']
        derivative_at_zero_amount_temperature = gdp_target['derivative_at_zero_amount_temperature']

        print(f"  Quadratic target: global_mean={global_mean_amount}, zero_temp={zero_amount_temperature}, deriv_at_zero={derivative_at_zero_amount_temperature}")

        # Calculate GDP-weighted mean temperature
        total_gdp = np.sum(mean_gdp_per_cell[mean_gdp_per_cell > 0])
        gdp_weighted_tas = np.sum(mean_gdp_per_cell * tas0_2d) / total_gdp
        gdp_weighted_tas2 = np.sum(mean_gdp_per_cell * tas0_2d**2) / total_gdp

        T0 = zero_amount_temperature
        T_mean = gdp_weighted_tas
        T2_mean = gdp_weighted_tas2

        # From constraints:
        # a1 + 2*a2*T0 = derivative_at_zero_amount_temperature
        # a0 + a1*T0 + a2*T0^2 = 0
        # a0 + a1*T_mean + a2*T2_mean = global_mean_amount

        # Solve the system: eliminate a0 and solve for a1, a2
        # From first two: a0 = -a1*T0 - a2*T0^2
        # Substitute into third: -a1*T0 - a2*T0^2 + a1*T_mean + a2*T2_mean = global_mean_amount
        # a1*(T_mean - T0) + a2*(T2_mean - T0^2) = global_mean_amount
        # a1 + 2*a2*T0 = derivative_at_zero_amount_temperature

        # Matrix form: [T_mean-T0, T2_mean-T0^2] [a1] = [global_mean_amount]
        #              [1,         2*T0         ] [a2]   [derivative_at_zero_amount_temperature]

        det = (T_mean - T0) * 2 * T0 - (T2_mean - T0**2) * 1
        a1 = (global_mean_amount * 2 * T0 - derivative_at_zero_amount_temperature * (T2_mean - T0**2)) / det
        a2 = (derivative_at_zero_amount_temperature * (T_mean - T0) - global_mean_amount * 1) / det
        a0 = -a1 * T0 - a2 * T0**2

        print(f"  Quadratic coefficients: g0={a0:.6f}, g1={a1:.6f}, g2={a2:.6f}")
        print(f"  GDP-weighted mean temperature: {gdp_weighted_tas:.2f}°C")

        g0_array = np.full((nlat, nlon), a0)
        g1_array = np.full((nlat, nlon), a1)
        g2_array = np.full((nlat, nlon), a2)

    else:
        raise ValueError(f"Unknown target_shape: {target_shape}")

    # Compute GDP variability scaling factors at reference temperature for each grid cell
    target_scaling_factors_array = g0_array + g1_array * tas0_2d + g2_array * tas0_2d**2

    print(f"  Variability scaling at reference temperature range: {np.nanmin(target_scaling_factors_array):.6f} to {np.nanmax(target_scaling_factors_array):.6f}")

    # Initialize counters to match process_response_target_optimization return structure
    total_grid_cells = 0
    successful_optimizations = 0

    # Loop over response functions to match process_response_target_optimization structure
    for response_idx, response_scaling in enumerate(response_scalings):
        scaling_name = response_scaling['scaling_name']
        print(f"  Response function: {scaling_name} ({response_idx+1}/{n_response_functions})")

        # Progress indicator: print dots for each latitude band (like optimization function)
        for lat_idx in range(nlat):
            print(".", end="", flush=True)

            for lon_idx in range(nlon):
                # Count valid cells where we have finite scaling factors
                if np.isfinite(target_scaling_factors_array[lat_idx, lon_idx]) and np.isfinite(baseline_climate_parameters[lat_idx, lon_idx, 0]):
                    total_grid_cells += 1
                    successful_optimizations += 1

        # Calculate scaling factors by applying target scaling to reference scaled parameters
        # scaling_factor = target_scaling_factor (no optimization, direct application)
        scaling_factors[:, :, response_idx, target_idx] = target_scaling_factors_array

        # For variability targets, set optimization error to zero (no optimization performed)
        optimization_errors[:, :, response_idx, target_idx] = 0.0

        # Mark as converged where we have valid baseline parameters and finite scaling factors
        convergence_flags[:, :, response_idx, target_idx] = (
            np.isfinite(baseline_climate_parameters[:, :, 0]) &
            np.isfinite(target_scaling_factors_array)
        )

        # Store scaled parameters by applying target scaling to baseline parameters
        # For variability targets, we scale the baseline parameters by the target scaling factor
        for param_idx in range(baseline_climate_parameters.shape[2]):
            scaled_parameters[:, :, response_idx, target_idx, param_idx] = (
                baseline_climate_parameters[:, :, param_idx] * target_scaling_factors_array
            )

        # Newline after each response function completes its latitude bands
        print()

    # Summary statistics
    valid_cells = np.sum(np.isfinite(baseline_climate_parameters[:, :, 0]))
    applied_cells = np.sum(np.isfinite(target_scaling_factors_array))

    print(f"  Applied to {applied_cells}/{valid_cells} valid grid cells")
    if applied_cells > 0:
        scaling_range = target_scaling_factors_array[np.isfinite(target_scaling_factors_array)]
        print(f"  Scaling factors range: {np.min(scaling_range):.6f} to {np.max(scaling_range):.6f}")

    return {
        'g0_array': g0_array,
        'g1_array': g1_array,
        'g2_array': g2_array,
        'total_grid_cells': total_grid_cells,
        'successful_optimizations': successful_optimizations
    }