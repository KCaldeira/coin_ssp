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
from coin_ssp_utils import apply_time_series_filter, filter_scaling_params

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
    
    y_climate = 1.0
    y_climate += ( y_tas1 * tas  + y_tas2 * tas**2 ) -  ( y_tas1 * tas0  + y_tas2 * tas0**2 ) # units of fraction of capital
    y_climate += (y_pr1 * pr + y_pr2 * pr )**2  - (y_pr1 * pr0 + y_pr2 * pr0 )**2 # units of fraction of capital  
    k_climate = 1.0 
    k_climate += ( k_tas1 * tas  + k_tas2 * tas**2 ) -  ( k_tas1 * tas0  + k_tas2 * tas0**2 ) # units of fraction of capital
    k_climate += (k_pr1 * pr + k_pr2 * pr )**2  - (k_pr1 * pr0 + k_pr2 * pr0 )**2 # units of fraction of capital  
    tfp_climate = 1.0
    tfp_climate += ( tfp_tas1 * tas  + tfp_tas2 * tas**2 ) -  ( tfp_tas1 * tas0  + tfp_tas2 * tas0**2 ) # units of fraction of capital
    tfp_climate += (tfp_pr1 * pr + tfp_pr2 * pr )**2  - (tfp_pr1 * pr0 + tfp_pr2 * pr0 )**2 # units of fraction of capital  


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
    from coin_ssp_utils import get_ssp_data, get_grid_metadata

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


def calculate_reference_gdp_climate_variability(
    all_data, config, reference_tfp, response_scalings
):
    """
    Compute reference relationship between climate variability and GDP variability.

    Performs lat-lon loop with optimization to establish the reference relationship:
    y_weather ~ tas_weather linear regression for each grid cell.

    For each valid grid cell:
    1. Run optimization to get scaling factors (using first response function as reference)
    2. Compute y_weather time series using optimal scaling
    3. Compute linear regression: y_weather ~ tas_weather over historical period
    4. Store slope as reference relationship

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
    reference_slopes : np.ndarray
        Linear regression slopes [lat, lon] for y_weather ~ tas_weather relationship
    """

    print("Computing reference GDP-climate variability relationship...")

    # Extract data from all_data structure
    from coin_ssp_utils import get_ssp_data, get_grid_metadata

    # Get reference SSP from config
    reference_ssp = config['ssp_scenarios']['reference_ssp']

    tas_data = get_ssp_data(all_data, reference_ssp, 'tas')
    pr_data = get_ssp_data(all_data, reference_ssp, 'pr')
    pop_data = get_ssp_data(all_data, reference_ssp, 'pop')
    gdp_data = get_ssp_data(all_data, reference_ssp, 'gdp')
    tas_weather_data = all_data[reference_ssp]['tas_weather']
    pr_weather_data = all_data[reference_ssp]['pr_weather']
    years = all_data['years']

    # Get reference climate baselines from all_data
    tas0_2d = all_data['tas0_2d']
    pr0_2d = all_data['pr0_2d']

    # Extract TFP data
    valid_mask = reference_tfp['valid_mask']
    tfp_baseline = reference_tfp['tfp_baseline']

    # Get dimensions
    nlat, nlon = valid_mask.shape
    reference_slopes = np.full((nlat, nlon), np.nan)

    # Use first response function as reference for optimization
    reference_response_scaling = response_scalings[0]
    scaling_config = filter_scaling_params(reference_response_scaling)
    scaling_params = ScalingParams(**scaling_config)

    # Get historical period indices for regression
    time_periods = config['time_periods']
    hist_start_year = time_periods['historical_period']['start_year']
    hist_end_year = time_periods['historical_period']['end_year']

    hist_start_idx = np.where(years >= hist_start_year)[0][0]
    hist_end_idx = np.where(years <= hist_end_year)[0][-1]

    # Get reference period indices for filtering
    ref_start_year = time_periods['reference_period']['start_year']
    ref_end_year = time_periods['reference_period']['end_year']
    ref_start_idx = np.where(years == ref_start_year)[0][0]
    ref_end_idx = np.where(years == ref_end_year)[0][0]

    # Create dummy GDP target for optimization with exact specifications
    dummy_gdp_target = {
        'target_type': 'damage',
        'target_shape': 'constant',
        'gdp_amount': -0.10,  # 10% reduction as reference
        'target_name': 'variability_reference'
    }

    # Create dummy target_results for the reference case
    # We need a constant reduction array for the reference optimization
    constant_reduction = np.full((nlat, nlon), -0.10)  # 10% reduction everywhere
    dummy_target_results = {
        'variability_reference': {
            'reduction_array': constant_reduction
        }
    }

    # Initialize arrays for the optimization process
    n_response_functions = len(response_scalings)
    n_targets = 1  # dummy target for reference computation
    scaling_factors = np.zeros((nlat, nlon, n_response_functions, n_targets))
    optimization_errors = np.zeros((nlat, nlon, n_response_functions, n_targets))
    convergence_flags = np.zeros((nlat, nlon, n_response_functions, n_targets), dtype=bool)
    scaled_parameters = np.zeros((nlat, nlon, n_response_functions, n_targets), dtype=object)
    total_grid_cells = 0
    successful_optimizations = 0

    # Run process_response_target_optimization to get scaled_parameters
    print("Running reference optimization for variability relationship...")
    reference_results = process_response_target_optimization(
        0,  # target_idx (dummy)
        dummy_gdp_target,
        dummy_target_results,
        response_scalings,
        tas_data, pr_data, pop_data, gdp_data,
        reference_tfp, valid_mask, tfp_baseline, years, config,
        scaling_factors, optimization_errors, convergence_flags, scaled_parameters,
        total_grid_cells, successful_optimizations,
        tas_weather_data, pr_weather_data
    )

    # Extract scaled_parameters from results
    scaled_parameters = reference_results['scaled_parameters']

    processed_cells = 0
    valid_cells = np.sum(valid_mask)

    for lat_idx in range(nlat):
        # Progress indicator
        if lat_idx % 10 == 0:
            print(f"  Processing latitude band {lat_idx+1}/{nlat} ({100*processed_cells/valid_cells:.1f}% of valid cells)")

        for lon_idx in range(nlon):
            # Check if grid cell is valid
            if not valid_mask[lat_idx, lon_idx]:
                continue

            processed_cells += 1

            # Extract time series for this grid cell
            cell_tas = tas_data[:, lat_idx, lon_idx]  # [time]
            cell_pr = pr_data[:, lat_idx, lon_idx]  # [time]
            cell_pop = pop_data[:, lat_idx, lon_idx]  # [time]
            cell_gdp = gdp_data[:, lat_idx, lon_idx]  # [time]
            cell_tfp_baseline = tfp_baseline[:, lat_idx, lon_idx]  # [time]

            # Extract weather data from pre-computed arrays
            cell_tas_weather = tas_weather_data[:, lat_idx, lon_idx]
            cell_pr_weather = pr_weather_data[:, lat_idx, lon_idx]

            try:
                # Use pre-computed scaled parameters from reference optimization
                # scaled_parameters is [lat, lon, response_idx] - use first response function (response_idx=0)
                params_scaled = scaled_parameters[lat_idx, lon_idx, 0]

                # Compute y_weather time series using optimal scaling
                y_weather, *_ = calculate_coin_ssp_forward_model(
                    cell_tfp_baseline, cell_pop, cell_tas_weather, cell_pr_weather, params_scaled
                )

                # Extract historical period data for regression
                tas_weather_hist = cell_tas_weather[hist_start_idx:hist_end_idx+1]
                y_weather_hist = y_weather[hist_start_idx:hist_end_idx+1]

                # Perform linear regression: y_weather ~ tas_weather
                if len(tas_weather_hist) > 10 and len(y_weather_hist) > 10:  # Need sufficient data
                    slope, intercept, r_value, p_value, std_err = stats.linregress(tas_weather_hist, y_weather_hist)
                    reference_slopes[lat_idx, lon_idx] = slope
                else:
                    # Insufficient data
                    reference_slopes[lat_idx, lon_idx] = 0.0

            except Exception as e:
                # Optimization failed
                print(f"    Warning: Optimization failed at grid cell ({lat_idx}, {lon_idx}): {e}")
                reference_slopes[lat_idx, lon_idx] = 0.0

    print(f"Reference relationship computation complete.")
    print(f"  Valid slope estimates: {np.sum(np.isfinite(reference_slopes) & (reference_slopes != 0))}/{valid_cells}")

    return reference_slopes


def apply_variability_target_scaling(
    reference_slopes, gdp_target, tas_data, pr_data,
    tas0_2d, pr0_2d, target_idx, response_scalings,
    scaling_factors, optimization_errors, convergence_flags, scaled_parameters
):
    """
    Apply variability target scaling using reference relationship.

    Uses the reference slopes and current temperature data to compute variability effects
    for the specified target, avoiding expensive optimization.

    The variability scaling applies the relationship:
    scaling_factor = reference_slope * target_scaling_parameter

    Where target_scaling_parameter comes from the gdp_target configuration.

    Parameters
    ----------
    reference_slopes : np.ndarray
        Reference slopes [lat, lon] from calculate_reference_gdp_climate_variability
    gdp_target : dict
        Target configuration with variability parameters
    tas_data, pr_data : np.ndarray
        Climate data [time, lat, lon]
    tas0_2d, pr0_2d : np.ndarray
        Reference baselines [lat, lon]
    target_idx : int
        Target index for result storage
    response_scalings : list
        Damage scaling configurations
    scaling_factors, optimization_errors, convergence_flags, scaled_parameters : np.ndarray
        Output arrays to populate (modified in place)

    Returns
    -------
    dict
        Results summary
    """
    nlat, nlon = tas0_2d.shape
    n_response_functions = len(response_scalings)

    target_name = gdp_target['target_name']
    print(f"Applying variability target: {target_name}")

    # Extract target scaling parameter from gdp_target configuration
    # For now, use the same parameter naming as damage targets
    # TODO: Define specific variability parameter names
    target_scaling_param = gdp_target.get('gdp_amount', 1.0)  # Default multiplier

    print(f"  Target scaling parameter: {target_scaling_param}")
    print(f"  Reference slopes range: {np.nanmin(reference_slopes):.6f} to {np.nanmax(reference_slopes):.6f}")

    # Apply scaling to all response functions (same relationship applies to all)
    for response_idx in range(n_response_functions):
        # Compute variability scaling factors using reference relationship
        # scaling_factor = reference_slope * target_parameter
        target_scaling_factors = reference_slopes * target_scaling_param

        # Store results in output arrays
        scaling_factors[:, :, response_idx, target_idx] = target_scaling_factors

        # For variability targets, set optimization error to zero (no optimization performed)
        optimization_errors[:, :, response_idx, target_idx] = 0.0

        # Mark as converged where we have valid reference slopes
        convergence_flags[:, :, response_idx, target_idx] = np.isfinite(reference_slopes)

        # For scaled_parameters, we don't have meaningful values for variability targets
        # Leave them as NaN to indicate they're not applicable
        # (variability targets use the reference relationship directly)

    # Summary statistics
    valid_cells = np.sum(np.isfinite(reference_slopes))
    applied_cells = np.sum(np.isfinite(target_scaling_factors))

    print(f"  Applied to {applied_cells}/{valid_cells} valid grid cells")
    if applied_cells > 0:
        scaling_range = target_scaling_factors[np.isfinite(target_scaling_factors)]
        print(f"  Scaling factors range: {np.min(scaling_range):.6f} to {np.max(scaling_range):.6f}")

    return {
        'variability_target_processed': True,
        'target_name': target_name,
        'applied_cells': applied_cells,
        'valid_cells': valid_cells
    }