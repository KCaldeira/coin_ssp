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
from scipy.optimize import minimize_scalar

@dataclass
# parametes for the COIN-SSP model
class ModelParams:
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

    return a, k

def calculate_coin_ssp_forward_model(tfp, pop, gdp, tas, pr, params: ModelParams):

    # This function calculates the forward model for the COIN-SSP economic model.
    # It takes in total factor productivity (tfp), population (pop), gross domestic product
    # (gdp), temperature (tas), and a set of model parameters (params).
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
    y = gdp/gdp[0] # output normalized to year 0
    l = pop/pop[0] # population normalized to year 0

    # assume that at year 0, the system is in steady-state, do d k / dt = 0 at year 0, and a[0] = 1.
    # 0 == s * y[0] - delta * k[0]
    k0 = (s/delta) # everything is non0dimensionalized to 1 at year 0
    # y[0] ==  a[0] * k[0]**alpha * l[0]**(1-alpha)
    a0 = k0**(-alpha) # nondimensionalized Total Factor Productivity is 0 in year 0
    # since we are assuming steady state, the capital stock will be the same at the start of year 1
    k = np.copy(y) # capital stock normalized to year 0
    a = np.copy(y) # total factor productivity normalized to year 0

    # compute climate effect on capital stock, tfp growth rate, and output
    #note that these are all defined so a positive number means a positive economic impact
    k_climate = 1.0 + k_tas1 * (tas - tas0) + k_tas2 * (tas - tas0)**2 # units of fraction of capital
    k_climate += k_pr1 * (pr - pr0) + k_pr2 * (pr - pr0)**2  
    tfp_climate = 1.0 + tfp_tas1 * (tas - tas0) + tfp_tas2 * (tas - tas0)**2 # units of fraction of TFP
    tfp_climate += tfp_pr1 * (pr - pr0) + tfp_pr2 * (pr - pr0)**2  
    y_climate = 1.0 + y_tas1 * (tas - tas0) + y_tas2 * (tas - tas0)**2  # units of fraction of output
    y_climate += y_pr1 * (pr - pr0) + y_pr2 * (pr - pr0)**2

    
    for t in range(len(y)-1):

        # compute climate responses
        # Note that the climate response is computed at the start of year t, and then applied
        # to the change in capital stock and TFP over year t to year t+1

        # apply climate effect to tfp growth rate
        if t==0:
            a[t] = a0 * tfp_climate[t]
        else:
            a[t] = a[t-1] * tfp_growth[t-1] * tfp_climate[t]
        
        # in year t, we are assume that the damage to capital stock occurs before production occurs
        # so that production in year t is based on the capital stock after climate damage
        # and before investment occurs
        y[t] = a[t] * (k[t]*k_climate[t])**alpha * l[t]**(1-alpha) * y_climate[t]

        # capital stock is then updated based on savings, depereciation, and climate damage
        k[t+1] = (k[t] * k_climate[t]) + s * y[t] - delta * k[t] 

    # compute the last year's TFP and output
    t = len(y)-1
    a[t] = a[t-1] * tfp_growth[t-1] * tfp_climate[t]
    y[t] = a[t] * (k[t]*k_climate[t])**alpha * l[t]**(1-alpha) * y_climate[t]

    return y, a, k, y_climate, tfp_climate, k_climate

def optimize_climate_response_scaling(year, amount, 
        years, tfp, pop, gdp, tas, pr, params: ModelParams, scaling:ScalingParams):
    # This function optimizes the scaling of climate response parameters to minimize to achieve a target difference
    # between climate and weather GDP at a specified year.

    params_copy = copy.copy(params)

    # now we want to find the value of <scale> such that if idx is the index of <year> in <years>,
    # then we want to minimize (() y_climate[idx]/y_weather[idx] ) - (1 + amount) )**2
    # where y_climate and y_weather are the outputs of the calculate_coin_ssp_forward
    def objective_function(scale):
        # scale is a vector of length 4, with the scaling factors for k_tas1, k_tas2, tfp_tas1, tfp_tas2
        params_copy.k_tas1 = params.k_tas1 * (scale * scaling.k_tas1)
        params_copy.k_tas2 = params.k_tas2 * (scale * scaling.k_tas2)
        params_copy.tfp_tas1 = params.tfp_tas1 * (scale * scaling.tfp_tas1)
        params_copy.tfp_tas2 = params.tfp_tas2 * (scale * scaling.tfp_tas2)
        params_copy.y_tas1 = params.y_tas1 * (scale * scaling.y_tas1)
        params_copy.y_tas2 = params.y_tas2 * (scale * scaling.y_tas2)
        params_copy.k_pr1 = params.k_pr1 * (scale * scaling.k_pr1)
        params_copy.k_pr2 = params.k_pr2 * (scale * scaling.k_pr2)
        params_copy.tfp_pr1 = params.tfp_pr1 * (scale * scaling.tfp_pr1)
        params_copy.tfp_pr2 = params.tfp_pr2 * (scale * scaling.tfp_pr2)
        params_copy.y_pr1 = params.y_pr1 * (scale * scaling.y_pr1)
        params_copy.y_pr2 = params.y_pr2 * (scale * scaling.y_pr2)

        y_climate, a_climate, k_climate, _, _, _ = calculate_coin_ssp_forward_model(
            tfp, pop, gdp, tas, pr, params_copy)

        y_weather, a_weather, k_weather, _, _, _ = calculate_coin_ssp_forward_model(
            tfp, pop, gdp, tas*0.0 + params.tas0, pr*0.0 + params.pr0, params)

        idx = np.where(years == year)[0]
        if len(idx) == 0:
            raise ValueError(f"Year {year} not found in years array")
        idx = idx[0]

        ratio = y_climate[idx]/y_weather[idx]
        target = 1.0 + amount

        return (ratio - target)**2
    
    # Perform optimization to find optimal scaling factor
    result = minimize_scalar(objective_function, bounds=(0.0, 10.0), method='bounded')
    
    optimal_scale = result.x
    final_error = result.fun
    
    return optimal_scale, final_error