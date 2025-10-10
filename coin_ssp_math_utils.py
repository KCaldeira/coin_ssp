import numpy as np
import xarray as xr
from skmisc.loess import loess





def _apply_loess_smoothing(time_series, filter_width):
    """
    Core degree-2 LOESS smoothing function.

    Parameters
    ----------
    time_series : array-like
        Annual time series values
    filter_width : int
        Width parameter (approx. number of years to smooth over)

    Returns
    -------
    numpy.ndarray
        LOESS smoothed time series (degree-2 quadratic)
    """
    ts = np.array(time_series, dtype=float)
    n = len(ts)
    t = np.arange(n, dtype=float)

    # Map filter_width (years) to fraction of total series
    frac = min(1.0, filter_width / n)

    # Run LOESS smoothing with degree=2 (quadratic)
    lo = loess(t, ts)
    lo.model.span = frac
    lo.model.degree = 2
    lo.fit()

    smoothed_series = lo.predict(t, stderror=False).values
    return smoothed_series


def apply_loess_subtract(time_series, filter_width, ref_period_slice):
    """
    Apply degree-2 LOESS smoothing and subtract trend, adding back reference period mean.

    This is the standard approach for extracting weather variability components
    from climate variables like temperature and precipitation.

    Parameters
    ----------
    time_series : xr.DataArray
        Annual time series values with time coordinate
    filter_width : int
        Width parameter (approx. number of years to smooth over)
    ref_period_slice : slice
        Reference period as slice(start_year, end_year)

    Returns
    -------
    xr.DataArray
        Time series with trend subtracted and reference period mean added back
    """
    ts = time_series.values

    # Calculate reference period mean using coordinate-based selection
    mean_of_reference_period = time_series.sel(time=ref_period_slice).mean().values

    # Get LOESS smoothed trend
    smoothed_series = _apply_loess_smoothing(ts, filter_width)

    # Subtract trend and add reference period mean
    result_values = ts - smoothed_series + mean_of_reference_period

    # Return as DataArray with same coordinates
    result = xr.DataArray(result_values, coords=time_series.coords, dims=time_series.dims)

    return result


def apply_loess_divide(time_series, filter_width):
    """
    Apply LOESS smoothing and divide original series by smoothed trend.

    This approach is used to get fractional deviations from the long-term trend,
    particularly useful for economic variables where we want to analyze
    proportional changes rather than absolute deviations.

    Parameters
    ----------
    time_series : array-like
        Annual time series values
    filter_width : int
        Width parameter (approx. number of years to smooth over)

    Returns
    -------
    numpy.ndarray
        Time series divided by LOESS smoothed trend (ratio values)
        
        While this conceptually a division, because it is typically applied to GDP which is 
        quasi-exponential, we are actually going to take the difference of the logs, which would
        have similar values in the generatl case, but this way the loess smoothing doesn't need to take into
        account the exponential growth in gdp (because it is quasi-linear growth in log space)
    """
    lnts = np.log(np.array(time_series, dtype=float))

    # Get LOESS smoothed trend
    smoothed_series = _apply_loess_smoothing(lnts, filter_width)

    # Divide original by smoothed trend
    # result = ts / smoothed_series
    # Difference in logs is close to ratio for small changes, but should be better behaved for quasi-exponential data
    result = lnts - smoothed_series

    return result


def calculate_zero_biased_range(data_values, percentile_low=2.5, percentile_high=97.5):
    """
    Calculate visualization range that includes zero when appropriate.

    Logic:
    - If all positive and min > 0.5 * max: extend range to include zero
    - If all negative and max < 0.5 * min: extend range to include zero
    - Always make symmetric around zero for proper colormap centering

    Parameters
    ----------
    data_values : array-like
        Data values to calculate range for (should be finite values only)
    percentile_low : float
        Lower percentile for range calculation (default 2.5 for 95% coverage)
    percentile_high : float
        Upper percentile for range calculation (default 97.5 for 95% coverage)

    Returns
    -------
    tuple
        (vmin, vmax) range values, symmetric around zero
    """
    if len(data_values) == 0:
        return -0.01, 0.01

    # Calculate percentile range
    p_min = np.percentile(data_values, percentile_low)
    p_max = np.percentile(data_values, percentile_high)

    # Apply zero-bias logic
    if p_min > 0 and p_max > 0:  # All positive
        if p_min > 0.5 * p_max:  # Lower end is > 50% of upper end
            range_min = 0.0  # Extend to zero
        else:
            range_min = p_min
        range_max = p_max
    elif p_min < 0 and p_max < 0:  # All negative
        if p_max < 0.5 * p_min:  # Upper end is > 50% of |lower end|
            range_max = 0.0  # Extend to zero
        else:
            range_max = p_max
        range_min = p_min
    else:  # Mixed positive and negative
        range_min = p_min
        range_max = p_max

    # Always include zero if not already included
    range_min = min(range_min, 0.0)
    range_max = max(range_max, 0.0)

    # Make symmetric around zero for proper colormap centering
    abs_max = max(abs(range_min), abs(range_max))

    # Ensure non-zero range
    if abs_max == 0:
        abs_max = 0.01

    return -abs_max, abs_max


def calculate_zero_biased_axis_range(data_values, padding_factor=0.05):
    """
    Calculate axis range for x-y plots that includes zero when appropriate.

    Same zero-bias logic as color scales but without forcing symmetry.
    Adds padding for visual clarity.

    Parameters
    ----------
    data_values : array-like
        Data values to calculate range for (should be finite values only)
    padding_factor : float
        Fraction of range to add as padding (default 0.05 for 5%)

    Returns
    -------
    tuple
        (ymin, ymax) range values with zero-bias logic and padding
    """
    if len(data_values) == 0:
        return -0.1, 0.1

    # Calculate data range
    data_min = np.nanmin(data_values)
    data_max = np.nanmax(data_values)

    # Apply zero-bias logic (same as colormap function)
    if data_min > 0 and data_max > 0:  # All positive
        if data_min > 0.5 * data_max:  # Lower end is > 50% of upper end
            range_min = 0.0  # Extend to zero
        else:
            range_min = data_min
        range_max = data_max
    elif data_min < 0 and data_max < 0:  # All negative
        if data_max < 0.5 * data_min:  # Upper end is > 50% of |lower end|
            range_max = 0.0  # Extend to zero
        else:
            range_max = data_max
        range_min = data_min
    else:  # Mixed positive and negative
        range_min = data_min
        range_max = data_max

    # Add padding for visual clarity
    data_range = range_max - range_min
    if data_range > 0:
        padding = data_range * padding_factor
        range_min -= padding
        range_max += padding

    # Ensure non-zero range
    if abs(range_max - range_min) < 1e-10:
        center = (range_min + range_max) / 2
        range_min = center - 0.1
        range_max = center + 0.1

    return range_min, range_max

def calculate_area_weights(lat):
    """
    Calculate area weights proportional to cosine of latitude.

    Parameters
    ----------
    lat : array
        Latitude coordinates in degrees

    Returns
    -------
    array
        Area weights with same shape as lat, normalized to sum to 1
    """
    weights = np.cos(np.radians(lat))
    return weights / weights.sum()


def calculate_time_means(data, start_year, end_year):
    """
    Calculate temporal mean over specified year range.

    Parameters
    ----------
    data : xr.DataArray
        Data array with time coordinate
    start_year : int
        Start year (inclusive)
    end_year : int
        End year (inclusive)

    Returns
    -------
    xr.DataArray
        Time-averaged data with time dimension removed
    """
    return data.sel(time=slice(start_year, end_year)).mean(dim='time')


def calculate_global_mean(data, valid_mask):
    """
    Calculate area-weighted global mean using only valid grid cells.

    Parameters
    ----------
    data : xr.DataArray
        Spatial data with lat, lon coordinates
    valid_mask : xr.DataArray
        Boolean mask (lat, lon) indicating valid grid cells

    Returns
    -------
    float
        Area-weighted global mean over valid cells
    """
    # Calculate area weights from latitude coordinate
    weights = np.cos(np.deg2rad(data.lat))
    weights = weights / weights.sum()

    # Apply mask and weights
    masked_data = data.where(valid_mask)

    # Use xarray's weighted operations
    weighted_mean = masked_data.weighted(weights).mean(dim=['lat', 'lon'])

    return float(weighted_mean.values)


def calculate_gdp_weighted_mean(variable_series, gdp_series, area_weights, valid_mask, start_year, end_year):
    """
    Calculate GDP-weighted mean of a variable over a specified time period.

    Correctly computes mean over time of [sum(area×GDP×variable) / sum(area×GDP)] for each year,
    accounting for grid cell area to get proper weighting by total GDP rather than GDP density.

    Parameters
    ----------
    variable_series : xr.DataArray
        Variable time series with time, lat, lon coordinates
    gdp_series : xr.DataArray
        GDP density time series with time, lat, lon coordinates (GDP per unit area)
    area_weights : xr.DataArray
        Grid cell area weights [lat, lon]
    valid_mask : xr.DataArray
        Boolean mask for valid economic grid cells [lat, lon]
    start_year : int
        Start year of period (inclusive)
    end_year : int
        End year of period (inclusive)

    Returns
    -------
    float
        GDP-weighted mean of variable over the specified period
    """
    # Select time period using coordinate-based slicing
    var_period = variable_series.sel(time=slice(start_year, end_year))
    gdp_period = gdp_series.sel(time=slice(start_year, end_year))

    # Apply valid mask (broadcasts automatically across time)
    var_masked = var_period.where(valid_mask)
    gdp_masked = gdp_period.where(valid_mask)

    # Multiply GDP by area to get total GDP (not density)
    # area_weights [lat, lon] broadcasts across time dimension
    gdp_total = gdp_masked * area_weights

    # Calculate GDP-weighted mean over entire space-time period
    # This is sum(GDP×area×variable) / sum(GDP×area) over all time and space
    gdp_weighted_mean = (
        (gdp_total * var_masked).sum(dim=['time', 'lat', 'lon']) /
        gdp_total.sum(dim=['time', 'lat', 'lon'])
    )

    return float(gdp_weighted_mean.values)