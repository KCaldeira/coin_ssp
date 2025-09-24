import copy
import json
import numpy as np
import statsmodels.api as sm
import xarray as xr
from typing import Dict, Any





def apply_time_series_filter(time_series, filter_width, ref_start_idx, ref_end_idx):
    """
    Apply LOESS filter to time series and remove trend relative to reference period mean.

    Parameters
    ----------
    time_series : array-like
        Annual time series values (first element corresponds to year 0)
    filter_width : int
        Width parameter (approx. number of years to smooth over)
    ref_start_idx : int
        Start index of reference period (0-indexed)
    ref_end_idx : int
        End index of reference period (0-indexed, inclusive)

    Returns
    -------
    numpy.ndarray
        Filtered time series with trend removed and reference period mean added back
    """
    ts = np.array(time_series, dtype=float)
    n = len(ts)
    t = np.arange(n, dtype=float)

    # Calculate reference period mean
    mean_of_reference_period = np.mean(ts[ref_start_idx:ref_end_idx+1])

    # LOWESS expects frac = proportion of data used in each local regression
    # Map filter_width (years) to fraction of total series
    frac = min(1.0, filter_width / n)

    # Run LOESS smoothing
    filtered_series = sm.nonparametric.lowess(ts, t, frac=frac,
                                              it=1, return_sorted=False)

    # Apply detrending to all years: remove trend and add reference period mean
    result = ts - filtered_series + mean_of_reference_period

    return result


def _apply_loess_smoothing(time_series, filter_width):
    """
    Core LOESS smoothing function.

    Parameters
    ----------
    time_series : array-like
        Annual time series values
    filter_width : int
        Width parameter (approx. number of years to smooth over)

    Returns
    -------
    numpy.ndarray
        LOESS smoothed time series
    """
    ts = np.array(time_series, dtype=float)
    n = len(ts)
    t = np.arange(n, dtype=float)

    # Map filter_width (years) to fraction of total series
    frac = min(1.0, filter_width / n)

    # Run LOESS smoothing
    smoothed_series = sm.nonparametric.lowess(ts, t, frac=frac,
                                              it=1, return_sorted=False)
    return smoothed_series


def apply_loess_subtract(time_series, filter_width, ref_start_idx, ref_end_idx):
    """
    Apply LOESS smoothing and subtract trend, adding back reference period mean.

    This is the standard approach for extracting weather variability components
    from climate variables like temperature and precipitation.

    Parameters
    ----------
    time_series : array-like
        Annual time series values
    filter_width : int
        Width parameter (approx. number of years to smooth over)
    ref_start_idx : int
        Start index of reference period (0-indexed)
    ref_end_idx : int
        End index of reference period (0-indexed, inclusive)

    Returns
    -------
    numpy.ndarray
        Time series with trend subtracted and reference period mean added back
    """
    ts = np.array(time_series, dtype=float)

    # Calculate reference period mean
    mean_of_reference_period = np.mean(ts[ref_start_idx:ref_end_idx+1])

    # Get LOESS smoothed trend
    smoothed_series = _apply_loess_smoothing(ts, filter_width)

    # Subtract trend and add reference period mean
    result = ts - smoothed_series + mean_of_reference_period

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
    """
    ts = np.array(time_series, dtype=float)

    # Get LOESS smoothed trend
    smoothed_series = _apply_loess_smoothing(ts, filter_width)

    # Divide original by smoothed trend
    result = ts / smoothed_series

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


def calculate_time_means(data, years, start_year, end_year):
    """
    Calculate temporal mean over specified year range.

    Parameters
    ----------
    data : array
        Data array with time as first dimension
    years : array
        Year values corresponding to time dimension
    start_year : int
        Start year (inclusive)
    end_year : int
        End year (inclusive)

    Returns
    -------
    array
        Time-averaged data with time dimension removed
    """
    mask = (years >= start_year) & (years <= end_year)
    return data[mask].mean(axis=0)


def calculate_global_mean(data, lat, valid_mask):
    """
    Calculate area-weighted global mean using only valid grid cells.

    Parameters
    ----------
    data : array
        2D spatial data (lat, lon)
    lat : array
        Latitude coordinates
    valid_mask : array
        2D boolean mask (lat, lon) indicating valid grid cells

    Returns
    -------
    float
        Area-weighted global mean over valid cells
    """
    weights = calculate_area_weights(lat)
    # Expand weights to match data shape: (lat,) -> (lat, lon)
    weights_2d = np.broadcast_to(weights[:, np.newaxis], data.shape)

    # Apply mask to both data and weights
    masked_data = np.where(valid_mask, data, np.nan)
    masked_weights = np.where(valid_mask, weights_2d, 0.0)

    # Use np.nansum and np.sum to handle NaN values properly
    return np.nansum(masked_data * masked_weights) / np.sum(masked_weights)