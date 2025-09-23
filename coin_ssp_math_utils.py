import copy
import json
import numpy as np
import statsmodels.api as sm
import xarray as xr
from typing import Dict, Any


def get_adaptive_subplot_layout(n_targets):
    """
    Calculate optimal subplot layout based on number of targets.

    For 3 or fewer targets: single column layout
    For 4+ targets: two-column layout to maintain reasonable aspect ratios

    Parameters
    ----------
    n_targets : int
        Number of targets/plots per page

    Returns
    -------
    tuple
        (rows, cols, figsize) for matplotlib subplot layout
    """
    if n_targets <= 3:
        # Single column layout for 3 or fewer
        return (n_targets, 1, (12, 16))
    else:
        # Two column layout for 4+
        rows = (n_targets + 1) // 2  # Ceiling division
        cols = 2
        height = 4 * rows + 4  # Scale height with number of rows
        return (rows, cols, (16, height))


def create_serializable_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a JSON-serializable version of the configuration by filtering out non-serializable objects.

    Parameters
    ----------
    config : Dict[str, Any]
        Original configuration dictionary

    Returns
    -------
    Dict[str, Any]
        Filtered configuration dictionary safe for JSON serialization
    """

    # Make a deep copy to avoid modifying the original
    filtered_config = copy.deepcopy(config)

    # Remove known non-serializable objects
    non_serializable_keys = ['model_params_factory']

    def remove_non_serializable(obj, path=""):
        if isinstance(obj, dict):
            keys_to_remove = []
            for key, value in obj.items():
                current_path = f"{path}.{key}" if path else key
                if key in non_serializable_keys:
                    keys_to_remove.append(key)
                else:
                    try:
                        # Test if the value is JSON serializable
                        json.dumps(value)
                    except (TypeError, ValueError):
                        # If not serializable, try to recurse or remove
                        if isinstance(value, dict):
                            remove_non_serializable(value, current_path)
                        elif isinstance(value, list):
                            remove_non_serializable(value, current_path)
                        else:
                            keys_to_remove.append(key)

            for key in keys_to_remove:
                obj.pop(key)

        elif isinstance(obj, list):
            items_to_remove = []
            for i, item in enumerate(obj):
                try:
                    json.dumps(item)
                except (TypeError, ValueError):
                    if isinstance(item, (dict, list)):
                        remove_non_serializable(item, f"{path}[{i}]")
                    else:
                        items_to_remove.append(i)

            for i in reversed(items_to_remove):
                obj.pop(i)

    remove_non_serializable(filtered_config)
    return filtered_config


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


def extract_year_coordinate(dataset, coord_names=None):
    """
    Extract actual year values from NetCDF time coordinate.

    Parameters
    ----------
    dataset : xarray.Dataset
        NetCDF dataset with time coordinate
    coord_names : list, optional
        Coordinate names to try, by default ['year', 'time', 'axis_0']

    Returns
    -------
    tuple
        (years, valid_mask) where years are the extracted year values
        and valid_mask indicates which time indices are valid
    """
    if coord_names is None:
        coord_names = ['year', 'time', 'axis_0']

    time_coord = None
    for coord_name in coord_names:
        if coord_name in dataset.coords:
            time_coord = dataset.coords[coord_name]
            break

    if time_coord is None:
        raise ValueError(f"No time coordinate found. Tried: {coord_names}")

    # Handle different time coordinate types
    time_values = time_coord.values

    # If already integers (years), return as-is
    if np.issubdtype(time_values.dtype, np.integer):
        return time_values.astype(int), np.ones(len(time_values), dtype=bool)

    # If datetime-like objects, extract year
    if hasattr(time_values[0], 'year'):
        years = np.array([t.year for t in time_values])
        return years, np.ones(len(years), dtype=bool)

    # If floating point values, treat as raw years (ignore "years since" units)
    if np.issubdtype(time_values.dtype, np.floating):
        # Filter for reasonable year values and ignore units encoding
        valid_mask = np.isfinite(time_values) & (time_values > 1800) & (time_values < 2200)
        if not np.any(valid_mask):
            raise ValueError("No valid years found in time coordinate")

        valid_years = time_values[valid_mask].astype(int)
        if np.sum(valid_mask) != len(time_values):
            print(f"    Warning: Filtered out {len(time_values) - np.sum(valid_mask)} time values with incomplete data")

        return valid_years, valid_mask

    raise ValueError(f"Cannot extract years from time coordinate with dtype {time_values.dtype}")


def interpolate_to_annual_grid(original_years, data_array, target_years):
    """
    Linearly interpolate data to annual time grid.

    Parameters
    ----------
    original_years : numpy.ndarray
        Original year values (1D array)
    data_array : numpy.ndarray
        Data array with time as first dimension: (time, ...)
    target_years : numpy.ndarray
        Target annual year values (1D array)

    Returns
    -------
    numpy.ndarray
        Interpolated data array with shape (len(target_years), ...)
    """
    # If already on target grid, return as-is
    if len(original_years) == len(target_years) and np.allclose(original_years, target_years):
        return data_array

    # Get spatial dimensions
    spatial_shape = data_array.shape[1:]

    # Flatten spatial dimensions for vectorized interpolation
    data_flat = data_array.reshape(len(original_years), -1)

    # Interpolate each spatial point
    interpolated_flat = np.zeros((len(target_years), data_flat.shape[1]))

    for i in range(data_flat.shape[1]):
        interpolated_flat[:, i] = np.interp(target_years, original_years, data_flat[:, i])

    # Reshape back to original spatial dimensions
    return interpolated_flat.reshape((len(target_years),) + spatial_shape)


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