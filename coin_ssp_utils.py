import numpy as np
from scipy.signal import savgol_filter

def apply_time_series_filter(time_series, filter_width, start_year):
    """
    Apply Savitzky-Golay filter to time series with temporal constraints.
    
    Parameters
    ----------
    time_series : array-like
        Annual time series values (first element corresponds to year 0)
    filter_width : int
        Width of the Savitzky-Golay filter window (must be odd)
    start_year : int
        Year at which filtering begins (0-indexed)
        
    Returns
    -------
    numpy.ndarray
        Filtered time series with original values for t <= start_year,
        and adjusted values for t > start_year
    """
    ts = np.array(time_series)
    
    # Ensure filter_width is odd
    if filter_width % 2 == 0:
        filter_width += 1
    
    # Apply Savitzky-Golay filter to entire series
    # Using polynomial order 3 for smooth trends typical in economic data
    filtered_series = savgol_filter(ts, filter_width, 3)
    
    # Create result array starting with original series
    result = np.copy(ts)
    
    # For years after start_year, apply the adjustment:
    # result = original - filtered + filtered_at_start_year
    if start_year < len(ts):
        filtered_at_start = filtered_series[start_year]
        for t in range(start_year + 1, len(ts)):
            result[t] = ts[t] - filtered_series[t] + filtered_at_start
    
    return result