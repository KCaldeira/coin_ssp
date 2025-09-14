import os
import numpy as np
import statsmodels.api as sm
import copy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
import xarray as xr
from datetime import datetime
from typing import Dict, Any
from coin_ssp_core import ScalingParams

def apply_time_series_filter(time_series, filter_width, start_year):
    """
    Apply LOESS filter to time series with temporal constraints.

    Parameters
    ----------
    time_series : array-like
        Annual time series values (first element corresponds to year 0)
    filter_width : int
        Width parameter (approx. number of years to smooth over)
    start_year : int
        Year at which filtering begins (0-indexed)

    Returns
    -------
    numpy.ndarray
        Filtered time series with original values for t <= start_year,
        and adjusted values for t > start_year
    """
    ts = np.array(time_series, dtype=float)
    n = len(ts)
    t = np.arange(n, dtype=float)

    # LOWESS expects frac = proportion of data used in each local regression
    # Map filter_width (years) to fraction of total series
    frac = min(1.0, filter_width / n)

    # Run LOESS smoothing
    filtered_series = sm.nonparametric.lowess(ts, t, frac=frac,
                                              it=1, return_sorted=False)

    # Create result array starting with original series
    result = np.copy(ts)

    # Apply the same adjustment logic you used for Savitzky–Golay
    if start_year < n:
        filtered_at_start = filtered_series[start_year]
        for ti in range(start_year + 1, n):
            result[ti] = ts[ti] - filtered_series[ti] + filtered_at_start

    return result

def create_country_scaling_page(country, scaling_name, results, scaling_result, params, fig):
    """Create a single page with three panels for one country and scaling set."""
    fig.suptitle(f'{country} - {scaling_name}', fontsize=16, fontweight='bold')
    
    years = results['years']
    
    # Panel 1: GDP
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(years, results['gdp_observed'], 'k-', label='Baseline', linewidth=2)
    ax1.plot(years, scaling_result['gdp_climate'], 'r-', label='Climate', linewidth=1.5)
    ax1.plot(years, scaling_result['gdp_weather'], 'b--', label='Weather', linewidth=1.5)
    ax1.set_ylabel('GDP (billion $)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Add scaling info box in lower right corner with all climate parameters
    ps = scaling_result["params_scaled"]
    scaling_text = (f'Scaling: {scaling_name}\n'
                   f'Scale factor: {scaling_result["optimal_scale"]:.4f}\n'
                   f'Target: {params.amount_scale:.1%} by {params.year_scale}\n'
                   f'k_tas1: {ps.k_tas1:.6f}  k_tas2: {ps.k_tas2:.6f}\n'
                   f'tfp_tas1: {ps.tfp_tas1:.6f}  tfp_tas2: {ps.tfp_tas2:.6f}\n'
                   f'y_tas1: {ps.y_tas1:.6f}  y_tas2: {ps.y_tas2:.6f}\n'
                   f'k_pr1: {ps.k_pr1:.6f}  k_pr2: {ps.k_pr2:.6f}\n'
                   f'tfp_pr1: {ps.tfp_pr1:.6f}  tfp_pr2: {ps.tfp_pr2:.6f}\n'
                   f'y_pr1: {ps.y_pr1:.6f}  y_pr2: {ps.y_pr2:.6f}')
    
    ax1.text(0.98, 0.02, scaling_text, transform=ax1.transAxes, fontsize=8,
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Panel 2: TFP
    ax2 = fig.add_subplot(3, 1, 2)
    ax2.plot(years, results['tfp_baseline'], 'k-', label='Baseline', linewidth=2)
    ax2.plot(years, scaling_result['tfp_climate'], 'r-', label='Climate', linewidth=1.5)
    ax2.plot(years, scaling_result['tfp_weather'], 'b--', label='Weather', linewidth=1.5)
    ax2.set_ylabel('Total Factor Productivity')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Capital Stock
    ax3 = fig.add_subplot(3, 1, 3)
    ax3.plot(years, results['k_baseline'], 'k-', label='Baseline', linewidth=2)
    ax3.plot(years, scaling_result['k_climate'], 'r-', label='Climate', linewidth=1.5)
    ax3.plot(years, scaling_result['k_weather'], 'b--', label='Weather', linewidth=1.5)
    ax3.set_ylabel('Capital Stock (normalized)')
    ax3.set_xlabel('Year')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def filter_scaling_params(scaling_config):
    allowed_keys = set(ScalingParams.__dataclass_fields__.keys())
    return {k: v for k, v in scaling_config.items() if k in allowed_keys}

def create_country_pdf_books(all_results, params, output_dir, run_name, timestamp):
    """Create PDF books with one book per country, one page per scaling set."""
    output_dir = Path(output_dir)
    
    print(f"\nCreating PDF books for {len(all_results)} countries...")
    
    pdf_files = []
    for i, (country, results) in enumerate(sorted(all_results.items()), 1):
        print(f"  [{i}/{len(all_results)}] Creating book for {country}...")
        
        pdf_file = output_dir / f"COIN_SSP_{country.replace(' ', '_')}_Book_{run_name}_{timestamp}.pdf"
        
        with PdfPages(pdf_file) as pdf:
            for j, (scaling_name, scaling_result) in enumerate(results['scaling_results'].items(), 1):
                print(f"    Page {j}: {scaling_name}")
                
                # Create figure for this scaling set
                fig = plt.figure(figsize=(8.5, 11))  # Letter size portrait
                create_country_scaling_page(country, scaling_name, results, scaling_result, params, fig)
                
                # Save to PDF
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
        
        pdf_files.append(pdf_file)
        print(f"    Saved: {pdf_file}")
    
    print(f"  Created {len(pdf_files)} PDF books")
    return pdf_files

# =============================================================================
# Temporal Alignment Utilities
# =============================================================================

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
            print(f"    Warning: Filtered out {len(time_values) - np.sum(valid_mask)} invalid time values")

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

# =============================================================================
# Target GDP Utilities (moved from coin_ssp_target_gdp.py)
# =============================================================================

def load_gridded_data(model_name, case_name):
    """
    Load all four NetCDF files and return as a temporally-aligned dataset.

    All variables are interpolated to annual resolution and aligned to the
    same common year range that all variables share after interpolation.

    Parameters
    ----------
    model_name : str
        Climate model name
    case_name : str
        SSP scenario name

    Returns
    -------
    dict
        Dictionary containing temporally-aligned data:
        - 'tas': temperature data (time, lat, lon) - annual, common years
        - 'pr': precipitation data (time, lat, lon) - annual, common years
        - 'gdp': GDP data (time, lat, lon) - annual, common years (interpolated)
        - 'pop': population data (time, lat, lon) - annual, common years
        - 'lat': latitude coordinates
        - 'lon': longitude coordinates
        - 'tas_years': temperature time axis (annual years)
        - 'pr_years': precipitation time axis (annual years)
        - 'gdp_years': GDP time axis (annual years, interpolated)
        - 'pop_years': population time axis (annual years)
        - 'common_years': final common year range for all variables
    """
    print(f"Loading and aligning NetCDF data for {model_name} {case_name}...")

    # Load temperature data
    print("  Loading temperature data...")
    tas_ds = xr.open_dataset(f'data/input/gridRaw_tas_{model_name}_{case_name}.nc', decode_times=False)
    tas_raw_all = tas_ds.tas.values - 273.15  # Convert from Kelvin to Celsius
    tas_years_raw, tas_valid_mask = extract_year_coordinate(tas_ds)
    tas_raw = tas_raw_all[tas_valid_mask]

    # Load precipitation data
    print("  Loading precipitation data...")
    pr_ds = xr.open_dataset(f'data/input/gridRaw_pr_{model_name}_{case_name}.nc', decode_times=False)
    pr_raw_all = pr_ds.pr.values
    pr_years_raw, pr_valid_mask = extract_year_coordinate(pr_ds)
    pr_raw = pr_raw_all[pr_valid_mask]

    # Load GDP data
    print("  Loading GDP data...")
    gdp_ds = xr.open_dataset(f'data/input/gridded_gdp_regrid_{model_name}_{case_name}.nc', decode_times=False)
    gdp_raw_all = gdp_ds.gdp_20202100ssp5.values
    gdp_years_raw, gdp_valid_mask = extract_year_coordinate(gdp_ds)
    gdp_raw = gdp_raw_all[gdp_valid_mask]

    # Load population data
    print("  Loading population data...")
    pop_ds = xr.open_dataset(f'data/input/gridded_pop_regrid_{model_name}_{case_name}.nc', decode_times=False)
    pop_raw_all = pop_ds.pop_20062100ssp5.values
    pop_years_raw, pop_valid_mask = extract_year_coordinate(pop_ds)
    pop_raw = pop_raw_all[pop_valid_mask]

    # Get coordinate arrays
    lat = tas_ds.lat.values
    lon = tas_ds.lon.values

    print(f"  Original time ranges:")
    print(f"    Temperature: {tas_years_raw.min()}-{tas_years_raw.max()} ({len(tas_years_raw)} points)")
    print(f"    Precipitation: {pr_years_raw.min()}-{pr_years_raw.max()} ({len(pr_years_raw)} points)")
    print(f"    GDP: {gdp_years_raw.min()}-{gdp_years_raw.max()} ({len(gdp_years_raw)} points)")
    print(f"    Population: {pop_years_raw.min()}-{pop_years_raw.max()} ({len(pop_years_raw)} points)")

    # Create annual grids for each variable
    print("  Interpolating to annual grids...")
    tas_years_annual = np.arange(tas_years_raw.min(), tas_years_raw.max() + 1)
    pr_years_annual = np.arange(pr_years_raw.min(), pr_years_raw.max() + 1)
    gdp_years_annual = np.arange(gdp_years_raw.min(), gdp_years_raw.max() + 1)
    pop_years_annual = np.arange(pop_years_raw.min(), pop_years_raw.max() + 1)

    # Interpolate each variable to annual grid
    tas_annual = interpolate_to_annual_grid(tas_years_raw, tas_raw, tas_years_annual)
    pr_annual = interpolate_to_annual_grid(pr_years_raw, pr_raw, pr_years_annual)
    gdp_annual = interpolate_to_annual_grid(gdp_years_raw, gdp_raw, gdp_years_annual)
    pop_annual = interpolate_to_annual_grid(pop_years_raw, pop_raw, pop_years_annual)

    # Find common year range (intersection of all ranges)
    common_start = max(tas_years_annual.min(), pr_years_annual.min(),
                      gdp_years_annual.min(), pop_years_annual.min())
    common_end = min(tas_years_annual.max(), pr_years_annual.max(),
                    gdp_years_annual.max(), pop_years_annual.max())
    common_years = np.arange(common_start, common_end + 1)

    print(f"  Common year range: {common_start}-{common_end} ({len(common_years)} years)")

    # Subset all variables to common years
    def subset_to_common_years(data, years, common_years):
        start_idx = np.where(years == common_years[0])[0][0]
        end_idx = np.where(years == common_years[-1])[0][0] + 1
        return data[start_idx:end_idx], years[start_idx:end_idx]

    tas_aligned, tas_years_aligned = subset_to_common_years(tas_annual, tas_years_annual, common_years)
    pr_aligned, pr_years_aligned = subset_to_common_years(pr_annual, pr_years_annual, common_years)
    gdp_aligned, gdp_years_aligned = subset_to_common_years(gdp_annual, gdp_years_annual, common_years)
    pop_aligned, pop_years_aligned = subset_to_common_years(pop_annual, pop_years_annual, common_years)

    # Verify alignment
    assert np.array_equal(tas_years_aligned, common_years), "Temperature years not aligned"
    assert np.array_equal(pr_years_aligned, common_years), "Precipitation years not aligned"
    assert np.array_equal(gdp_years_aligned, common_years), "GDP years not aligned"
    assert np.array_equal(pop_years_aligned, common_years), "Population years not aligned"

    print(f"  ✅ All variables aligned to {len(common_years)} common years")

    # Close datasets
    tas_ds.close()
    pr_ds.close()
    gdp_ds.close()
    pop_ds.close()

    return {
        'tas': tas_aligned,
        'pr': pr_aligned,
        'gdp': gdp_aligned,
        'pop': pop_aligned,
        'lat': lat,
        'lon': lon,
        'tas_years': tas_years_aligned,
        'pr_years': pr_years_aligned,
        'gdp_years': gdp_years_aligned,
        'pop_years': pop_years_aligned,
        'common_years': common_years
    }

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

def calculate_global_mean(data, lat):
    """
    Calculate area-weighted global mean.
    
    Parameters
    ----------
    data : array
        2D spatial data (lat, lon) 
    lat : array
        Latitude coordinates
        
    Returns
    -------
    float
        Area-weighted global mean
    """
    weights = calculate_area_weights(lat)
    # Expand weights to match data shape: (lat,) -> (lat, lon)  
    weights_2d = np.broadcast_to(weights[:, np.newaxis], data.shape)
    return np.average(data, weights=weights_2d)


# =============================================================================
# Target GDP Reduction Calculation Functions
# Extracted from calculate_target_gdp_reductions.py for reuse in integrated workflow
# =============================================================================

def calculate_constant_target_reduction(gdp_reduction_value, temp_ref_shape):
    """
    Calculate constant GDP reduction across all grid cells.
    
    Parameters
    ----------
    gdp_reduction_value : float
        Constant reduction value (e.g., -0.10 for 10% reduction)
    temp_ref_shape : tuple
        Shape of temperature reference array for output sizing
        
    Returns
    -------
    np.ndarray
        Constant reduction array with shape temp_ref_shape
    """
    return np.full(temp_ref_shape, gdp_reduction_value, dtype=np.float64)


def calculate_linear_target_reduction(linear_config, temp_ref, gdp_target, lat):
    """
    Calculate linear temperature-dependent GDP reduction using constraint satisfaction.
    
    Implements the mathematical framework:
    reduction(T) = a0 + a1 * T
    
    Subject to two constraints:
    1. Point constraint: reduction(T_ref) = value_at_ref
    2. GDP-weighted global mean: ∑[w_i * gdp_i * (1 + reduction(T_i))] / ∑[w_i * gdp_i] = target_mean
    
    Parameters
    ----------
    linear_config : dict
        Configuration containing:
        - 'global_mean_reduction': Target GDP-weighted global mean (e.g., -0.10)
        - 'reference_temperature': Reference temperature point (e.g., 30.0)
        - 'reduction_at_reference_temp': Reduction at reference temperature (e.g., -0.25)
    temp_ref : np.ndarray
        Reference period temperature array [lat, lon]
    gdp_target : np.ndarray
        Target period GDP array [lat, lon]
    lat : np.ndarray
        Latitude coordinate array for area weighting
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'reduction_array': Linear reduction array [lat, lon]
        - 'coefficients': {'a0': intercept, 'a1': slope}
        - 'constraint_verification': Verification of constraint satisfaction
    """
    # Extract configuration parameters
    global_mean_linear = linear_config['global_mean_reduction']
    T_ref_linear = linear_config['reference_temperature'] 
    value_at_ref_linear = linear_config['reduction_at_reference_temp']
    
    # Calculate GDP-weighted global means
    global_gdp_target = calculate_global_mean(gdp_target, lat)
    gdp_weighted_temp_mean = np.float64(calculate_global_mean(gdp_target * temp_ref, lat) / global_gdp_target)
    
    # Set up weighted least squares system for exact constraint satisfaction
    X = np.array([
        [1.0, T_ref_linear],                    # Point constraint equation
        [1.0, gdp_weighted_temp_mean]           # GDP-weighted global mean equation
    ], dtype=np.float64)
    
    y = np.array([
        value_at_ref_linear,                    # Target at reference temperature
        global_mean_linear                      # Target GDP-weighted global mean  
    ], dtype=np.float64)
    
    # Solve for coefficients: [a0, a1]
    coefficients = np.linalg.solve(X, y)
    a0_linear, a1_linear = coefficients
    
    # Calculate linear reduction array
    linear_reduction = a0_linear + a1_linear * temp_ref
    
    # Verify constraint satisfaction
    constraint1_check = a0_linear + a1_linear * T_ref_linear  # Should equal value_at_ref_linear
    constraint2_check = calculate_global_mean(gdp_target * (1 + linear_reduction), lat) / global_gdp_target - 1  # Should equal global_mean_linear
    
    return {
        'reduction_array': linear_reduction.astype(np.float64),
        'coefficients': {'a0': float(a0_linear), 'a1': float(a1_linear)},
        'constraint_verification': {
            'point_constraint': {
                'achieved': float(constraint1_check),
                'target': float(value_at_ref_linear),
                'error': float(abs(constraint1_check - value_at_ref_linear))
            },
            'global_mean_constraint': {
                'achieved': float(constraint2_check),
                'target': float(global_mean_linear),
                'error': float(abs(constraint2_check - global_mean_linear))
            }
        },
        'gdp_weighted_temp_mean': float(gdp_weighted_temp_mean)
    }


def calculate_quadratic_target_reduction(quadratic_config, temp_ref, gdp_target, lat):
    """
    Calculate quadratic temperature-dependent GDP reduction using constraint satisfaction.
    
    Implements the mathematical framework:
    reduction(T) = a + b * (T - T0) + c * (T - T0)²
    
    Subject to three constraints:
    1. Zero point: reduction(T0) = 0
    2. Reference point: reduction(T_ref) = value_at_ref  
    3. GDP-weighted global mean: ∑[w_i * gdp_i * (1 + reduction(T_i))] / ∑[w_i * gdp_i] = target_mean
    
    Parameters
    ----------
    quadratic_config : dict
        Configuration containing:
        - 'global_mean_reduction': Target GDP-weighted global mean (e.g., -0.10)
        - 'reference_temperature': Reference temperature point (e.g., 30.0)
        - 'reduction_at_reference_temp': Reduction at reference temperature (e.g., -0.75)
        - 'zero_reduction_temperature': Temperature with zero reduction (e.g., 13.5)
        - 'max_reduction': Optional maximum reduction bound (e.g., -0.80)
    temp_ref : np.ndarray
        Reference period temperature array [lat, lon]
    gdp_target : np.ndarray
        Target period GDP array [lat, lon]
    lat : np.ndarray
        Latitude coordinate array for area weighting
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'reduction_array': Quadratic reduction array [lat, lon], optionally bounded
        - 'coefficients': {'a': constant, 'b': linear, 'c': quadratic}
        - 'constraint_verification': Verification of constraint satisfaction
        - 'bounds_applied': Whether maximum reduction bounds were applied
    """
    # Extract configuration parameters
    global_mean_quad = quadratic_config['global_mean_reduction']
    T_ref_quad = quadratic_config['reference_temperature']
    value_at_ref_quad = quadratic_config['reduction_at_reference_temp']
    T0 = quadratic_config['zero_reduction_temperature']
    max_reduction = quadratic_config.get('max_reduction', None)
    
    # Calculate GDP-weighted global means
    global_gdp_target = calculate_global_mean(gdp_target, lat)
    gdp_weighted_temp_mean = np.float64(calculate_global_mean(gdp_target * temp_ref, lat) / global_gdp_target)
    gdp_weighted_temp2_mean = np.float64(calculate_global_mean(gdp_target * temp_ref**2, lat) / global_gdp_target)
    
    # Set up constraint system for quadratic: a + b*(T-T0) + c*(T-T0)²
    # Constraint 1: a + b*(T0-T0) + c*(T0-T0)² = 0  =>  a = 0
    # Constraint 2: a + b*(T_ref-T0) + c*(T_ref-T0)² = value_at_ref_quad
    # Constraint 3: GDP-weighted global mean constraint
    
    a_quad = 0.0  # From zero point constraint
    
    # Solve for b and c using remaining constraints
    X = np.array([
        [T_ref_quad - T0, (T_ref_quad - T0)**2],                                    # Reference point constraint
        [gdp_weighted_temp_mean - T0, gdp_weighted_temp2_mean - T0**2]              # GDP-weighted constraint
    ], dtype=np.float64)
    
    y = np.array([
        value_at_ref_quad,          # Target at reference temperature
        global_mean_quad            # Target GDP-weighted global mean
    ], dtype=np.float64)
    
    # Solve for coefficients: [b, c]
    bc_coefficients = np.linalg.solve(X, y)
    b_quad, c_quad = bc_coefficients
    
    # Calculate quadratic reduction array
    temp_shifted = temp_ref - T0
    quadratic_reduction = a_quad + b_quad * temp_shifted + c_quad * temp_shifted**2
    
    # Apply bounds if specified
    bounds_applied = False
    if max_reduction is not None:
        bounded_reduction = np.maximum(quadratic_reduction, max_reduction)
        if not np.array_equal(bounded_reduction, quadratic_reduction):
            bounds_applied = True
        quadratic_reduction = bounded_reduction
    
    # Verify constraint satisfaction (using unbounded values for mathematical verification)
    temp_shifted_ref = T_ref_quad - T0
    constraint1_check = a_quad  # Should be 0
    constraint2_check = a_quad + b_quad * temp_shifted_ref + c_quad * temp_shifted_ref**2  # Should equal value_at_ref_quad
    
    # Global mean constraint (use bounded values if applied)
    constraint3_check = calculate_global_mean(gdp_target * (1 + quadratic_reduction), lat) / global_gdp_target - 1
    
    return {
        'reduction_array': quadratic_reduction.astype(np.float64),
        'coefficients': {'a': float(a_quad), 'b': float(b_quad), 'c': float(c_quad)},
        'constraint_verification': {
            'zero_point_constraint': {
                'achieved': float(constraint1_check),
                'target': 0.0,
                'error': float(abs(constraint1_check))
            },
            'reference_point_constraint': {
                'achieved': float(constraint2_check),
                'target': float(value_at_ref_quad),
                'error': float(abs(constraint2_check - value_at_ref_quad))
            },
            'global_mean_constraint': {
                'achieved': float(constraint3_check),
                'target': float(global_mean_quad),
                'error': float(abs(constraint3_check - global_mean_quad))
            }
        },
        'gdp_weighted_temp_mean': float(gdp_weighted_temp_mean),
        'gdp_weighted_temp2_mean': float(gdp_weighted_temp2_mean),
        'bounds_applied': bounds_applied,
        'max_reduction_bound': max_reduction
    }


def calculate_all_target_reductions(target_configs, gridded_data):
    """
    Calculate all configured target GDP reductions using gridded data.
    
    This function processes multiple target configurations and returns
    results for all target types (constant, linear, quadratic).
    
    Parameters
    ----------
    target_configs : list
        List of target configuration dictionaries, each containing:
        - 'target_name': Unique identifier
        - 'target_type': 'constant', 'linear', or 'quadratic'  
        - Type-specific parameters
    gridded_data : dict
        Dictionary containing gridded data arrays:
        - 'temp_ref': Reference period temperature [lat, lon]
        - 'gdp_target': Target period GDP [lat, lon]
        - 'lat': Latitude coordinates
        
    Returns
    -------
    dict
        Dictionary with target_name keys, each containing:
        - 'target_type': Type of target function
        - 'reduction_array': Calculated reduction array [lat, lon]
        - 'coefficients': Function coefficients (if applicable)
        - 'constraint_verification': Constraint satisfaction results
        - 'global_statistics': Global mean calculations
    """
    results = {}
    
    temp_ref = gridded_data['temp_ref']
    gdp_target = gridded_data['gdp_target'] 
    lat = gridded_data['lat']
    
    for target_config in target_configs:
        target_name = target_config['target_name']
        target_type = target_config['target_type']
        
        if target_type == 'constant':
            reduction_array = calculate_constant_target_reduction(
                target_config['gdp_reduction'], temp_ref.shape
            )
            result = {
                'target_type': target_type,
                'reduction_array': reduction_array,
                'coefficients': None,
                'constraint_verification': None,
                'global_statistics': {
                    'gdp_weighted_mean': target_config['gdp_reduction']
                }
            }
            
        elif target_type == 'linear':
            result = calculate_linear_target_reduction(target_config, temp_ref, gdp_target, lat)
            result['target_type'] = target_type
            
        elif target_type == 'quadratic':
            result = calculate_quadratic_target_reduction(target_config, temp_ref, gdp_target, lat)
            result['target_type'] = target_type
            
        else:
            raise ValueError(f"Unknown target type: {target_type}")
            
        results[target_name] = result
    
    return results


# =============================================================================
# Centralized NetCDF Data Loading Functions
# For efficient loading of all SSP scenario data upfront
# =============================================================================

def load_all_netcdf_data(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load all NetCDF files for all SSP scenarios at start of processing.
    
    This function loads all gridded data upfront to avoid repeated file I/O
    operations during processing steps. Since NetCDF files are small, this 
    approach optimizes performance by loading everything into memory once.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Integrated configuration dictionary containing:
        - climate_model: model name and file patterns
        - ssp_scenarios: reference_ssp and forward_simulation_ssps
        - time_periods: reference and target period specifications
    
    Returns
    -------
    Dict[str, Any]
        Nested dictionary organized as:
        data[ssp_name][data_type] = array
        
        Structure:
        {
            'ssp245': {
                'temperature': np.array([lat, lon, time]),  # °C
                'precipitation': np.array([lat, lon, time]), # mm/day  
                'gdp': np.array([lat, lon, time]),          # economic units
                'population': np.array([lat, lon, time])    # people
            },
            'ssp585': { ... },
            '_metadata': {
                'lat': np.array([lat]),           # latitude coordinates
                'lon': np.array([lon]),           # longitude coordinates  
                'time_periods': time_periods,     # reference and target periods
                'ssp_list': ['ssp245', 'ssp585', ...],  # all loaded SSPs
                'grid_shape': (nlat, nlon),       # spatial dimensions
                'time_shape': ntime               # temporal dimension
            }
        }
    """
    import os
    
    print("\n" + "="*60)
    print("LOADING ALL NETCDF DATA")  
    print("="*60)
    
    # Extract configuration
    model_name = config['climate_model']['model_name']
    reference_ssp = config['ssp_scenarios']['reference_ssp']
    forward_ssps = config['ssp_scenarios']['forward_simulation_ssps']
    time_periods = config['time_periods']
    
    # Create comprehensive SSP list (reference + forward, deduplicated)
    all_ssps = list(set([reference_ssp] + forward_ssps))
    all_ssps.sort()  # Consistent ordering
    
    print(f"Climate model: {model_name}")
    print(f"Reference SSP: {reference_ssp}")
    print(f"All SSPs to load: {all_ssps}")
    print(f"Total SSP scenarios: {len(all_ssps)}")
    
    # Initialize data structure
    all_data = {
        '_metadata': {
            'ssp_list': all_ssps,
            'time_periods': time_periods,
            'model_name': model_name
        }
    }
    
    # Load data for each SSP scenario
    for i, ssp_name in enumerate(all_ssps):
        print(f"\nLoading SSP scenario: {ssp_name} ({i+1}/{len(all_ssps)})")
        
        try:
            # Resolve file paths for this SSP
            temp_file = resolve_netcdf_filepath(config, 'temperature', ssp_name) 
            precip_file = resolve_netcdf_filepath(config, 'precipitation', ssp_name)
            gdp_file = resolve_netcdf_filepath(config, 'gdp', ssp_name)
            pop_file = resolve_netcdf_filepath(config, 'population', ssp_name)
            
            print(f"  Temperature: {os.path.basename(temp_file)}")
            print(f"  Precipitation: {os.path.basename(precip_file)}")
            print(f"  GDP: {os.path.basename(gdp_file)}")
            print(f"  Population: {os.path.basename(pop_file)}")
            
            # Load gridded data for this SSP using existing function
            ssp_data = load_gridded_data(model_name, ssp_name)
            
            # Store in organized structure
            all_data[ssp_name] = {
                'temperature': ssp_data['tas'],      # [lat, lon, time]
                'precipitation': ssp_data['pr'],     # [lat, lon, time]  
                'gdp': ssp_data['gdp'],              # [lat, lon, time]
                'population': ssp_data['pop'],       # [lat, lon, time]
                'temperature_years': ssp_data['tas_years'],
                'precipitation_years': ssp_data['pr_years'],
                'gdp_years': ssp_data['gdp_years'],
                'population_years': ssp_data['pop_years']
            }
            
            # Store metadata from first SSP (coordinates same for all)
            if i == 0:
                all_data['_metadata'].update({
                    'lat': ssp_data['lat'],
                    'lon': ssp_data['lon'],
                    'grid_shape': (len(ssp_data['lat']), len(ssp_data['lon'])),
                    'time_shape': len(ssp_data['tas_years'])  # Assuming all same length
                })
            
            print(f"  Data shape: {ssp_data['tas'].shape}")
            print(f"  ✅ Successfully loaded {ssp_name}")
            
        except Exception as e:
            print(f"  ❌ Failed to load {ssp_name}: {e}")
            raise RuntimeError(f"Could not load data for {ssp_name}: {e}")
    
    # Summary information
    nlat, nlon = all_data['_metadata']['grid_shape']
    ntime = all_data['_metadata']['time_shape']
    total_grid_cells = nlat * nlon
    
    print(f"\n📊 Data Loading Summary:")
    print(f"  Grid dimensions: {nlat} × {nlon} = {total_grid_cells} cells")
    print(f"  Time dimension: {ntime} years")
    print(f"  SSP scenarios loaded: {len(all_ssps)}")
    print(f"  Total data arrays: {len(all_ssps) * 4} (4 variables × {len(all_ssps)} SSPs)")
    
    # Estimate memory usage
    bytes_per_array = nlat * nlon * ntime * 8  # 8 bytes per float64
    total_arrays = len(all_ssps) * 4  # 4 data types per SSP
    total_bytes = bytes_per_array * total_arrays
    total_mb = total_bytes / (1024 * 1024)
    
    print(f"  Estimated memory usage: {total_mb:.1f} MB")
    print("  ✅ All NetCDF data loaded successfully")

    # Create global valid mask (check all time points for economic activity)
    print("Computing global valid grid cell mask...")

    # Use reference SSP for validity checking
    ref_gdp = all_data[reference_ssp]['gdp']  # [lat, lon, time]
    ref_pop = all_data[reference_ssp]['population']  # [lat, lon, time]

    valid_mask = np.zeros((nlat, nlon), dtype=bool)
    valid_count = 0

    for lat_idx in range(nlat):
        for lon_idx in range(nlon):
            gdp_timeseries = ref_gdp[lat_idx, lon_idx, :]
            pop_timeseries = ref_pop[lat_idx, lon_idx, :]

            # Grid cell is valid if GDP and population are positive at ALL time points
            if np.all(gdp_timeseries > 0) and np.all(pop_timeseries > 0):
                valid_mask[lat_idx, lon_idx] = True
                valid_count += 1

    print(f"  Valid economic grid cells: {valid_count} / {total_grid_cells} ({100*valid_count/total_grid_cells:.1f}%)")

    # Add valid mask to metadata
    all_data['_metadata']['valid_mask'] = valid_mask
    all_data['_metadata']['valid_count'] = valid_count

    return all_data


def resolve_netcdf_filepath(config: Dict[str, Any], data_type: str, ssp_name: str) -> str:
    """
    Resolve NetCDF file path using naming convention: {prefix}_{model_name}_{ssp_name}.nc
    
    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary
    data_type : str
        Type of data ('temperature', 'precipitation', 'gdp', 'population', 'target_reductions')
    ssp_name : str
        SSP scenario name (e.g., 'ssp245', 'ssp585')
        
    Returns
    -------
    str
        Full path to NetCDF file
    """
    climate_model = config['climate_model']
    model_name = climate_model['model_name']
    input_dir = climate_model['input_directory']
    prefix = climate_model['netcdf_file_patterns'][data_type]
    
    filename = f"{prefix}_{model_name}_{ssp_name}.nc"
    filepath = os.path.join(input_dir, filename)
    
    return filepath


def get_ssp_data(all_data: Dict[str, Any], ssp_name: str, data_type: str) -> np.ndarray:
    """
    Extract specific data array from loaded NetCDF data structure.
    
    Parameters
    ----------
    all_data : Dict[str, Any]
        Result from load_all_netcdf_data()
    ssp_name : str
        SSP scenario name (e.g., 'ssp245')
    data_type : str
        Data type ('temperature', 'precipitation', 'gdp', 'population')
        
    Returns
    -------
    np.ndarray
        Data array with shape [lat, lon, time]
    """
    if ssp_name not in all_data:
        raise KeyError(f"SSP scenario '{ssp_name}' not found in loaded data. Available: {all_data['_metadata']['ssp_list']}")
    
    if data_type not in all_data[ssp_name]:
        raise KeyError(f"Data type '{data_type}' not found for {ssp_name}. Available: {list(all_data[ssp_name].keys())}")
    
    return all_data[ssp_name][data_type]


def get_grid_metadata(all_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract grid metadata from loaded NetCDF data structure.
    
    Parameters
    ----------
    all_data : Dict[str, Any]
        Result from load_all_netcdf_data()
        
    Returns
    -------
    Dict[str, Any]
        Metadata dictionary containing coordinates and dimensions
    """
    return all_data['_metadata']


# =============================================================================
# TFP Calculation Functions
# =============================================================================

def calculate_tfp_coin_ssp(pop, gdp, params):
    """
    Calculate total factor productivity time series using the Solow-Swan growth model.

    Parameters
    ----------
    pop : array-like
        Time series of population (L) in people
    gdp : array-like
        Time series of gross domestic product (Y) in $/yr
    params : dict or ModelParams
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


# =============================================================================
# Output Writing Functions
# =============================================================================

def save_step1_results_netcdf(target_results: Dict[str, Any], output_path: str) -> str:
    """
    Save Step 1 target GDP reduction results to NetCDF file.
    
    Parameters
    ----------
    target_results : Dict[str, Any]
        Results from step1_calculate_target_gdp_changes()
    output_path : str
        Complete output file path
        
    Returns
    -------
    str
        Path to saved NetCDF file
    """
    import xarray as xr
    import os
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Extract metadata
    metadata = target_results['_metadata']
    
    # Create arrays for each target type
    target_arrays = []
    target_names = []
    
    for target_name, target_data in target_results.items():
        if target_name != '_metadata':
            target_arrays.append(target_data['reduction_array'])
            target_names.append(target_name)
    
    # Stack arrays: (target_type, lat, lon)
    target_reductions = np.stack(target_arrays, axis=0)
    
    # Create xarray dataset
    ds = xr.Dataset(
        {
            'target_gdp_reductions': (['target_type', 'lat', 'lon'], target_reductions),
            'temperature_ref': (['lat', 'lon'], metadata['temp_ref']),
            'gdp_target': (['lat', 'lon'], metadata['gdp_target'])
        },
        coords={
            'target_type': target_names,
            'lat': metadata['lat'],
            'lon': metadata['lon']
        }
    )
    
    # Add attributes
    ds.target_gdp_reductions.attrs = {
        'long_name': 'Target GDP reductions',
        'units': 'fractional reduction',
        'description': f'Target reduction patterns for {len(target_names)} cases'
    }
    
    ds.temperature_ref.attrs = {
        'long_name': 'Reference period temperature',
        'units': '°C'
    }
    
    ds.gdp_target.attrs = {
        'long_name': 'Target period GDP',
        'units': 'economic units'
    }
    
    # Add global attributes
    ds.attrs = {
        'title': 'COIN-SSP Target GDP Reductions - Step 1 Results',
        'reference_ssp': metadata['reference_ssp'],
        'reference_period': f"{metadata['time_periods']['reference_period']['start_year']}-{metadata['time_periods']['reference_period']['end_year']}",
        'target_period': f"{metadata['time_periods']['target_period']['start_year']}-{metadata['time_periods']['target_period']['end_year']}",
        'global_temp_ref': metadata['global_temp_ref'],
        'global_gdp_target': metadata['global_gdp_target'],
        'creation_date': datetime.now().isoformat()
    }
    
    # Save to file
    ds.to_netcdf(output_path)
    print(f"Step 1 results saved to {output_path}")
    return output_path


def save_step2_results_netcdf(tfp_results: Dict[str, Any], output_path: str, model_name: str) -> str:
    """
    Save Step 2 baseline TFP results to NetCDF file.
    
    Parameters
    ----------
    tfp_results : Dict[str, Any]
        Results from step2_calculate_baseline_tfp()
    output_path : str
        Complete output file path  
    model_name : str
        Climate model name
        
    Returns
    -------
    str
        Path to saved NetCDF file
    """
    import xarray as xr
    import os
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Get SSP names and dimensions from first result
    ssp_names = [ssp for ssp in tfp_results.keys() if ssp != '_coordinates']
    first_ssp = tfp_results[ssp_names[0]]
    nlat, nlon, ntime = first_ssp['tfp_baseline'].shape
    
    # Create coordinate arrays that match the actual data dimensions
    lat_coords = np.arange(nlat)  # Use actual latitude dimension
    lon_coords = np.arange(nlon)  # Use actual longitude dimension
    
    # Create arrays for all SSPs
    tfp_all_ssps = np.full((len(ssp_names), nlat, nlon, ntime), np.nan)
    k_all_ssps = np.full((len(ssp_names), nlat, nlon, ntime), np.nan)
    valid_masks = np.full((len(ssp_names), nlat, nlon), False)
    
    for i, ssp_name in enumerate(ssp_names):
        tfp_all_ssps[i] = tfp_results[ssp_name]['tfp_baseline']
        k_all_ssps[i] = tfp_results[ssp_name]['k_baseline']
        valid_masks[i] = tfp_results[ssp_name]['valid_mask']
    
    # Create coordinate arrays (assuming annual time steps starting from year 0)
    time_coords = np.arange(ntime)
    
    # Create xarray dataset
    ds = xr.Dataset(
        {
            'tfp_baseline': (['ssp', 'lat', 'lon', 'time'], tfp_all_ssps),
            'k_baseline': (['ssp', 'lat', 'lon', 'time'], k_all_ssps),
            'valid_mask': (['ssp', 'lat', 'lon'], valid_masks)
        },
        coords={
            'ssp': ssp_names,
            'lat': lat_coords,
            'lon': lon_coords,
            'time': time_coords
        }
    )
    
    # Add attributes
    ds.tfp_baseline.attrs = {
        'long_name': 'Baseline Total Factor Productivity',
        'units': 'normalized to year 0',
        'description': 'TFP time series without climate effects, calculated using Solow-Swan model'
    }
    
    ds.k_baseline.attrs = {
        'long_name': 'Baseline Capital Stock',
        'units': 'normalized to year 0', 
        'description': 'Capital stock time series without climate effects'
    }
    
    ds.valid_mask.attrs = {
        'long_name': 'Valid economic grid cells',
        'units': 'boolean',
        'description': 'True for grid cells with economic activity (GDP > 0 and population > 0)'
    }
    
    # Add global attributes  
    total_processed = sum(result['grid_cells_processed'] for ssp_name, result in tfp_results.items() 
                         if ssp_name != '_coordinates')
    ds.attrs = {
        'title': 'COIN-SSP Baseline TFP - Step 2 Results',
        'model_name': model_name,
        'total_grid_cells_processed': total_processed,
        'ssp_scenarios': ', '.join(ssp_names),
        'description': 'Baseline Total Factor Productivity calculated for each SSP scenario without climate effects',
        'creation_date': datetime.now().isoformat()
    }
    
    # Save to file
    ds.to_netcdf(output_path)
    print(f"Step 2 results saved to {output_path}")
    return output_path


def save_step3_results_netcdf(scaling_results: Dict[str, Any], output_path: str, model_name: str) -> str:
    """
    Save Step 3 scaling factor results to NetCDF file.
    
    Parameters
    ----------
    scaling_results : Dict[str, Any]
        Results from step3_calculate_scaling_factors_per_cell()
    output_path : str
        Complete output file path
    model_name : str
        Climate model name
        
    Returns
    -------
    str
        Path to saved NetCDF file
    """
    import xarray as xr
    import os
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Extract data arrays and metadata
    scaling_factors = scaling_results['scaling_factors']  # [lat, lon, damage_func, target]
    optimization_errors = scaling_results['optimization_errors']  # [lat, lon, damage_func, target]
    convergence_flags = scaling_results['convergence_flags']  # [lat, lon, damage_func, target]
    scaled_parameters = scaling_results['scaled_parameters']  # [lat, lon, damage_func, target, param]
    valid_mask = scaling_results['valid_mask']  # [lat, lon]
    
    # Get dimensions and coordinate info
    nlat, nlon, n_damage_func, n_target = scaling_factors.shape
    n_scaled_params = scaled_parameters.shape[4]
    damage_function_names = scaling_results['damage_function_names']
    target_names = scaling_results['target_names']
    scaled_param_names = scaling_results['scaled_param_names']
    coordinates = scaling_results['_coordinates']
    
    # Create xarray dataset
    ds = xr.Dataset(
        {
            'scaling_factors': (['lat', 'lon', 'damage_func', 'target'], scaling_factors),
            'optimization_errors': (['lat', 'lon', 'damage_func', 'target'], optimization_errors),
            'convergence_flags': (['lat', 'lon', 'damage_func', 'target'], convergence_flags),
            'scaled_parameters': (['lat', 'lon', 'damage_func', 'target', 'param'], scaled_parameters),
            'valid_mask': (['lat', 'lon'], valid_mask)
        },
        coords={
            'lat': coordinates['lat'],
            'lon': coordinates['lon'],
            'damage_func': damage_function_names,
            'target': target_names,
            'param': scaled_param_names
        }
    )
    
    # Add attributes
    ds.scaling_factors.attrs = {
        'long_name': 'Climate response scaling factors',
        'units': 'dimensionless',
        'description': 'Optimized scaling factors for climate damage functions per grid cell'
    }
    
    ds.optimization_errors.attrs = {
        'long_name': 'Optimization error values',
        'units': 'dimensionless', 
        'description': 'Final objective function values from scaling factor optimization'
    }
    
    ds.convergence_flags.attrs = {
        'long_name': 'Optimization convergence flags',
        'units': 'boolean',
        'description': 'True where optimization converged successfully'
    }
    
    ds.scaled_parameters.attrs = {
        'long_name': 'Scaled climate damage function parameters',
        'units': 'various',
        'description': 'Climate damage function parameters (scaling_factor × base_parameter) for each grid cell and combination',
        'parameter_names': ', '.join(scaled_param_names),
        'parameter_groups': 'Capital: k_tas1,k_tas2,k_pr1,k_pr2; TFP: tfp_tas1,tfp_tas2,tfp_pr1,tfp_pr2; Output: y_tas1,y_tas2,y_pr1,y_pr2',
        'climate_variables': 'tas=temperature, pr=precipitation; 1=linear, 2=quadratic'
    }
    
    ds.valid_mask.attrs = {
        'long_name': 'Valid economic grid cells',
        'units': 'boolean',
        'description': 'True for grid cells with economic activity used in optimization'
    }
    
    # Add global attributes
    ds.attrs = {
        'title': 'COIN-SSP Scaling Factors - Step 3 Results',
        'model_name': model_name,
        'reference_ssp': scaling_results['reference_ssp'],
        'total_grid_cells': scaling_results['total_grid_cells'],
        'successful_optimizations': scaling_results['successful_optimizations'],
        'success_rate_percent': 100 * scaling_results['successful_optimizations'] / max(1, scaling_results['total_grid_cells']),
        'damage_functions': ', '.join(damage_function_names),
        'target_patterns': ', '.join(target_names),
        'description': 'Per-grid-cell scaling factors optimized using reference SSP scenario',
        'creation_date': datetime.now().isoformat()
    }
    
    # Save to file
    ds.to_netcdf(output_path)
    print(f"Step 3 results saved to {output_path}")
    return output_path


def save_step4_results_netcdf(step4_results: Dict[str, Any], output_path: str, model_name: str) -> str:
    """
    Save Step 4 forward model results to NetCDF file.
    
    Parameters
    ----------
    step4_results : Dict[str, Any]
        Results from step4_forward_integration_all_ssps()
    output_path : str
        Complete output file path
    model_name : str
        Climate model name
        
    Returns
    -------
    str
        Path to saved NetCDF file
    """
    import xarray as xr
    import os
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Extract metadata and structure
    forward_results = step4_results['forward_results']
    damage_function_names = step4_results['damage_function_names']
    target_names = step4_results['target_names']
    valid_mask = step4_results['valid_mask']
    coordinates = step4_results['_coordinates']
    
    # Get SSP names and dimensions from first SSP result
    ssp_names = list(forward_results.keys())
    first_ssp = forward_results[ssp_names[0]]
    nlat, nlon, n_damage_func, n_target, ntime = first_ssp['gdp_climate'].shape
    
    # Create arrays for all SSPs: [ssp, lat, lon, damage_func, target, time]
    n_ssps = len(ssp_names)
    gdp_climate_all = np.full((n_ssps, nlat, nlon, n_damage_func, n_target, ntime), np.nan)
    gdp_weather_all = np.full((n_ssps, nlat, nlon, n_damage_func, n_target, ntime), np.nan)
    tfp_climate_all = np.full((n_ssps, nlat, nlon, n_damage_func, n_target, ntime), np.nan)
    tfp_weather_all = np.full((n_ssps, nlat, nlon, n_damage_func, n_target, ntime), np.nan)
    k_climate_all = np.full((n_ssps, nlat, nlon, n_damage_func, n_target, ntime), np.nan)
    k_weather_all = np.full((n_ssps, nlat, nlon, n_damage_func, n_target, ntime), np.nan)
    
    # Stack results from all SSPs
    for i, ssp_name in enumerate(ssp_names):
        ssp_result = forward_results[ssp_name]
        gdp_climate_all[i] = ssp_result['gdp_climate']
        gdp_weather_all[i] = ssp_result['gdp_weather']
        tfp_climate_all[i] = ssp_result['tfp_climate']
        tfp_weather_all[i] = ssp_result['tfp_weather']
        k_climate_all[i] = ssp_result['k_climate']
        k_weather_all[i] = ssp_result['k_weather']
    
    # Create time coordinate (assuming annual steps starting from 2015)
    time_coords = np.arange(ntime)
    
    # Create xarray dataset
    ds = xr.Dataset(
        {
            'gdp_climate': (['ssp', 'lat', 'lon', 'damage_func', 'target', 'time'], gdp_climate_all),
            'gdp_weather': (['ssp', 'lat', 'lon', 'damage_func', 'target', 'time'], gdp_weather_all),
            'tfp_climate': (['ssp', 'lat', 'lon', 'damage_func', 'target', 'time'], tfp_climate_all),
            'tfp_weather': (['ssp', 'lat', 'lon', 'damage_func', 'target', 'time'], tfp_weather_all),
            'k_climate': (['ssp', 'lat', 'lon', 'damage_func', 'target', 'time'], k_climate_all),
            'k_weather': (['ssp', 'lat', 'lon', 'damage_func', 'target', 'time'], k_weather_all),
            'valid_mask': (['lat', 'lon'], valid_mask)
        },
        coords={
            'ssp': ssp_names,
            'lat': coordinates['lat'],
            'lon': coordinates['lon'],
            'damage_func': damage_function_names,
            'target': target_names,
            'time': time_coords
        }
    )
    
    # Add attributes
    ds.gdp_climate.attrs = {
        'long_name': 'GDP with climate effects',
        'units': 'economic units per year',
        'description': 'GDP projections including full climate change effects'
    }
    
    ds.gdp_weather.attrs = {
        'long_name': 'GDP with weather effects only',
        'units': 'economic units per year',
        'description': 'GDP projections with weather variability but no climate trends'
    }
    
    ds.tfp_climate.attrs = {
        'long_name': 'Total Factor Productivity with climate effects', 
        'units': 'normalized to year 0',
        'description': 'TFP projections including full climate change effects'
    }
    
    ds.tfp_weather.attrs = {
        'long_name': 'Total Factor Productivity with weather effects only',
        'units': 'normalized to year 0',
        'description': 'TFP projections with weather variability but no climate trends'
    }
    
    ds.k_climate.attrs = {
        'long_name': 'Capital stock with climate effects',
        'units': 'normalized to year 0',
        'description': 'Capital stock projections including full climate change effects'
    }
    
    ds.k_weather.attrs = {
        'long_name': 'Capital stock with weather effects only', 
        'units': 'normalized to year 0',
        'description': 'Capital stock projections with weather variability but no climate trends'
    }
    
    ds.valid_mask.attrs = {
        'long_name': 'Valid economic grid cells',
        'units': 'boolean',
        'description': 'True for grid cells with economic activity used in forward modeling'
    }
    
    # Add global attributes
    total_successful = sum(forward_results[ssp]['successful_forward_runs'] for ssp in ssp_names)
    total_runs = sum(forward_results[ssp]['total_forward_runs'] for ssp in ssp_names)
    
    ds.attrs = {
        'title': 'COIN-SSP Forward Model Results - Step 4 Results',
        'model_name': model_name,
        'total_ssps_processed': step4_results['total_ssps_processed'],
        'ssp_scenarios': ', '.join(ssp_names),
        'damage_functions': ', '.join(damage_function_names),
        'target_patterns': ', '.join(target_names), 
        'total_forward_runs': total_runs,
        'successful_forward_runs': total_successful,
        'overall_success_rate_percent': 100 * total_successful / max(1, total_runs),
        'description': 'Climate-integrated economic projections using per-grid-cell scaling factors for all SSP scenarios',
        'creation_date': datetime.now().isoformat()
    }
    
    # Save to file
    ds.to_netcdf(output_path)
    print(f"Step 4 results saved to {output_path}")
    return output_path
