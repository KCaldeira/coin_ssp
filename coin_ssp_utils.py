import numpy as np
import statsmodels.api as sm
import copy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
import xarray as xr

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
# Target GDP Utilities (moved from coin_ssp_target_gdp.py)
# =============================================================================

def load_gridded_data():
    """
    Load all four NetCDF files and return as a unified dataset.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'tas': temperature data (axis_0, lat, lon) 
        - 'pr': precipitation data (axis_0, lat, lon)
        - 'gdp': GDP data (axis_0, lat, lon) 
        - 'pop': population data (time, lat, lon)
        - 'lat': latitude coordinates
        - 'lon': longitude coordinates
        - 'tas_years': temperature time axis (0-85, assuming 2015-2100)
        - 'pr_years': precipitation time axis (0-85, assuming 2015-2100) 
        - 'gdp_years': GDP time axis (0-17, assuming 5-year intervals)
        - 'pop_years': population time axis (years since reference)
    """
    
    # Load temperature data
    tas_ds = xr.open_dataset('data/input/gridRaw_tas_CanESM5_ssp585.nc', decode_times=False)
    tas = tas_ds.tas.values - 273.15  # Convert from Kelvin to Celsius
    
    # Load precipitation data  
    pr_ds = xr.open_dataset('data/input/gridRaw_pr_CanESM5_ssp585.nc', decode_times=False)
    pr = pr_ds.pr.values  # (86, 64, 128)
    
    # Load GDP data
    gdp_ds = xr.open_dataset('data/input/gridded_gdp_regrid_CanESM5.nc', decode_times=False)
    gdp = gdp_ds.gdp_20202100ssp5.values  # (18, 64, 128)
    
    # Load population data
    pop_ds = xr.open_dataset('data/input/gridded_pop_regrid_CanESM5.nc', decode_times=False)
    pop = pop_ds.pop_20062100ssp5.values  # (95, 64, 128)
    
    # Get coordinate arrays
    lat = tas_ds.lat.values
    lon = tas_ds.lon.values
    
    # Create year arrays (assuming standard time conventions)
    tas_years = np.arange(2015, 2015 + len(tas_ds.axis_0))  # 2015-2100
    pr_years = np.arange(2015, 2015 + len(pr_ds.axis_0))    # 2015-2100
    gdp_years = np.arange(2020, 2020 + len(gdp_ds.axis_0) * 5, 5)  # 2020-2100 in 5-year steps
    pop_years = np.arange(2006, 2006 + len(pop_ds.time))     # 2006-2100
    
    return {
        'tas': tas,
        'pr': pr, 
        'gdp': gdp,
        'pop': pop,
        'lat': lat,
        'lon': lon,
        'tas_years': tas_years,
        'pr_years': pr_years,
        'gdp_years': gdp_years,
        'pop_years': pop_years
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
