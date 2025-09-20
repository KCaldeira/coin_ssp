#!/usr/bin/env python3

import json
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
from coin_ssp_utils import (
    load_gridded_data, calculate_time_means, calculate_global_mean,
    calculate_all_target_reductions
)

def load_config(config_file):
    """Load configuration from JSON file."""
    with open(config_file, 'r') as f:
        return json.load(f)

def calculate_target_reductions(config_file):
    """
    Calculate target GDP reductions for constant, linear, and quadratic functions.
    
    Parameters
    ----------
    config_file : str
        Path to JSON configuration file
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'constant': 2D array (lat, lon) of constant GDP reductions
        - 'linear': 2D array (lat, lon) of linear GDP reductions  
        - 'quadratic': 2D array (lat, lon) of quadratic GDP reductions
        - 'temperature_ref': 2D array (lat, lon) of reference period temperature
        - 'temperature_target': 2D array (lat, lon) of target period temperature
        - 'gdp_target': 2D array (lat, lon) of target period GDP
        - 'lat': latitude coordinates
        - 'lon': longitude coordinates
        - 'global_stats': dictionary of global mean statistics
    """
    
    # Load configuration
    config = load_config(config_file)
    
    # Load gridded data
    print("Loading gridded data...")
    model_name = config.get('model_name', 'CanESM5')
    case_name = config.get('case_name', 'ssp585')
    # Create temporary config structure for load_gridded_data
    temp_config = {
        'climate_model': {
            'model_name': model_name,
            'input_directory': 'data/input',
            'netcdf_file_patterns': {
                'temperature': 'gridRawAlt_tas',
                'precipitation': 'gridRawAlt_pr',
                'gdp': 'Gridded_GDPdensity',
                'population': 'Gridded_POPdensity'
            }
        }
    }
    data = load_gridded_data(temp_config, model_name, case_name)
    
    # Extract configuration parameters
    ref_start = config['reference_period']['start_year']
    ref_end = config['reference_period']['end_year'] 
    target_start = config['target_period']['start_year']
    target_end = config['target_period']['end_year']
    
    print(f"Reference period: {ref_start}-{ref_end}")
    print(f"Target period: {target_start}-{target_end}")
    
    # Calculate time means
    print("Calculating temporal means...")
    
    # Temperature reference period mean
    temp_ref = calculate_time_means(data['tas'], data['tas_years'], ref_start, ref_end)
    
    # GDP target period mean 
    gdp_target = calculate_time_means(data['gdp'], data['gdp_years'], target_start, target_end)
    
    # Calculate global means for verification
    print("Calculating global means...")
    global_temp_ref = calculate_global_mean(temp_ref, data['lat'])
    global_gdp_target = calculate_global_mean(gdp_target, data['lat'])
    
    # Calculate GDP-weighted global mean temperature for reference period
    gdp_weighted_temp_ref = calculate_global_mean(gdp_target * temp_ref, data['lat']) / global_gdp_target
    
    print(f"Global mean reference temperature: {global_temp_ref:.2f}°C")
    print(f"GDP-weighted global mean reference temperature: {gdp_weighted_temp_ref:.2f}°C")
    print(f"Global mean target GDP: {global_gdp_target:.2e}")
    
    # Calculate target reductions using extracted functions
    print("Calculating target GDP reductions...")
    
    # Prepare gridded data for the calculation functions
    gridded_data = {
        'temp_ref': temp_ref,
        'gdp_target': gdp_target,
        'lat': data['lat']
    }
    
    # Convert old-style config to new-style target configurations
    target_configs = [
        {
            'target_name': 'constant',
            'target_shape': 'constant',
            'gdp_amount': config['constant_target']['gdp_amount']
        },
        {
            'target_name': 'linear', 
            'target_shape': 'linear',
            'global_mean_amount': config['linear_target']['global_mean_amount'],
            'reference_temperature': config['linear_target']['reference_temperature'],
            'amount_at_reference_temp': config['linear_target']['amount_at_reference_temp']
        },
        {
            'target_name': 'quadratic',
            'target_shape': 'quadratic', 
            'global_mean_amount': config['quadratic_target']['global_mean_amount'],
            'reference_temperature': config['quadratic_target']['reference_temperature'],
            'amount_at_reference_temp': config['quadratic_target']['amount_at_reference_temp'],
            'zero_amount_temperature': config['quadratic_target']['zero_amount_temperature']
        }
    ]
    
    # Calculate all target reductions using extracted functions
    calculation_results = calculate_all_target_reductions(target_configs, gridded_data)
    
    # Extract arrays and print verification information (maintaining existing output format)
    constant_reduction = calculation_results['constant']['reduction_array']
    
    linear_result = calculation_results['linear']
    linear_reduction = linear_result['reduction_array']
    linear_coeffs = linear_result['coefficients']
    linear_verification = linear_result['constraint_verification']
    
    print(f"Linear algorithm check:")
    print(f"  T_ref_linear: {target_configs[1]['reference_temperature']}°C")
    print(f"  GDP-weighted temp mean: {linear_result['gdp_weighted_temp_mean']:.6f}°C")
    print(f"  value_at_ref_linear: {target_configs[1]['amount_at_reference_temp']}")
    print(f"  global_mean_linear: {target_configs[1]['global_mean_amount']}")
    print(f"  OLS solution verification:")
    print(f"    At T_ref: {linear_verification['point_constraint']['achieved']:.10f} (target: {linear_verification['point_constraint']['target']})")
    print(f"    Global mean: {linear_verification['global_mean_constraint']['achieved']:.10f} (target: {linear_verification['global_mean_constraint']['target']:.10f})")
    print(f"  Linear coefficients: a0={linear_coeffs['a0']:.10f}, a1={linear_coeffs['a1']:.10f}")
    
    quadratic_result = calculation_results['quadratic']
    quadratic_reduction = quadratic_result['reduction_array']
    quadratic_coeffs = quadratic_result['coefficients']
    quadratic_verification = quadratic_result['constraint_verification']
    
    print(f"Quadratic algorithm check:")
    print(f"  T0 (zero point): {target_configs[2]['zero_amount_temperature']}°C")
    print(f"  T_ref_quad: {target_configs[2]['reference_temperature']}°C")
    print(f"  GDP-weighted temp mean: {quadratic_result['gdp_weighted_temp_mean']:.6f}°C")
    print(f"  GDP-weighted temp² mean: {quadratic_result['gdp_weighted_temp2_mean']:.6f}°C²")
    print(f"  value_at_ref_quad: {target_configs[2]['amount_at_reference_temp']}")
    print(f"  global_mean_quad: {target_configs[2]['global_mean_amount']}")
    print(f"  Constraint verification:")
    print(f"    At T0=0: {quadratic_verification['zero_point_constraint']['achieved']:.10f} (target: {quadratic_verification['zero_point_constraint']['target']})")
    print(f"    At T_ref: {quadratic_verification['reference_point_constraint']['achieved']:.10f} (target: {quadratic_verification['reference_point_constraint']['target']})")
    print(f"    Global mean: {quadratic_verification['global_mean_constraint']['achieved']:.10f} (target: {quadratic_verification['global_mean_constraint']['target']:.10f})")
    
    # Extract coefficients for backward compatibility
    a0_linear, a1_linear = linear_coeffs['a0'], linear_coeffs['a1']
    a_quad, b_quad, c_quad = quadratic_coeffs['a'], quadratic_coeffs['b'], quadratic_coeffs['c']
    
    # Package results
    results = {
        'constant': constant_reduction,
        'linear': linear_reduction, 
        'quadratic': quadratic_reduction,
        'temperature_ref': temp_ref,
        'gdp_target': gdp_target,
        'lat': data['lat'],
        'lon': data['lon'],
        'global_stats': {
            'global_temp_ref': global_temp_ref,
            'global_gdp_target': global_gdp_target,
            'linear_coeffs': {'a0': a0_linear, 'a1': a1_linear},
            'quadratic_coeffs': {'a': a_quad, 'b': b_quad, 'c': c_quad}
        }
    }
    
    print("Target GDP reduction calculations completed.")
    print(f"Linear coefficients: a0={a0_linear:.10f}, a1={a1_linear:.10f}")
    print(f"Quadratic coefficients: a={a_quad:.10f}, b={b_quad:.10f}, c={c_quad:.12f}")
    
    return results

def save_results_netcdf(results, output_filename):
    """
    Save results to NetCDF file in data/output directory.
    
    Parameters
    ----------
    results : dict
        Results from calculate_target_reductions()
    output_filename : str
        Output NetCDF filename (without path)
    """
    
    # Ensure output directory exists
    os.makedirs('data/output', exist_ok=True)
    output_path = os.path.join('data/output', output_filename)
    
    # Create 3D array: (3, lat, lon) for constant, linear, quadratic
    target_reductions = np.stack([
        results['constant'],
        results['linear'], 
        results['quadratic']
    ], axis=0)
    
    # Create xarray dataset
    ds = xr.Dataset(
        {
            'target_gdp_amounts': (['reduction_type', 'lat', 'lon'], target_reductions),
            'temperature_ref': (['lat', 'lon'], results['temperature_ref']),
            'gdp_target': (['lat', 'lon'], results['gdp_target'])
        },
        coords={
            'reduction_type': ['constant', 'linear', 'quadratic'],
            'lat': results['lat'],
            'lon': results['lon']
        }
    )
    
    # Add attributes
    ds.target_gdp_amounts.attrs = {
        'long_name': 'Target GDP reductions',
        'units': 'fractional reduction',
        'description': 'Layer 0: constant, Layer 1: linear in temperature, Layer 2: quadratic in temperature'
    }
    
    ds.temperature_ref.attrs = {
        'long_name': 'Reference period temperature',
        'units': '°C'
    }
    
    ds.gdp_target.attrs = {
        'long_name': 'Target period GDP',
        'units': 'economic units'
    }
    
    # Save to file
    ds.to_netcdf(output_path)
    print(f"Results saved to {output_path}")
    return output_path

def create_global_maps(results, config, pdf_filename):
    """
    Create global maps of target GDP reductions with red-white-blue colormap.
    
    Parameters
    ----------
    results : dict
        Results from calculate_target_reductions()
    config : dict
        Configuration dictionary with target values
    pdf_filename : str
        Output PDF filename (without path)
    """
    
    # Ensure output directory exists
    os.makedirs('data/output', exist_ok=True)
    pdf_path = os.path.join('data/output', pdf_filename)
    
    # Extract data
    constant = results['constant']
    linear = results['linear']
    quadratic = results['quadratic']
    gdp_target = results['gdp_target']
    lat = results['lat']
    lon = results['lon']
    
    # Create meshgrid for plotting
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    
    # Set fixed color scale limits
    vmin = -1.0
    vmax = 1.0
    
    # Calculate actual data ranges and global means for annotation
    all_data = np.concatenate([constant.flatten(), linear.flatten(), quadratic.flatten()])
    actual_min = np.min(all_data)
    actual_max = np.max(all_data)
    
    # Calculate global means for each reduction type
    global_mean_constant = np.mean(constant)
    global_mean_linear = np.mean(linear) 
    global_mean_quadratic = np.mean(quadratic)
    
    # Calculate GDP-weighted global means
    total_gdp = calculate_global_mean(gdp_target, lat)
    gdp_weighted_constant = calculate_global_mean(gdp_target * (1 + constant), lat) / total_gdp - 1
    gdp_weighted_linear = calculate_global_mean(gdp_target * (1 + linear), lat) / total_gdp - 1
    gdp_weighted_quadratic = calculate_global_mean(gdp_target * (1 + quadratic), lat) / total_gdp - 1
    
    # Create red-white-blue colormap centered at zero
    colors = ['red', 'white', 'blue']
    n_bins = 256
    cmap = mcolors.LinearSegmentedColormap.from_list('rwb', colors, N=n_bins)
    
    # Create multi-page PDF
    from matplotlib.backends.backend_pdf import PdfPages
    
    with PdfPages(pdf_path) as pdf:
        # Page 1: Maps
        fig, axes = plt.subplots(3, 1, figsize=(12, 16))
        fig.suptitle(f'Target GDP Reductions\nActual Range: {actual_min:.3f} to {actual_max:.3f}', 
                    fontsize=16, fontweight='bold')
    
        # Plot constant reduction
        im1 = axes[0].pcolormesh(lon_grid, lat_grid, constant, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
        const_min, const_max = constant.min(), constant.max()
        axes[0].set_title(f'Constant Reduction\nRange: {const_min:.4f} to {const_max:.4f}, GDP-weighted: {gdp_weighted_constant:.4f}', 
                         fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Longitude')
        axes[0].set_ylabel('Latitude')
        axes[0].grid(True, alpha=0.3)
        cbar1 = plt.colorbar(im1, ax=axes[0], shrink=0.8)
        cbar1.set_label('Fractional GDP Reduction\n(Scale: -1 to +1)', rotation=270, labelpad=25)
        
        # Plot linear reduction
        im2 = axes[1].pcolormesh(lon_grid, lat_grid, linear, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
        linear_min, linear_max = linear.min(), linear.max()
        axes[1].set_title(f'Linear Reduction (Temperature-Dependent)\nRange: {linear_min:.4f} to {linear_max:.4f}, GDP-weighted: {gdp_weighted_linear:.4f}', 
                         fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Longitude')
        axes[1].set_ylabel('Latitude')
        axes[1].grid(True, alpha=0.3)
        cbar2 = plt.colorbar(im2, ax=axes[1], shrink=0.8)
        cbar2.set_label('Fractional GDP Reduction\n(Scale: -1 to +1)', rotation=270, labelpad=25)
        
        # Plot quadratic reduction
        im3 = axes[2].pcolormesh(lon_grid, lat_grid, quadratic, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
        quad_min, quad_max = quadratic.min(), quadratic.max()
        axes[2].set_title(f'Quadratic Reduction (Temperature-Dependent)\nRange: {quad_min:.4f} to {quad_max:.4f}, GDP-weighted: {gdp_weighted_quadratic:.4f}', 
                         fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Longitude')
        axes[2].set_ylabel('Latitude')
        axes[2].grid(True, alpha=0.3)
        cbar3 = plt.colorbar(im3, ax=axes[2], shrink=0.8)
        cbar3.set_label('Fractional GDP Reduction\n(Scale: -1 to +1)', rotation=270, labelpad=25)
        
        # Adjust layout and save page 1
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Page 2: Line plot of functions vs temperature
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Temperature range for plotting
        temp_range = np.linspace(-15, 40, 1000)
        
        # Extract coefficients from results for function evaluation
        global_stats = results['global_stats']
        a0, a1 = global_stats['linear_coeffs']['a0'], global_stats['linear_coeffs']['a1']
        a_quad, b_quad, c_quad = global_stats['quadratic_coeffs']['a'], global_stats['quadratic_coeffs']['b'], global_stats['quadratic_coeffs']['c']
        
        # Get constant value from config
        constant_value = config['constant_target']['gdp_amount']
        
        # Calculate function values
        constant_values = np.full_like(temp_range, constant_value)
        linear_values = a0 + a1 * temp_range
        quadratic_values = a_quad + b_quad * temp_range + c_quad * temp_range**2
        
        # Plot the three functions
        ax.plot(temp_range, constant_values, 'k-', linewidth=2, label='Constant', alpha=0.8)
        ax.plot(temp_range, linear_values, 'r-', linewidth=2, label='Linear', alpha=0.8)
        ax.plot(temp_range, quadratic_values, 'b-', linewidth=2, label='Quadratic', alpha=0.8)
        
        # Add reference points
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=30, color='gray', linestyle='--', alpha=0.5, label='30°C reference')
        ax.axvline(x=13.5, color='orange', linestyle='--', alpha=0.5, label='13.5°C zero point')
        
        # Mark key points from config
        linear_ref_temp = config['linear_target']['reference_temperature']
        linear_ref_value = config['linear_target']['amount_at_reference_temp']
        quad_ref_temp = config['quadratic_target']['reference_temperature']
        quad_ref_value = config['quadratic_target']['amount_at_reference_temp']
        quad_zero_temp = config['quadratic_target']['zero_amount_temperature']
        
        ax.plot(linear_ref_temp, linear_ref_value, 'ro', markersize=8, 
                label=f'Linear: {linear_ref_temp}°C = {linear_ref_value}')
        ax.plot(quad_ref_temp, quad_ref_value, 'bo', markersize=8, 
                label=f'Quadratic: {quad_ref_temp}°C = {quad_ref_value}')
        ax.plot(quad_zero_temp, 0, 'bs', markersize=8, 
                label=f'Quadratic: {quad_zero_temp}°C = 0')
        
        # Format plot
        ax.set_xlabel('Temperature (°C)', fontsize=14)
        ax.set_ylabel('Fractional GDP Reduction', fontsize=14)
        ax.set_title('Target GDP Reduction Functions vs Temperature', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)
        
        # Set axis limits
        ax.set_xlim(-10, 35)
        
        # Set y-axis limits to show the range nicely
        all_values = np.concatenate([constant_values, linear_values, quadratic_values])
        y_min, y_max = all_values.min(), all_values.max()
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)
        
        # Add equations as text annotations
        equation_text = (
            f"Constant: reduction = {constant_value:.3f}\n"
            f"Linear: reduction = {a0:.6f} + {a1:.6f} × T\n"
            f"Quadratic: reduction = {a_quad:.6f} + {b_quad:.6f} × T + {c_quad:.9f} × T²"
        )
        
        # Position the text box in the upper left corner
        ax.text(0.02, 0.98, equation_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    print(f"Global maps saved to {pdf_path}")
    
    return pdf_path

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python calculate_target_gdp_amounts.py config.json")
        sys.exit(1)
        
    config_file = sys.argv[1]
    
    # Extract wildcard from config filename
    import os
    config_basename = os.path.basename(config_file)  # Remove path
    if config_basename.startswith('target_gdp_config_') and config_basename.endswith('.json'):
        wildcard = config_basename[18:-5]  # Remove 'target_gdp_config_' and '.json'
    else:
        wildcard = "default"
    
    # Load config and calculate target reductions
    config = load_config(config_file)
    results = calculate_target_reductions(config_file)
    
    # Extract model_name and case_name for output filenames
    model_name = config.get('model_name', 'CanESM5')
    case_name = config.get('case_name', 'ssp585')
    
    # Save results to NetCDF with model_name, case_name, and wildcard in filename
    netcdf_filename = f"target_gdp_amounts_{model_name}_{case_name}_{wildcard}.nc"
    save_results_netcdf(results, netcdf_filename)
    
    # Create global maps and save to PDF with model_name, case_name, and wildcard in filename
    pdf_filename = f"target_gdp_amounts_maps_{model_name}_{case_name}_{wildcard}.pdf"
    create_global_maps(results, config, pdf_filename)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Constant reduction range: {results['constant'].min():.4f} to {results['constant'].max():.4f}")
    print(f"Linear reduction range: {results['linear'].min():.4f} to {results['linear'].max():.4f}")
    print(f"Quadratic reduction range: {results['quadratic'].min():.4f} to {results['quadratic'].max():.4f}")