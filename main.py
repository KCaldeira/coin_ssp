#!/usr/bin/env python3
"""
COIN-SSP Climate Economics Main Processing Script

This script processes climate-economic data for each country using JSON configuration:
1. Loads JSON configuration file with model parameters and scaling workflow
2. Loads Historical/SSP merged dataset
3. For each country, extracts time series data (population, GDP, temperature, precipitation)
4. Calculates baseline TFP without climate effects using Solow-Swan model
5. Runs optimization workflow to calibrate climate sensitivity parameters
6. Runs forward model with climate effects to get climate-adjusted economic outcomes

Usage:
    python main.py <config.json> [--max-countries N]
"""

import pandas as pd
import numpy as np
from pathlib import Path
from coin_ssp_core import ModelParams, ScalingParams, calculate_tfp_coin_ssp, calculate_coin_ssp_forward_model, optimize_climate_response_scaling
from coin_ssp_utils import apply_time_series_filter
import argparse
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
import json
import copy

def load_config(config_file):
    """Load JSON configuration file containing model parameters and workflow."""
    print(f"Loading configuration from {config_file}...")
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Extract model parameters section (optional)
    model_params_config = config.get('model_params', {})
    
    # Extract scaling parameters section (required) - should be a list of parameter sets
    if 'scaling_params' not in config:
        raise ValueError("Configuration file must contain 'scaling_params' section")
    scaling_params_list = config['scaling_params']
    
    if not isinstance(scaling_params_list, list):
        raise ValueError("'scaling_params' section must be a list of parameter sets")
    
    return model_params_config, scaling_params_list

def load_data(data_file="./data/input/Historical_SSP5_annual.csv"):
    """Load the merged climate-economic dataset."""
    print(f"Loading data from {data_file}...")
    data = pd.read_csv(data_file)
    
    print(f"Dataset loaded: {len(data)} rows")
    print(f"Countries: {len(data.country.unique())}")
    print(f"Years: {data.year.min()}-{data.year.max()}")
    print(f"Columns: {list(data.columns)}")
    
    return data

def extract_country_data(data, country):
    """Extract time series for a specific country."""
    country_data = data[data.country == country].sort_values('year').copy()
    
    if len(country_data) == 0:
        raise ValueError(f"No data found for country: {country}")
    
    # Extract vectors
    years = country_data.year.values
    population = country_data.population.values
    gdp = country_data.GDP.values  # Note: column is 'GDP' not 'gdp'
    tas = country_data.tas.values
    pr = country_data.pr.values
    
    print(f"  {country}: {len(years)} years ({years[0]}-{years[-1]})")
    
    return {
        'years': years,
        'population': population, 
        'gdp': gdp,
        'tas': tas,
        'pr': pr
    }


def process_country_with_scaling(country_data, params, scaling_param_sets):
    """
    Process a single country through the complete COIN-SSP pipeline with multiple scaling sets.
    
    Steps:
    1. Calculate baseline TFP without climate effects (once per country)
    2. For each scaling set:
       - Run optimization to find optimal scaling factor
       - Run forward model with optimized climate effects
       - Run forward model with weather-only effects
    
    Returns:
    --------
    dict: Results containing baseline and all scaling set results
    """
    
    # Step 0: Create variables needed for later processing
    if params.year_diverge < country_data['years'][0]:
        year_diverge_loc = 0
    else:
        year_diverge_loc = np.where(country_data['years'] == params.year_diverge)[0][0]
    params.tas0 = np.mean(country_data['tas'][:year_diverge_loc+1])
    params.pr0 = np.mean(country_data['pr'][:year_diverge_loc+1])
    # Pre-calculate weather data and add to country_data (same for all scaling sets within this country)
    filter_width = 30  # years
    country_data['tas_weather'] = apply_time_series_filter(country_data['tas'], filter_width, year_diverge_loc)
    country_data['pr_weather'] = apply_time_series_filter(country_data['pr'], filter_width, year_diverge_loc)
    
    # Step 1: Calculate baseline TFP without climate effects (once per country)
    print(f"    Calculating baseline TFP...")
    country_data['tfp_baseline'], country_data['k_baseline'] = calculate_tfp_coin_ssp(
        country_data['population'], 
        country_data['gdp'], 
        params
    )
    
    # Initialize results structure
    results = {
        'years': country_data['years'],
        'population': country_data['population'],
        'gdp_observed': country_data['gdp'],
        'tas': country_data['tas'],
        'pr': country_data['pr'],
        'tas_weather': country_data['tas_weather'],
        'pr_weather': country_data['pr_weather'],
        'tfp_baseline': country_data['tfp_baseline'],
        'k_baseline': country_data['k_baseline'],
        'scaling_results': {}  # Will contain results for each scaling set
    }
    
    # Step 2: Process each scaling set
    for scaling_params in scaling_param_sets:
        print(f"    Processing scaling set: {scaling_params.scaling_name}")
        
        # Get scaling factor: use provided value or optimize
        if scaling_params.scale_factor is not None:
            print(f"      Using provided scale factor: {scaling_params.scale_factor:.6f}")
            optimal_scale = scaling_params.scale_factor
            final_error = None
        else:
            print(f"      Optimizing climate response scaling...")
            optimal_scale, final_error = optimize_climate_response_scaling(
                country_data, params, scaling_params
            )
            print(f"      Scale factor: {optimal_scale:.6f}, error: {final_error:.6f}")
        
        # Create scaled parameters for this optimization
        params_scaled = copy.deepcopy(params)
        params_scaled.k_tas1 *= optimal_scale * scaling_params.k_tas1
        params_scaled.k_tas2 *= optimal_scale * scaling_params.k_tas2
        params_scaled.k_pr1 *= optimal_scale * scaling_params.k_pr1
        params_scaled.k_pr2 *= optimal_scale * scaling_params.k_pr2
        params_scaled.tfp_tas1 *= optimal_scale * scaling_params.tfp_tas1
        params_scaled.tfp_tas2 *= optimal_scale * scaling_params.tfp_tas2
        params_scaled.tfp_pr1 *= optimal_scale * scaling_params.tfp_pr1
        params_scaled.tfp_pr2 *= optimal_scale * scaling_params.tfp_pr2
        params_scaled.y_tas1 *= optimal_scale * scaling_params.y_tas1
        params_scaled.y_tas2 *= optimal_scale * scaling_params.y_tas2
        params_scaled.y_pr1 *= optimal_scale * scaling_params.y_pr1
        params_scaled.y_pr2 *= optimal_scale * scaling_params.y_pr2
        
        # Run forward model with scaled climate effects
        print(f"      Running forward model with scaled climate effects...")
        y_climate, a_climate, k_climate, y_climate_factor, tfp_climate_factor, k_climate_factor = calculate_coin_ssp_forward_model(
            country_data['tfp_baseline'],
            country_data['population'],
            country_data['tas'],
            country_data['pr'],
            params_scaled
        )
        
        # Run forward model with weather only (no climate trends after year_diverge)
        print(f"      Running forward model with weather effects only...")
        
        y_weather, a_weather, k_weather, y_weather_factor, tfp_weather_factor, k_weather_factor = calculate_coin_ssp_forward_model(
            country_data['tfp_baseline'],
            country_data['population'],
            country_data['tas_weather'],
            country_data['pr_weather'],
            params_scaled
        )
        
        # Store results for this scaling set
        scaling_result = {
            'optimal_scale': optimal_scale,
            'final_error': final_error if final_error is not None else 0.0,
            'gdp_climate': y_climate * country_data['gdp'][0],
            'tfp_climate': a_climate,
            'k_climate': k_climate,
            'y_climate_factor': y_climate_factor,
            'tfp_climate_factor': tfp_climate_factor,
            'k_climate_factor': k_climate_factor,
            'gdp_weather': y_weather * country_data['gdp'][0],
            'tfp_weather': a_weather,
            'k_weather': k_weather,
            'y_weather_factor': y_weather_factor,
            'tfp_weather_factor': tfp_weather_factor,
            'k_weather_factor': k_weather_factor
        }
        
        results['scaling_results'][scaling_params.scaling_name] = scaling_result
    
    return results

def save_country_results(country, results, output_dir, run_name, timestamp):
    """Save results for a single country to CSV with all scaling sets."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Start with common columns
    df_data = {
        'year': results['years'],
        'population': results['population'],
        'gdp_observed': results['gdp_observed'],
        'tas': results['tas'],
        'pr': results['pr'],
        'tas_weather': results['tas_weather'],
        'pr_weather': results['pr_weather'],
        'tfp_baseline': results['tfp_baseline'],
        'k_baseline': results['k_baseline']
    }
    
    # Add columns for each scaling set
    for scaling_name, scaling_result in results['scaling_results'].items():
        df_data[f'gdp_climate_{scaling_name}'] = scaling_result['gdp_climate']
        df_data[f'tfp_climate_{scaling_name}'] = scaling_result['tfp_climate']
        df_data[f'k_climate_{scaling_name}'] = scaling_result['k_climate']
        df_data[f'gdp_weather_{scaling_name}'] = scaling_result['gdp_weather']
        df_data[f'tfp_weather_{scaling_name}'] = scaling_result['tfp_weather']
        df_data[f'k_weather_{scaling_name}'] = scaling_result['k_weather']
    
    df = pd.DataFrame(df_data)
    
    output_file = output_dir / f"{country.replace(' ', '_')}_results_{run_name}_{timestamp}.csv"
    df.to_csv(output_file, index=False)
    print(f"    Saved results: {output_file}")

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
    
    # Add scaling info box in lower right corner
    scaling_text = (f'Scaling: {scaling_name}\n'
                   f'Optimal scale: {scaling_result["optimal_scale"]:.4f}\n'
                   f'Final error: {scaling_result["final_error"]:.6f}\n'
                   f'Target year: {params.year_scale}\n'
                   f'Target amount: {params.amount_scale:.3f}')
    
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

def parse_args():
    parser = argparse.ArgumentParser(description="COIN-SSP Climate Economics Main Processing Script")
    parser.add_argument("config_file", help="JSON configuration file containing model parameters and workflow")
    parser.add_argument("--max-countries", type=int, default=None, help="Maximum number of countries to process")
    return parser.parse_args()


def main():
    """Main processing function."""
    print("=== COIN-SSP Climate Economics Processing ===\n")
    args = parse_args()
    
    # Extract run name from JSON filename (coin_ssp_*.json -> *)
    config_path = Path(args.config_file)
    if not config_path.name.startswith('coin_ssp_') or not config_path.name.endswith('.json'):
        raise ValueError("Config file must be named coin_ssp_*.json")
    run_name = config_path.stem[9:]  # Remove 'coin_ssp_' prefix
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Run name: {run_name}")
    print(f"Run timestamp: {timestamp}")
    
    # Load configuration from JSON file
    model_params_config, scaling_params_list = load_config(args.config_file)
    
    # Parse command line argument for max countries
    max_countries = args.max_countries
    
    # Load data
    data = load_data()
    
    # Create ModelParams with defaults, overridden by config file
    params = ModelParams(**model_params_config)
    
    # Create list of ScalingParams from config file
    scaling_param_sets = [ScalingParams(**scaling_config) for scaling_config in scaling_params_list]
    
    print(f"\nModel Parameters:")
    print(f"  Savings rate: {params.s}")
    print(f"  Capital elasticity: {params.alpha}")  
    print(f"  Depreciation rate: {params.delta}")
    print(f"  Reference temperature: {params.tas0}°C")
    print(f"  Reference precipitation: {params.pr0} mm/day")
    print(f"  Year diverge: {params.year_diverge}")
    print(f"  Year scale: {params.year_scale}")
    print(f"  Amount scale: {params.amount_scale}")
    
    print(f"\nScaling Parameter Sets: {len(scaling_param_sets)} loaded")
    for i, scaling_params in enumerate(scaling_param_sets):
        print(f"  Set {i+1}: k_tas2={scaling_params.k_tas2}, tfp_tas2={scaling_params.tfp_tas2}, y_tas2={scaling_params.y_tas2}")
        # Show key non-zero parameters for each set
    
    # Create timestamped output directory with run name
    base_output_dir = Path("./data/output")
    output_dir = base_output_dir / f"run_{run_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Process each country
    countries = sorted(data.country.unique())
    if max_countries:
        countries = countries[:max_countries]
    print(f"\nProcessing {len(countries)} countries...")
    
    all_results = {}
    
    for i, country in enumerate(countries, 1):
        print(f"\n[{i}/{len(countries)}] Processing {country}...")
        
        # Extract country data
        country_data = extract_country_data(data, country)
        
        # Process country through COIN-SSP pipeline with all scaling sets
        results = process_country_with_scaling(country_data, params, scaling_param_sets)
        
        # Save results
        save_country_results(country, results, output_dir, run_name, timestamp)
        
        # Store in memory for summary analysis
        all_results[country] = results
    
    print(f"\n=== Processing Complete ===")
    print(f"Successfully processed {len(all_results)} countries")
    print(f"Results saved in {output_dir}")
    
    # Create PDF books for each country  
    if all_results:
        create_country_pdf_books(all_results, params, output_dir, run_name, timestamp)
    
    # Quick summary
    if all_results:
        sample_country = list(all_results.keys())[0]
        sample_results = all_results[sample_country]
        sample_scaling = list(sample_results['scaling_results'].keys())[0]
        sample_scaling_result = sample_results['scaling_results'][sample_scaling]
        
        print(f"\nSample results for {sample_country} (scaling: {sample_scaling}):")
        print(f"  Years: {sample_results['years'][0]}-{sample_results['years'][-1]}")
        print(f"  GDP observed (final): {sample_results['gdp_observed'][-1]:.1f}")
        print(f"  GDP with climate (final): {sample_scaling_result['gdp_climate'][-1]:.1f}")
        print(f"  TFP baseline (final): {sample_results['tfp_baseline'][-1]:.3f}")
        print(f"  TFP with climate (final): {sample_scaling_result['tfp_climate'][-1]:.3f}")
        print(f"  Scale factor: {sample_scaling_result['optimal_scale']:.6f}")

if __name__ == "__main__":
    main()