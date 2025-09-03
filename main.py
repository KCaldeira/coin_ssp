#!/usr/bin/env python3
"""
COIN-SSP Climate Economics Main Processing Script

This script processes climate-economic data for each country:
1. Loads Historical/SSP merged dataset
2. For each country, extracts time series data (population, GDP, temperature, precipitation)
3. Calculates baseline TFP without climate effects using Solow-Swan model
4. Runs forward model with climate effects to get climate-adjusted economic outcomes

Usage:
    python main.py [--max-countries N] [--k_tas2 -0.01] [other climate parameters...]
"""

import pandas as pd
import numpy as np
from pathlib import Path
from coin_ssp_core import ModelParams, calculate_tfp_coin_ssp, calculate_coin_ssp_forward_model
from coin_ssp_utils import apply_time_series_filter
import argparse
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

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

def process_country(country_data, params):
    """
    Process a single country through the complete COIN-SSP pipeline.
    
    Steps:
    1. Calculate baseline TFP without climate effects
    2. Run forward model with climate effects
    
    Returns:
    --------
    dict: Results containing baseline and climate-adjusted time series
    """
    
    # Step 1: Calculate baseline TFP without climate effects
    print(f"    Calculating baseline TFP...")
    tfp_baseline, k_baseline = calculate_tfp_coin_ssp(
        country_data['population'], 
        country_data['gdp'], 
        params
    )
    
    # Step 2: Run forward model with climate effects
    print(f"    Running forward model with climate effects...")
    
    # find location of year 2025 in years
    year_2025_loc = np.where(country_data['years'] == 2025)[0][0]
    # reference temperature and precipitation are set to the mean of the beginning of the dataset to 2025
    params.tas0 = np.mean(country_data['tas'][:year_2025_loc+1])
    params.pr0 = np.mean(country_data['pr'][:year_2025_loc+1])
       
    y_climate, a_climate, k_climate, y_climate_factor, tfp_climate_factor, k_climate_factor = calculate_coin_ssp_forward_model(
        tfp_baseline,
        country_data['population'],
        country_data['gdp'], 
        country_data['tas'],
        country_data['pr'],
        params
    )
    
    
    # Step 3: Run forward model with weather only (no climate trends) after year 2025
    print(f"    Running forward model with no climate effects after year 2025...")

    filter_width = 30  # years

    tas_weather = apply_time_series_filter(country_data['tas'], filter_width, year_2025_loc)
    pr_weather = apply_time_series_filter(country_data['pr'], filter_width, year_2025_loc)
    
       
    y_weather, a_weather, k_weather, y_weather_factor, tfp_weather_factor, k_weather_factor = calculate_coin_ssp_forward_model(
        tfp_baseline,
        country_data['population'],
        country_data['gdp'], 
        tas_weather,
        pr_weather,
        params
    )
    
    # Package results
    results = {
        'years': country_data['years'],
        'population': country_data['population'],
        'gdp_observed': country_data['gdp'],
        'tas': country_data['tas'],
        'pr': country_data['pr'],
        'tas_weather': tas_weather,
        'pr_weather': pr_weather,
        # Baseline results (no climate)
        'tfp_baseline': tfp_baseline,
        'k_baseline': k_baseline,
        # Climate-adjusted results  
        'gdp_climate': y_climate * country_data['gdp'][0],  # Convert back to absolute units
        'tfp_climate': a_climate,
        'k_climate': k_climate,
        # Climate effect factors
        'y_climate_factor': y_climate_factor,
        'tfp_climate_factor': tfp_climate_factor, 
        'k_climate_factor': k_climate_factor,
        # no-climate-adjusted (interannual weather variability only) results  
        'gdp_weather': y_weather * country_data['gdp'][0],  # Convert back to absolute units
        'tfp_weather': a_weather,
        'k_weather': k_weather,
        # Climate effect factors
        'y_weather_factor': y_weather_factor,
        'tfp_weather_factor': tfp_weather_factor, 
        'k_weather_factor': k_weather_factor
    }
    
    return results

def save_country_results(country, results, output_dir, timestamp):
    """Save results for a single country to CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create DataFrame from results
    df = pd.DataFrame({
        'year': results['years'],
        'population': results['population'],
        'gdp_observed': results['gdp_observed'],
        'tas': results['tas'],
        'pr': results['pr'],
        'tfp_baseline': results['tfp_baseline'],
        'k_baseline': results['k_baseline'],
        'gdp_climate': results['gdp_climate'],
        'tfp_climate': results['tfp_climate'], 
        'k_climate': results['k_climate'],
        'y_climate_factor': results['y_climate_factor'],
        'tfp_climate_factor': results['tfp_climate_factor'],
        'k_climate_factor': results['k_climate_factor'],
        'gdp_weather': results['gdp_weather'],
        'tfp_weather': results['tfp_weather'], 
        'k_weather': results['k_weather'],
        'y_weather_factor': results['y_weather_factor'],
        'tfp_weather_factor': results['tfp_weather_factor'],
        'k_weather_factor': results['k_weather_factor']
    })
    
    output_file = output_dir / f"{country.replace(' ', '_')}_results_{timestamp}.csv"
    df.to_csv(output_file, index=False)
    print(f"    Saved results: {output_file}")

def create_country_page(country, results, params, fig):
    """Create a single page with three panels for one country."""
    fig.suptitle(f'{country}', fontsize=16, fontweight='bold')
    
    years = results['years']
    
    # Panel 1: GDP
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(years, results['gdp_observed'], 'k-', label='Baseline', linewidth=2)
    ax1.plot(years, results['gdp_climate'], 'r-', label='Climate', linewidth=1.5)
    ax1.plot(years, results['gdp_weather'], 'b--', label='Weather', linewidth=1.5)
    ax1.set_ylabel('GDP (billion $)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Add parameter box in lower right corner
    param_text = (f'Climate Parameters:\n'
                  f'k_tas1={params.k_tas1:g}, k_tas2={params.k_tas2:g}\n'
                  f'k_pr1={params.k_pr1:g}, k_pr2={params.k_pr2:g}\n'
                  f'tfp_tas1={params.tfp_tas1:g}, tfp_tas2={params.tfp_tas2:g}\n'
                  f'tfp_pr1={params.tfp_pr1:g}, tfp_pr2={params.tfp_pr2:g}\n'
                  f'y_tas1={params.y_tas1:g}, y_tas2={params.y_tas2:g}\n'
                  f'y_pr1={params.y_pr1:g}, y_pr2={params.y_pr2:g}')
    
    ax1.text(0.98, 0.02, param_text, transform=ax1.transAxes, fontsize=8,
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Panel 2: TFP
    ax2 = fig.add_subplot(3, 1, 2)
    ax2.plot(years, results['tfp_baseline'], 'k-', label='Baseline', linewidth=2)
    ax2.plot(years, results['tfp_climate'], 'r-', label='Climate', linewidth=1.5)
    ax2.plot(years, results['tfp_weather'], 'b--', label='Weather', linewidth=1.5)
    ax2.set_ylabel('Total Factor Productivity')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Capital Stock
    ax3 = fig.add_subplot(3, 1, 3)
    ax3.plot(years, results['k_baseline'], 'k-', label='Baseline', linewidth=2)
    ax3.plot(years, results['k_climate'], 'r-', label='Climate', linewidth=1.5)
    ax3.plot(years, results['k_weather'], 'b--', label='Weather', linewidth=1.5)
    ax3.set_ylabel('Capital Stock (normalized)')
    ax3.set_xlabel('Year')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_results_pdf(all_results, params, output_dir, timestamp):
    """Create a PDF book with one page per country showing results."""
    output_dir = Path(output_dir)
    pdf_file = output_dir / f"COIN_SSP_Results_Book_{timestamp}.pdf"
    
    print(f"\nCreating PDF results book...")
    print(f"  Generating {len(all_results)} country pages...")
    
    with PdfPages(pdf_file) as pdf:
        for i, (country, results) in enumerate(sorted(all_results.items()), 1):
            print(f"  [{i}/{len(all_results)}] Adding {country}...")
            
            # Create figure for this country
            fig = plt.figure(figsize=(8.5, 11))  # Letter size portrait
            create_country_page(country, results, params, fig)
            
            # Save to PDF
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    
    print(f"  PDF book saved: {pdf_file}")
    return pdf_file

def parse_args():
    parser = argparse.ArgumentParser(description="COIN-SSP Climate Economics Main Processing Script")
    parser.add_argument("--max-countries", type=int, default=None, help="Maximum number of countries to process")
    parser.add_argument("--k_tas1", type=float, default=0.0, help="Linear temperature sensitivity for capital loss")
    parser.add_argument("--k_tas2", type=float, default=0.0, help="Quadratic temperature sensitivity for capital loss")
    parser.add_argument("--k_pr1", type=float, default=0.0, help="Linear precipitation sensitivity for capital loss")
    parser.add_argument("--k_pr2", type=float, default=0.0, help="Quadratic precipitation sensitivity for capital loss")
    parser.add_argument("--tfp_tas1", type=float, default=0.0, help="Linear temperature sensitivity for TFP loss")
    parser.add_argument("--tfp_tas2", type=float, default=0.0, help="Quadratic temperature sensitivity for TFP loss")
    parser.add_argument("--tfp_pr1", type=float, default=0.0, help="Linear precipitation sensitivity for TFP loss")
    parser.add_argument("--tfp_pr2", type=float, default=0.0, help="Quadratic precipitation sensitivity for TFP loss")
    parser.add_argument("--y_tas1", type=float, default=0.0, help="Linear temperature sensitivity for output loss")
    parser.add_argument("--y_tas2", type=float, default=0.0, help="Quadratic temperature sensitivity for output loss")
    parser.add_argument("--y_pr1", type=float, default=0.0, help="Linear precipitation sensitivity for output loss")
    parser.add_argument("--y_pr2", type=float, default=0.0, help="Quadratic precipitation sensitivity for output loss")
    return parser.parse_args()


def main():
    """Main processing function."""
    print("=== COIN-SSP Climate Economics Processing ===\n")
    args = parse_args()
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Run timestamp: {timestamp}")
    
    # Parse command line argument for max countries
    max_countries = args.max_countries
    
    # Load data
    data = load_data()
    
    # Define model parameters using command-line arguments
    params = ModelParams(
        s=0.3,          # savings rate (30%)
        alpha=0.3,      # capital elasticity  
        delta=0.1,      # depreciation rate (10% per year)
        tas0=20.0,      # reference temperature (°C)
        pr0=1.0,        # reference precipitation (mm/day)
        k_tas1=args.k_tas1,
        k_tas2=args.k_tas2,
        k_pr1=args.k_pr1,
        k_pr2=args.k_pr2,
        tfp_tas1=args.tfp_tas1,
        tfp_tas2=args.tfp_tas2,
        tfp_pr1=args.tfp_pr1,
        tfp_pr2=args.tfp_pr2,
        y_tas1=args.y_tas1,
        y_tas2=args.y_tas2,
        y_pr1=args.y_pr1,
        y_pr2=args.y_pr2
    )
    
    print(f"\nModel Parameters:")
    print(f"  Savings rate: {params.s}")
    print(f"  Capital elasticity: {params.alpha}")  
    print(f"  Depreciation rate: {params.delta}")
    print(f"  Reference temperature: {params.tas0}°C")
    print(f"  Reference precipitation: {params.pr0} mm/day")
    
    # Create timestamped output directory
    base_output_dir = Path("./data/output")
    output_dir = base_output_dir / f"run_{timestamp}"
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
        
        try:
            # Extract country data
            country_data = extract_country_data(data, country)
            
            # Process country through COIN-SSP pipeline
            results = process_country(country_data, params)
            
            # Save results
            save_country_results(country, results, output_dir, timestamp)
            
            # Store in memory for summary analysis
            all_results[country] = results
            
        except Exception as e:
            print(f"    ERROR processing {country}: {e}")
            continue
    
    print(f"\n=== Processing Complete ===")
    print(f"Successfully processed {len(all_results)} countries")
    print(f"Results saved in {output_dir}")
    
    # Create PDF results book
    if all_results:
        create_results_pdf(all_results, params, output_dir, timestamp)
    
    # Quick summary
    if all_results:
        sample_country = list(all_results.keys())[0]
        sample_results = all_results[sample_country]
        print(f"\nSample results for {sample_country}:")
        print(f"  Years: {sample_results['years'][0]}-{sample_results['years'][-1]}")
        print(f"  GDP observed (final): {sample_results['gdp_observed'][-1]:.1f}")
        print(f"  GDP with climate (final): {sample_results['gdp_climate'][-1]:.1f}")
        print(f"  TFP baseline (final): {sample_results['tfp_baseline'][-1]:.3f}")
        print(f"  TFP with climate (final): {sample_results['tfp_climate'][-1]:.3f}")

if __name__ == "__main__":
    main()