#!/usr/bin/env python3
"""
COIN-SSP Climate Economics Main Processing Script

This script processes climate-economic data for each country:
1. Loads Historical/SSP merged dataset
2. For each country, extracts time series data (population, GDP, temperature, precipitation)
3. Calculates baseline TFP without climate effects using Solow-Swan model
4. Runs forward model with climate effects to get climate-adjusted economic outcomes

Usage:
    python main.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from coin_ssp_core import ModelParams, calculate_tfp_coin_ssp, calculate_coin_ssp_forward_model

def load_data(data_file="./data/input/Historical_SSP5_mapped.csv"):
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
        country_data['gdp'], 
        country_data['population'], 
        params
    )
    
    # Step 2: Run forward model with climate effects
    print(f"    Running forward model with climate effects...")
    
    # Note: We need to handle the fact that calculate_coin_ssp_forward_model 
    # expects 'pr' but we may not have it in all datasets
    if 'pr' in country_data:
        pr_data = country_data['pr']
    else:
        # If no precipitation data, create zeros
        pr_data = np.zeros_like(country_data['tas'])
        
    y_climate, a_climate, k_climate, y_climate_factor, tfp_climate_factor, k_climate_factor = calculate_coin_ssp_forward_model(
        tfp_baseline,
        country_data['population'],
        country_data['gdp'], 
        country_data['tas'],
        params
    )
    
    # Package results
    results = {
        'years': country_data['years'],
        'population': country_data['population'],
        'gdp_observed': country_data['gdp'],
        'tas': country_data['tas'],
        'pr': pr_data,
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
        'k_climate_factor': k_climate_factor
    }
    
    return results

def save_country_results(country, results, output_dir="./data/output"):
    """Save results for a single country to CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
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
        'k_climate_factor': results['k_climate_factor']
    })
    
    output_file = output_dir / f"{country.replace(' ', '_')}_results.csv"
    df.to_csv(output_file, index=False)
    print(f"    Saved results: {output_file}")

def main():
    """Main processing function."""
    print("=== COIN-SSP Climate Economics Processing ===\n")
    
    # Load data
    data = load_data()
    
    # Define model parameters
    # Start with baseline parameters (no climate sensitivity for testing)
    params = ModelParams(
        s=0.3,          # savings rate (30%)
        alpha=0.3,      # capital elasticity  
        delta=0.1,      # depreciation rate (10% per year)
        tas0=20.0,      # reference temperature (°C)
        pr0=1.0,        # reference precipitation (mm/day)
        # Climate sensitivity parameters (start with zeros for testing)
        k_tas1=0.0,     # linear temperature sensitivity for capital
        k_tas2=0.0,     # quadratic temperature sensitivity for capital  
        tfp_tas1=0.0,   # linear temperature sensitivity for TFP
        tfp_tas2=0.0,   # quadratic temperature sensitivity for TFP
        y_tas1=0.0,     # linear temperature sensitivity for output
        y_tas2=0.0      # quadratic temperature sensitivity for output
    )
    
    print(f"\nModel Parameters:")
    print(f"  Savings rate: {params.s}")
    print(f"  Capital elasticity: {params.alpha}")  
    print(f"  Depreciation rate: {params.delta}")
    print(f"  Reference temperature: {params.tas0}°C")
    print(f"  Reference precipitation: {params.pr0} mm/day")
    
    # Process each country
    countries = sorted(data.country.unique())
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
            save_country_results(country, results)
            
            # Store in memory for summary analysis
            all_results[country] = results
            
        except Exception as e:
            print(f"    ERROR processing {country}: {e}")
            continue
    
    print(f"\n=== Processing Complete ===")
    print(f"Successfully processed {len(all_results)} countries")
    print(f"Results saved in ./data/output/")
    
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