import pandas as pd
import numpy as np

def create_historical_ssp_datasets():
    """
    Create 5 merged datasets combining Historical climate data with SSP economic scenarios.
    Output format: country, year, population, GDP, tas, pr
    Each scenario includes all countries that have complete data for that specific scenario.
    """
    
    # Load all three datasets
    print("Loading datasets...")
    hist_climate = pd.read_csv('./data/input/Data_regression_historical.csv')
    ssp_climate = pd.read_csv('./data/input/Data_regression_ssp585.csv') 
    economic = pd.read_csv('./data/input/ssp_clean_population_gdp.csv')
    
    print(f"Historical climate: {len(hist_climate)} rows")
    print(f"SSP585 climate: {len(ssp_climate)} rows") 
    print(f"Economic data: {len(economic)} rows")
    
    # Combine climate data
    climate_combined = pd.concat([hist_climate, ssp_climate], ignore_index=True)
    print(f"Combined climate: {len(climate_combined)} rows, years {climate_combined.year.min()}-{climate_combined.year.max()}")
    
    # Get unique countries from each dataset
    climate_countries = set(climate_combined.region.unique())
    economic_countries = set(economic.region.unique())
    
    print(f"Climate countries: {len(climate_countries)}")
    print(f"Economic countries: {len(economic_countries)}")
    
    # Find countries present in both datasets
    common_countries = climate_countries & economic_countries
    print(f"Common countries: {len(common_countries)}")
    
    # Get year ranges
    climate_years = set(climate_combined.year.unique())
    economic_years = set(economic.year.unique())
    common_years = climate_years & economic_years
    
    print(f"Climate years: {len(climate_years)} ({min(climate_years)}-{max(climate_years)})")
    print(f"Economic years: {len(economic_years)} ({min(economic_years)}-{max(economic_years)})")
    print(f"Common years: {len(common_years)} ({min(common_years)}-{max(common_years)})")
    
    # Filter climate data to common countries and years
    climate_filtered = climate_combined[
        (climate_combined.region.isin(common_countries)) & 
        (climate_combined.year.isin(common_years))
    ].copy()
    
    # Check for duplicates in climate data
    climate_dups = climate_filtered.groupby(['region', 'year']).size()
    if (climate_dups > 1).any():
        duplicates = climate_dups[climate_dups > 1]
        raise ValueError(f"Duplicate climate data found: {duplicates.head()}")
    
    print(f"Filtered climate data: {len(climate_filtered)} rows")
    
    # Pivot economic data to get GDP and Population as columns
    econ_pivot = economic[
        (economic.region.isin(common_countries)) & 
        (economic.year.isin(common_years))
    ].pivot_table(
        index=['scenario', 'region', 'year'],
        columns='variable', 
        values='value',
        aggfunc='first'  # Will fail if there are multiple values
    ).reset_index()
    
    # Check that we have both GDP|PPP and Population
    if 'GDP|PPP' not in econ_pivot.columns or 'Population' not in econ_pivot.columns:
        raise ValueError(f"Missing required variables. Available: {econ_pivot.columns.tolist()}")
    
    # Rename columns for clarity
    econ_pivot.columns.name = None
    econ_pivot = econ_pivot.rename(columns={
        'GDP|PPP': 'GDP',
        'Population': 'population',
        'region': 'country'
    })
    
    print(f"Pivoted economic data: {len(econ_pivot)} rows")
    print(f"Economic scenarios: {sorted(econ_pivot.scenario.unique())}")
    
    # Prepare climate data for merging
    climate_merge = climate_filtered[['region', 'year', 'tas', 'pr']].rename(columns={'region': 'country'})
    
    # Create merged datasets for each SSP scenario (1950-2100)
    ssp_scenarios = ['SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5']
    
    for scenario in ssp_scenarios:
        print(f"\nProcessing Historical/{scenario}...")
        
        # Combine Historical Reference (1950-2019) + SSP scenario (2020-2100)
        historical_econ = econ_pivot[econ_pivot.scenario == 'Historical Reference'][['country', 'year', 'population', 'GDP']]
        ssp_econ = econ_pivot[econ_pivot.scenario == scenario][['country', 'year', 'population', 'GDP']]
        
        # Filter to years: Historical Reference for 1950-2019, SSP for 2020-2100
        historical_years = historical_econ[historical_econ.year < 2020]
        ssp_years = ssp_econ[ssp_econ.year >= 2020]
        
        # Combine economic time series
        combined_econ = pd.concat([historical_years, ssp_years], ignore_index=True)
        
        # Remove any rows with missing GDP or Population data
        econ_complete = combined_econ.dropna(subset=['GDP', 'population'])
        
        # Find countries that have complete data for both time periods
        hist_countries = set(historical_years.dropna(subset=['GDP', 'population']).country.unique())
        ssp_countries = set(ssp_years.dropna(subset=['GDP', 'population']).country.unique())
        econ_countries = hist_countries & ssp_countries  # Must have both periods
        
        # Also ensure these countries have climate data
        climate_countries = set(climate_merge.country.unique())
        valid_countries = econ_countries & climate_countries
        
        print(f"  Historical economic: {len(hist_countries)} countries")
        print(f"  {scenario} economic: {len(ssp_countries)} countries")
        print(f"  Both periods: {len(econ_countries)} countries")
        print(f"  With climate data: {len(valid_countries)} countries")
        
        # Filter datasets to valid countries
        econ_filtered = econ_complete[econ_complete.country.isin(valid_countries)]
        climate_filtered = climate_merge[climate_merge.country.isin(valid_countries)]
        
        # Merge with climate data
        merged = pd.merge(
            econ_filtered,
            climate_filtered,
            on=['country', 'year'],
            how='inner'  # Only keep rows where both datasets have data
        )
        
        # Final check for missing data (should be none now)
        missing_data = merged.isnull().sum()
        if missing_data.any():
            raise ValueError(f"Unexpected missing data in Historical/{scenario}: {missing_data[missing_data > 0]}")
        
        # Sort by country and year
        merged = merged.sort_values(['country', 'year'])
        
        print(f"  Final dataset: {len(merged)} rows")
        print(f"  Countries: {len(merged.country.unique())}")
        print(f"  Years: {merged.year.min()}-{merged.year.max()}")
        
        # Save to file
        output_file = f'./data/input/Historical_{scenario}.csv'
        merged.to_csv(output_file, index=False)
        print(f"  Saved: {output_file}")
        
        # Show sample data spanning the transition
        sample_country = merged.country.iloc[0]
        sample_data = pd.concat([
            merged[(merged.country == sample_country) & (merged.year.isin([2015, 2020]))],
            merged[(merged.country == sample_country) & (merged.year == 2025)]
        ])
        print(f"  Sample data for {sample_country} (around 2020 transition):")
        print(sample_data.to_string(index=False))

if __name__ == "__main__":
    create_historical_ssp_datasets()