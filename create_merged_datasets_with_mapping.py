import pandas as pd
import numpy as np

def create_historical_ssp_datasets_with_mapping():
    """
    Create 5 merged datasets with country name mapping to maximize matches.
    """
    
    # Country name mappings: climate_name -> economic_name
    country_mappings = {
        'Bosnia and Herz.': 'Bosnia and Herzegovina',
        'Dem. Rep. Congo': 'Democratic Republic of the Congo',
        'Dominican Rep.': 'Dominican Republic',
        'Eq. Guinea': 'Equatorial Guinea',
        'Russia': 'Russian Federation',
        'S. Sudan': 'South Sudan',
        'United States of America': 'United States',
        'Vietnam': 'Viet Nam',
        'eSwatini': 'Eswatini'
    }
    
    # Load all three datasets
    print("Loading datasets...")
    hist_climate = pd.read_csv('./data/input/Data_regression_historical.csv')
    ssp_climate = pd.read_csv('./data/input/Data_regression_ssp585.csv') 
    economic = pd.read_csv('./data/input/ssp_clean_population_gdp.csv')
    
    print(f"Historical climate: {len(hist_climate)} rows")
    print(f"SSP585 climate: {len(ssp_climate)} rows") 
    print(f"Economic data: {len(economic)} rows")
    
    # Apply country mappings to climate data
    print(f"\nApplying {len(country_mappings)} country name mappings...")
    climate_combined = pd.concat([hist_climate, ssp_climate], ignore_index=True)
    
    # Apply mappings
    for old_name, new_name in country_mappings.items():
        mask = climate_combined.region == old_name
        if mask.any():
            climate_combined.loc[mask, 'region'] = new_name
            print(f"  Mapped: {old_name} -> {new_name} ({mask.sum()} rows)")
    
    # Get updated country lists
    climate_countries = set(climate_combined.region.unique())
    economic_countries = set(economic.region.unique())
    
    print(f"\nAfter mapping:")
    print(f"Climate countries: {len(climate_countries)}")
    print(f"Economic countries: {len(economic_countries)}")
    
    # Find countries present in both datasets
    common_countries = climate_countries & economic_countries
    print(f"Common countries: {len(common_countries)} (gained {len(common_countries)} - 139 = {len(common_countries) - 139})")
    
    # Get year ranges
    climate_years = set(climate_combined.year.unique())
    economic_years = set(economic.year.unique())
    common_years = climate_years & economic_years
    
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
        aggfunc='first'
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
        climate_countries_available = set(climate_merge.country.unique())
        valid_countries = econ_countries & climate_countries_available
        
        print(f"  Historical economic: {len(hist_countries)} countries")
        print(f"  {scenario} economic: {len(ssp_countries)} countries")
        print(f"  Both periods: {len(econ_countries)} countries")
        print(f"  With climate data: {len(valid_countries)} countries")
        
        # Filter datasets to valid countries
        econ_filtered = econ_complete[econ_complete.country.isin(valid_countries)]
        climate_filtered_scenario = climate_merge[climate_merge.country.isin(valid_countries)]
        
        # Merge with climate data
        merged = pd.merge(
            econ_filtered,
            climate_filtered_scenario,
            on=['country', 'year'],
            how='inner'
        )
        
        # Final check for missing data
        missing_data = merged.isnull().sum()
        if missing_data.any():
            raise ValueError(f"Unexpected missing data in Historical/{scenario}: {missing_data[missing_data > 0]}")
        
        # Sort by country and year
        merged = merged.sort_values(['country', 'year'])
        
        print(f"  Final dataset: {len(merged)} rows")
        print(f"  Countries: {len(merged.country.unique())}")
        print(f"  Years: {merged.year.min()}-{merged.year.max()}")
        
        # Save to file
        output_file = f'./data/input/Historical_{scenario}_mapped.csv'
        merged.to_csv(output_file, index=False)
        print(f"  Saved: {output_file}")
        
        # Show sample for a newly mapped country if available
        mapped_countries_in_output = [country_mappings[old] for old in country_mappings.keys() 
                                    if country_mappings[old] in merged.country.unique()]
        
        if mapped_countries_in_output:
            sample_country = mapped_countries_in_output[0]
            sample_data = merged[merged.country == sample_country].head(3)
            print(f"  Sample data for newly mapped country {sample_country}:")
            print(sample_data.to_string(index=False))

if __name__ == "__main__":
    create_historical_ssp_datasets_with_mapping()