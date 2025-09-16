import pandas as pd
import numpy as np

def create_annual_historical_ssp_datasets():
    """
    Create 5 merged datasets with annual resolution by interpolating economic data.
    Climate data is annual, economic data is 5-yearly - interpolate to match climate.
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
    
    for old_name, new_name in country_mappings.items():
        mask = climate_combined.region == old_name
        if mask.any():
            climate_combined.loc[mask, 'region'] = new_name
            print(f"  Mapped: {old_name} -> {new_name} ({mask.sum()} rows)")
    
    # Get updated country lists
    climate_countries = set(climate_combined.region.unique())
    economic_countries = set(economic.region.unique())
    common_countries = climate_countries & economic_countries
    
    print(f"\nAfter mapping:")
    print(f"Climate countries: {len(climate_countries)}")
    print(f"Economic countries: {len(economic_countries)}")
    print(f"Common countries: {len(common_countries)}")
    
    # Prepare climate data - this gives us our annual time grid
    climate_annual = climate_combined[
        climate_combined.region.isin(common_countries)
    ][['region', 'year', 'tas', 'pr']].rename(columns={'region': 'country'})
    
    print(f"Annual climate data: {len(climate_annual)} rows")
    print(f"Climate years: {climate_annual.year.min()}-{climate_annual.year.max()}")
    
    # Process each SSP scenario
    ssp_scenarios = ['SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5']
    
    for scenario in ssp_scenarios:
        print(f"\n=== Processing Historical/{scenario} ===")
        
        # Get economic data for Historical Reference (1950-2019) + SSP scenario (2020-2100)
        hist_econ = economic[economic.scenario == 'Historical Reference']
        ssp_econ = economic[economic.scenario == scenario]
        
        # Combine historical + SSP economic data
        combined_econ = pd.concat([
            hist_econ[hist_econ.year < 2020],
            ssp_econ[ssp_econ.year >= 2020]
        ], ignore_index=True)
        
        print(f"Combined economic data: {len(combined_econ)} rows")
        
        # Find countries with complete economic data for both periods
        hist_countries = set(hist_econ[hist_econ.year < 2020].region.unique())
        ssp_countries = set(ssp_econ[ssp_econ.year >= 2020].region.unique())
        econ_countries_complete = hist_countries & ssp_countries & common_countries
        
        print(f"Countries with complete data: {len(econ_countries_complete)}")
        
        # Process each country to create annual economic time series
        annual_results = []
        
        for country in sorted(econ_countries_complete):
            # Get climate data for this country (annual)
            country_climate = climate_annual[climate_annual.country == country].sort_values('year')
            if len(country_climate) == 0:
                continue
                
            # Get economic data for this country (5-yearly)
            country_econ = combined_econ[
                (combined_econ.region == country) & 
                (combined_econ.region.isin(econ_countries_complete))
            ]
            
            if len(country_econ) == 0:
                continue
                
            # Pivot economic data to get GDP and Population as columns
            econ_pivot = country_econ.pivot_table(
                index='year',
                columns='variable', 
                values='value',
                aggfunc='first'
            ).reset_index()
            
            if 'GDP|PPP' not in econ_pivot.columns or 'Population' not in econ_pivot.columns:
                continue
                
            # Remove any rows with missing data
            econ_pivot = econ_pivot.dropna(subset=['GDP|PPP', 'Population'])
            
            if len(econ_pivot) < 2:  # Need at least 2 points for interpolation
                continue
                
            # Create annual time series by interpolating economic data
            climate_years = country_climate.year.values
            econ_years = econ_pivot.year.values
            
            # Only interpolate within the range of available economic data
            valid_climate_years = climate_years[
                (climate_years >= econ_years.min()) & 
                (climate_years <= econ_years.max())
            ]
            
            if len(valid_climate_years) == 0:
                continue
            
            # Interpolate GDP and Population to annual resolution
            gdp_annual = np.interp(valid_climate_years, econ_years, econ_pivot['GDP|PPP'].values)
            pop_annual = np.interp(valid_climate_years, econ_years, econ_pivot['Population'].values)
            
            # Get corresponding climate data
            country_climate_valid = country_climate[
                country_climate.year.isin(valid_climate_years)
            ].sort_values('year')
            
            # Ensure we have matching lengths
            if len(country_climate_valid) != len(valid_climate_years):
                continue
                
            # Create annual dataset for this country
            country_annual = pd.DataFrame({
                'country': country,
                'year': valid_climate_years,
                'population': pop_annual,
                'GDP': gdp_annual,
                'tas': country_climate_valid.tas.values,
                'pr': country_climate_valid.pr.values
            })
            
            annual_results.append(country_annual)
        
        if not annual_results:
            print(f"  WARNING: No countries processed successfully for {scenario}")
            continue
            
        # Combine all countries
        scenario_dataset = pd.concat(annual_results, ignore_index=True)
        scenario_dataset = scenario_dataset.sort_values(['country', 'year'])
        
        print(f"  Final annual dataset: {len(scenario_dataset)} rows")
        print(f"  Countries: {len(scenario_dataset.country.unique())}")
        print(f"  Years: {scenario_dataset.year.min()}-{scenario_dataset.year.max()}")
        
        # Save to file
        output_file = f'./data/input/Historical_{scenario}_annual.csv'
        scenario_dataset.to_csv(output_file, index=False)
        print(f"  Saved: {output_file}")
        
        # Show interpolation example
        sample_country = scenario_dataset.country.iloc[0]
        sample_data = scenario_dataset[scenario_dataset.country == sample_country]
        
        print(f"  Sample interpolated data for {sample_country}:")
        print(f"    Years: {sample_data.year.min()}-{sample_data.year.max()} ({len(sample_data)} annual points)")
        print(f"    GDP range: {sample_data.GDP.min():.1f} - {sample_data.GDP.max():.1f}")
        print(f"    Population range: {sample_data.population.min():.1f} - {sample_data.population.max():.1f}")
        
        # Show a few sample years
        sample_years = sample_data[sample_data.year.isin([1990, 2000, 2010, 2020, 2030])][['year', 'population', 'GDP']]
        if len(sample_years) > 0:
            print("    Sample years:")
            for _, row in sample_years.iterrows():
                print(f"      {int(row.year)}: Pop={row.population:.1f}, GDP={row.GDP:.1f}")

if __name__ == "__main__":
    create_annual_historical_ssp_datasets()