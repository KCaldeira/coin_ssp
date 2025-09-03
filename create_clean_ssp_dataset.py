import pickle
import pandas as pd

# Load the cached data
with open('ssp_raw_data.pkl', 'rb') as f:
    df = pickle.load(f)

# Convert to pandas for easier filtering
pandas_df = df.as_pandas()

# Filter to Population and GDP|PPP variables only
target_variables = ['Population', 'GDP|PPP']
df_filtered = pandas_df[pandas_df['variable'].isin(target_variables)]

# Identify countries (exclude regional aggregates)
all_regions = df_filtered['region'].unique()
regional_keywords = ['R5', 'R9', 'R10', 'R11', 'R12', 'World', 'OECD', 'non-OECD', 
                     'Asia', 'Africa', 'Europe', 'America', 'Middle East', 'Oceania',
                     'Latin America', 'North America', 'Pacific OECD', 'European Union',
                     'Reforming Economies', 'Other']

# Filter out regional aggregates
countries = [r for r in all_regions if not any(keyword in r for keyword in regional_keywords)]

print(f"Total regions: {len(all_regions)}")
print(f"Countries identified: {len(countries)}")
print(f"Regional aggregates excluded: {len(all_regions) - len(countries)}")

# Filter to countries only
df_countries = df_filtered[df_filtered['region'].isin(countries)]

# Keep only essential columns
essential_columns = ['model', 'scenario', 'region', 'variable', 'year', 'value']
df_clean = df_countries[essential_columns].copy()

# Sort for better organization
df_clean = df_clean.sort_values(['scenario', 'region', 'variable', 'year'])

print(f"\nFinal dataset:")
print(f"Rows: {len(df_clean):,}")
print(f"Variables: {sorted(df_clean['variable'].unique())}")
print(f"Scenarios: {sorted(df_clean['scenario'].unique())}")
print(f"Years: {df_clean['year'].min()} to {df_clean['year'].max()}")
print(f"Countries: {len(df_clean['region'].unique())}")

# Save the clean dataset
df_clean.to_csv('ssp_clean_population_gdp.csv', index=False)
print(f"\nSaved clean dataset to: ssp_clean_population_gdp.csv")

# Show sample data
print(f"\nSample data:")
print(df_clean.head(10))