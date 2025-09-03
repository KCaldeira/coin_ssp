# 1) install once (shell)
# pip install pyam-iamc

import pyam
import pandas as pd
import os
import pickle

# (Optional) see what public platforms are available
print(pyam.iiasa.platforms())

# 2) read country-level Population and GDP (PPP + MER) for all SSPs
cache_file = "ssp_raw_data.pkl"

if os.path.exists(cache_file):
    print(f"Loading cached data from {cache_file}...")
    with open(cache_file, 'rb') as f:
        df = pickle.load(f)
    print("Cached data loaded successfully!")
else:
    print("Downloading SSP data from IIASA database...")
    df = pyam.read_iiasa(
        "ssp",
        # meta=True returns scenario metadata too (handy for discovery)
        meta=True,
    )
    
    # Save raw data to cache
    print(f"Saving raw data to {cache_file} for future use...")
    with open(cache_file, 'wb') as f:
        pickle.dump(df, f)
    print("Raw data cached successfully!")

# Filter to available variables
available_vars = [v for v in ["Population", "GDP|PPP"] if v in df.variable]
print(f"Using variables: {available_vars}")
df = df.filter(variable=available_vars)

# Use country names (not ISO3) - exclude regional aggregates
all_regions = set(df.region)
regional_keywords = ['R10', 'R11', 'R12', 'World', 'OECD', 'non-OECD', 'Asia', 'Africa', 'Europe', 'America', 'Middle East', 'Oceania']
country_regions = [r for r in all_regions if not any(keyword in r for keyword in regional_keywords)]

print(f"Found {len(country_regions)} countries (excluding {len(all_regions)-len(country_regions)} regional aggregates)")
print(f"Sample countries: {sorted(country_regions)[:10]}")

# Filter to countries only
df_c = df.filter(region=country_regions)

# Filter to main SSP scenarios (exclude Historical Reference)
want_ssps = ["SSP1", "SSP2", "SSP3", "SSP4", "SSP5"]
df_c = df_c.filter(scenario=want_ssps)

print(f"Final dataset: {len(df_c.data)} rows")

# 5) Split into tidy CSVs by variable
for var in ["Population", "GDP|PPP", "GDP|MER"]:
    out = df_c.filter(variable=var).as_pandas()  # columns: model, scenario, region, variable, unit, year, value
    out.to_csv(f"ssp_{var.replace('|','_')}_country.csv", index=False)

# 6) If you want a wide table (years as columns), pivot per variable:
pop_wide = (
    df_c.filter(variable="Population")
        .as_pandas()
        .pivot_table(index=["model","scenario","region","unit"],
                     columns="year", values="value")
        .reset_index()
)
pop_wide.to_csv("ssp_population_country_wide.csv", index=False)

# Similarly for GDP:
gdp_ppp_wide = (
    df_c.filter(variable="GDP|PPP").as_pandas()
        .pivot_table(index=["model","scenario","region","unit"],
                     columns="year", values="value")
        .reset_index()
)
gdp_ppp_wide.to_csv("ssp_gdp_ppp_country_wide.csv", index=False)
