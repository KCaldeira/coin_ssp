import pickle

# Load the cached data
with open('ssp_raw_data.pkl', 'rb') as f:
    df = pickle.load(f)

# Convert to pandas for easier analysis
pandas_df = df.as_pandas()

print("=== SSP DATASET SUMMARY ===\n")

print(f"MODELS ({len(pandas_df['model'].unique())}):")
for model in sorted(pandas_df['model'].unique()):
    print(f"  {model}")

print(f"\nSCENARIOS ({len(pandas_df['scenario'].unique())}):")
for scenario in sorted(pandas_df['scenario'].unique()):
    print(f"  {scenario}")

print(f"\nREGIONS ({len(pandas_df['region'].unique())}):")
for region in sorted(pandas_df['region'].unique()):
    print(f"  {region}")

print(f"\nVARIABLES ({len(pandas_df['variable'].unique())}):")
for variable in sorted(pandas_df['variable'].unique()):
    print(f"  {variable}")

years = sorted(pandas_df['year'].unique())
print(f"\nYEARS ({len(years)}):")
print(f"  Range: {min(years)} to {max(years)}")
print(f"  All years: {years}")

print(f"\nTOTAL ROWS: {len(pandas_df):,}")