import pickle
import pandas as pd

# Load the cached data
with open('./data/input/ssp_raw_data.pkl', 'rb') as f:
    df = pickle.load(f)

# Convert to pandas DataFrame for inspection
pandas_df = df.as_pandas()

# Save full dataset to CSV
pandas_df.to_csv('ssp_full_dataset.csv', index=False)

print(f"Exported {len(pandas_df)} rows to ssp_full_dataset.csv")
print(f"Columns: {list(pandas_df.columns)}")
print("\nFirst few rows:")
print(pandas_df.head())
print("\nUnique variables:")
print(pandas_df['variable'].unique()[:20])
print("\nUnique scenarios:")  
print(pandas_df['scenario'].unique())
print("\nSample regions:")
print(sorted(pandas_df['region'].unique())[:20])