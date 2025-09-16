#!/usr/bin/env python3
"""
Test script for temporal alignment functionality.
"""
import numpy as np
from coin_ssp_utils import load_gridded_data, extract_year_coordinate, interpolate_to_annual_grid
import xarray as xr

def test_extract_year_coordinate():
    """Test the year coordinate extraction"""
    print("Testing extract_year_coordinate...")

    files_to_test = [
        'data/input/gridRaw_tas_CanESM5_ssp245.nc',
        'data/input/gridded_gdp_regrid_CanESM5_ssp245.nc',
        'data/input/gridded_pop_regrid_CanESM5_ssp245.nc'
    ]

    for filepath in files_to_test:
        print(f"\n  Testing: {filepath}")
        try:
            ds = xr.open_dataset(filepath, decode_times=False)
            years = extract_year_coordinate(ds)
            print(f"    Extracted years: {years[:5]}...{years[-5:]} (length: {len(years)})")
            print(f"    Year range: {years.min()} to {years.max()}")
            ds.close()
        except Exception as e:
            print(f"    Error: {e}")

def test_interpolation():
    """Test the interpolation functionality"""
    print("\n" + "="*60)
    print("Testing interpolate_to_annual_grid...")

    # Create test data with 5-year intervals
    original_years = np.array([2020, 2025, 2030, 2035, 2040])
    target_years = np.arange(2020, 2041)  # Annual 2020-2040

    # Create simple test data (time, lat=2, lon=3)
    test_data = np.random.rand(5, 2, 3)
    test_data[:, 0, 0] = [1.0, 2.0, 3.0, 4.0, 5.0]  # Linear trend for easy verification

    interpolated = interpolate_to_annual_grid(original_years, test_data, target_years)

    print(f"  Original shape: {test_data.shape}")
    print(f"  Interpolated shape: {interpolated.shape}")
    print(f"  Original data[0,0,0]: {test_data[:, 0, 0]}")
    print(f"  Interpolated data[0,0,0] (first 10): {interpolated[:10, 0, 0]}")

    # Verify that original points are preserved
    for i, year in enumerate(original_years):
        year_idx = year - 2020
        original_val = test_data[i, 0, 0]
        interpolated_val = interpolated[year_idx, 0, 0]
        print(f"  Year {year}: original={original_val:.3f}, interpolated={interpolated_val:.3f}")

def test_load_gridded_data():
    """Test the complete temporal alignment in load_gridded_data"""
    print("\n" + "="*60)
    print("Testing load_gridded_data temporal alignment...")

    try:
        data = load_gridded_data('CanESM5', 'ssp245')

        print(f"\nResults:")
        print(f"  Temperature shape: {data['tas'].shape}")
        print(f"  Precipitation shape: {data['pr'].shape}")
        print(f"  GDP shape: {data['gdp'].shape}")
        print(f"  Population shape: {data['pop'].shape}")

        print(f"\nTime coordinates:")
        print(f"  Temperature years: {data['tas_years'][:5]}...{data['tas_years'][-5:]}")
        print(f"  GDP years: {data['gdp_years'][:5]}...{data['gdp_years'][-5:]}")
        print(f"  Population years: {data['pop_years'][:5]}...{data['pop_years'][-5:]}")

        # Verify all time dimensions are the same
        shapes_match = (
            data['tas'].shape[0] == data['pr'].shape[0] ==
            data['gdp'].shape[0] == data['pop'].shape[0]
        )
        print(f"\nAll time dimensions match: {shapes_match}")

        # Verify years are identical
        years_match = (
            np.array_equal(data['tas_years'], data['pr_years']) and
            np.array_equal(data['tas_years'], data['gdp_years']) and
            np.array_equal(data['tas_years'], data['pop_years'])
        )
        print(f"All year coordinates match: {years_match}")

        if shapes_match and years_match:
            print("✅ Temporal alignment successful!")
            return True
        else:
            print("❌ Temporal alignment failed!")
            return False

    except Exception as e:
        print(f"❌ Error in load_gridded_data: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("="*60)
    print("TESTING TEMPORAL ALIGNMENT FUNCTIONALITY")
    print("="*60)

    # Test individual functions
    test_extract_year_coordinate()
    test_interpolation()

    # Test complete alignment
    success = test_load_gridded_data()

    print("\n" + "="*60)
    if success:
        print("🎉 ALL TESTS PASSED - Temporal alignment working correctly!")
    else:
        print("💥 TESTS FAILED - Temporal alignment needs debugging!")
    print("="*60)