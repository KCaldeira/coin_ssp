# Claude Code Style Guide for COIN_SSP

## Coding Philosophy
This project prioritizes **elegant, fail-fast code** that surfaces errors quickly rather than hiding them.

## Core Style Requirements

### Error Handling
- **No input validation** on function parameters (except for command-line interfaces)
- **No defensive programming** - let exceptions bubble up naturally
- **Fail fast** - prefer code that crashes immediately on invalid inputs rather than continuing with bad data
- **No try-catch blocks** unless absolutely necessary for program logic (not error suppression)
- **No optional function arguments** - all parameters must be explicitly provided
- **Assume complete data** - do not check for missing data fields. If required data is missing, let the code fail with natural Python errors

### Code Elegance
- **Minimize conditional statements** - prefer functional approaches, mathematical expressions, and numpy vectorization
- **Favor mathematical clarity** over defensive checks
- **Use numpy operations** instead of loops and conditionals where possible
- **Prefer concise, readable expressions** over verbose defensive code
- **Compute once, use many times** - move invariant calculations outside loops and create centralized helper functions

### Function Design
- Functions should assume valid inputs and focus on their core mathematical/logical purpose
- Let Python's natural error messages guide debugging rather than custom error handling
- Prioritize algorithmic clarity over robustness checking

## Examples

### Preferred Style
```python
def calculate_growth_rate(values):
    return values[1:] / values[:-1] - 1

def apply_damage_function(output, temperature, sensitivity):
    return output * np.exp(-sensitivity * temperature**2)
```

### Avoid
```python
def calculate_growth_rate(values):
    if not isinstance(values, np.ndarray):
        raise TypeError("values must be numpy array")
    if len(values) < 2:
        raise ValueError("need at least 2 values")
    try:
        return values[1:] / values[:-1] - 1
    except Exception as e:
        logger.error(f"Growth calculation failed: {e}")
        return None
```

## Project Architecture

### Module Organization
- **`coin_ssp_core.py`**: Core economic model functions and parameter classes
- **`coin_ssp_utils.py`**: Consolidated utility functions for mathematical operations, visualization, NetCDF processing
- **`main_integrated.py`**: Complete 5-step processing pipeline
- **`main.py`**: Country-level workflow orchestration

### Key Technical Requirements
- **NetCDF Convention**: Arrays follow `[time, lat, lon]` dimension order - always use `data[time_idx, lat_idx, lon_idx]`
- **Grid Cell Validation**: Skip ocean/ice cells where `gdp_value <= 0` or `population_value <= 0`
- **Configuration Management**: Use `resolve_netcdf_filepath()` - NEVER hardcode file prefixes like `gridRaw_`
- **Mathematical Robustness**: Use `np.maximum(0, value)` to prevent negative values rather than conditional checks

### Processing Standards
- **Time Series**: Use LOESS filtering for climate vs weather separation
- **Interpolation**: Linear interpolation between known data points, preserve exact values at original points
- **Memory Efficiency**: Process large grids with chunking, write outputs after each step completion
- **Loop Hierarchy**: For visualizations, use target as innermost loop for 3-per-page grouping

### Grid Cell Processing Pattern
```python
def is_valid_economic_grid_cell(gdp_value, population_value):
    return gdp_value > 0 and population_value > 0

# Usage in processing loops:
if not is_valid_economic_grid_cell(gdp_cell, pop_cell):
    result[lat_idx, lon_idx] = 0.0  # or np.zeros() for arrays
    continue
```

### Visualization Standards
- **Zero-biased scaling**: Extend ranges to include zero when appropriate for signed variables
- **TFP visualization**: Use 0-to-max range (TFP values are always positive)
- **Valid cell filtering**: Apply `valid_mask` before calculating ranges and statistics
- **Consistent layouts**: 3 maps/charts per page arranged vertically with max/min value boxes

## Current Status (December 2025)

### ✅ **Production Ready**
Complete 5-step integrated processing pipeline with adaptive optimization and standardized visualization.

### 🔍 **Known Issues Under Investigation**
1. **SSP Scenario Differentiation**: SSP245/SSP585 forward model results nearly identical
2. **Combined Response Function**: Not achieving 10% target reduction under SSP245 quadratic scenario

### Recent Key Enhancements
- **Adaptive Bounds Expansion**: Step 3 optimization automatically expands search bounds by 10× when hitting limits
- **Visualization Standardization**: All maps use `pcolormesh` with 3-per-page layouts
- **Enhanced Analysis**: GDP-weighted statistics and objective function metrics in CSV outputs