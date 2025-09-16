# Claude Code Style Guide for COIN_SSP

## Coding Philosophy
This project prioritizes **elegant, fail-fast code** that surfaces errors quickly rather than hiding them.

## Code Style Preferences

### Error Handling
- **No input validation** on function parameters (except for command-line interfaces)
- **No defensive programming** - let exceptions bubble up naturally
- **Fail fast** - prefer code that crashes immediately on invalid inputs rather than continuing with bad data
- **No try-catch blocks** unless absolutely necessary for program logic (not error suppression)
- **Assume complete data** - do not check for missing data fields. If required data is missing, let the code fail with natural Python errors

### Code Elegance
- **Minimize conditional statements** - prefer functional approaches, mathematical expressions, and numpy vectorization
- **Favor mathematical clarity** over defensive checks
- **Use numpy operations** instead of loops and conditionals where possible
- **Prefer concise, readable expressions** over verbose defensive code
- **Compute once, use many times** - move invariant calculations outside loops and create centralized helper functions
- **Centralize repeated logic** - extract common operations into utility functions to prevent bugs and inconsistencies

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
- **`coin_ssp_utils.py`**: Consolidated utility functions for mathematical operations, visualization, time series processing, NetCDF data processing, and target GDP reduction calculations
- **`main_integrated.py`**: Integrated grid-cell processing pipeline implementing the complete 5-step workflow
- **`main.py`**: Country-level workflow orchestration

### Key Principles
- **Time Series Processing**: Use LOESS filtering for separating climate trends from weather variability
- **Mathematical Robustness**: Use `np.maximum(0, value)` to prevent negative capital stock rather than conditional checks
- **NetCDF Data Convention**: Arrays follow standard climate model dimension order `[time, lat, lon]`
- **Memory Efficiency**: Process large gridded datasets using chunking and lazy evaluation
- **Compute Once, Use Many Times**: Pre-compute years arrays, coordinate metadata, and other invariant calculations

### Configuration Management
- **⚠️ File Path Configuration**: NetCDF file prefixes are defined in JSON configuration under `climate_model.netcdf_file_patterns`
- **Use `resolve_netcdf_filepath()` function** rather than hardcoding prefixes in data loading functions
- **NEVER hardcode file prefixes** like `gridRaw_`, `gridRawAlt_`, or `Gridded_` - always read from configuration
- **Automatic year_diverge**: Calculated as `reference_period.end_year + 1` (no longer in JSON)

### Data Processing Philosophy
- **Interpolation over Extrapolation**: Use linear interpolation between known data points
- **Annual Resolution**: Process all time series at annual resolution for consistency
- **Quality Assurance**: Preserve exact values at original data points when interpolating
- **Handle Edge Cases Mathematically**: Use mathematical solutions rather than conditional checks

### Grid Cell Validation
Ocean and ice grid cells (GDP=0 or population=0) should be excluded from economic calculations:

```python
def is_valid_economic_grid_cell(gdp_value, population_value):
    return gdp_value > 0 and population_value > 0

# Usage in processing loops:
if not is_valid_economic_grid_cell(gdp_cell, pop_cell):
    result[lat_idx, lon_idx] = 0.0  # or np.zeros() for arrays
    continue
```

### Visualization Standards
- **Zero-biased scaling**: Extend color ranges to include zero when appropriate for variables that can be positive/negative
- **TFP visualization**: Use 0-to-max range (TFP values are always positive)
- **Valid cell filtering**: Apply valid_mask before calculating ranges and statistics
- **Consistent colormaps**: `RdBu_r` with `TwoSlopeNorm` for zero-centered data, `viridis` for positive-only data

## December 2025 Status

### ✅ **Major Issues Resolved**
1. **Step 3 Optimization Bug**: Fixed to use spatially-varying target reductions instead of constant values
2. **Configuration Cleanup**: Removed redundant `target_amount` fields from response function scalings
3. **Automatic Parameters**: `year_diverge` now calculated automatically from reference period
4. **Enhanced Analysis**: Added GDP- and area-weighted median calculations
5. **Visualization Improvements**: Proper valid cell filtering and appropriate color scaling for all steps

### 🔍 **Current Known Issues**
1. **SSP Scenario Differentiation**: SSP245 and SSP585 forward model maps are nearly identical (should show greater impacts for SSP585)
2. **Combined Response Function**: "combined_balanced_quadratic-based" produces much less than 10% target reduction under SSP245

### 📋 **Current Status**
**PRODUCTION-READY**: Complete 5-step integrated processing pipeline with all major optimization bugs resolved and comprehensive visualization capabilities.

**NEXT PRIORITIES**:
1. Debug combined response function target achievement
2. Investigate SSP scenario differentiation in Step 4
3. Validate climate vs weather scenario differences