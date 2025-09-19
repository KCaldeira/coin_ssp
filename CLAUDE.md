# Claude Code Style Guide for COIN_SSP

## Coding Philosophy
This project prioritizes **elegant, fail-fast code** that surfaces errors quickly rather than hiding them.

## 🔒 Preservation Markers for Critical Content
Use these markers to indicate content that should NEVER be removed during documentation updates:

- `<!-- CRITICAL-PRESERVE-START -->` and `<!-- CRITICAL-PRESERVE-END -->` for essential sections
- `<!-- VALUABLE-PRESERVE-START -->` and `<!-- VALUABLE-PRESERVE-END -->` for important but updateable content
- `<!-- LESSON-PRESERVE-START -->` and `<!-- LESSON-PRESERVE-END -->` for debugging lessons and technical insights

These markers ensure critical institutional knowledge is never lost during refactoring.


<!-- CRITICAL-PRESERVE-START -->`
## Core Style Requirements

### Error Handling
- **No input validation** on function parameters (except for command-line interfaces)
- **No defensive programming** - let exceptions bubble up naturally
- **Fail fast** - prefer code that crashes immediately on invalid inputs rather than continuing with bad data
- **No try-catch blocks** unless absolutely necessary for program logic (not error suppression)
- **No optional function arguments** - all parameters must be explicitly provided
- **No backward compatibility** - no conditional logic to handle missing arguments or legacy data formats
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
- **All function arguments must be explicitly provided** - no default values (=None) or conditional logic
- **Clean fail-fast approach** - if required arguments are not supplied, the code should fail immediately with a clear error

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
# ❌ Defensive programming with input validation
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

# ❌ Optional arguments with backward compatibility
def process_data(input_data, config, optional_data=None):
    if optional_data is not None:
        # Use provided data
        data = optional_data
    else:
        # Fallback to loading fresh data
        data = load_data(config)
    return analyze(data)
```

## Project Architecture

### Module Organization
- **`coin_ssp_core.py`**: Core economic model functions and parameter classes
- **`coin_ssp_utils.py`**: Consolidated utility functions for mathematical operations, visualization, NetCDF processing
- **`main.py`**: Complete 5-step processing pipeline

<!-- CRITICAL-PRESERVE-END -->`

### Key Technical Requirements
- **NetCDF Convention**: Arrays follow `[time, lat, lon]` dimension order - always use `data[time_idx, lat_idx, lon_idx]`
- **Grid Cell Validation**: Skip ocean/ice cells where `gdp_value <= 0` or `population_value <= 0`
- **Configuration Management**: Use `resolve_netcdf_filepath()` - NEVER hardcode file prefixes like `gridRaw_`
- **Mathematical Robustness**: Use `np.maximum(0, value)` to prevent negative values rather than conditional checks

### Processing Standards
- **Time Series**: Use LOESS filtering for weather extraction (detrends relative to reference period mean)
- **Interpolation**: Linear interpolation between known data points, preserve exact values at original points
- **Memory Efficiency**: Process large grids with chunking, write outputs after each step completion
- **Standard Loop Nesting Orders**: Two mandatory patterns for consistency

#### Computational Loop Order (Steps 3-4)
```python
for target_idx in range(n_targets):           # GDP reduction target (outermost)
    for damage_idx in range(n_damage_funcs):      # Damage function
        for lat_idx in range(nlat):                    # Latitude
            for lon_idx in range(nlon):                    # Longitude
```

#### Visualization Loop Order (All PDFs)
```python
for ssp in ssp_scenarios:                    # SSP scenario (outermost)
    for damage_idx in range(n_damage_funcs):    # Damage function
        for target_idx in range(n_targets):         # GDP target (INNERMOST)
```
**Critical**: Target innermost in visualizations groups all 3 GDP reduction targets on same page

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

#### Colormap Strategy
- **RdBu_r (Red-Blue)**: For zero-centered data with symmetric scaling (climate impacts, GDP changes)
  - Blue = positive values, Red = negative values, White = zero
  - Always use with `TwoSlopeNorm(vcenter=0.0)`
- **Viridis**: For non-zero-centered data and outlier detection (TFP, objective functions, log ratios)
  - Use full data range to highlight outliers and extreme values
  - Use with `Normalize()` for min-to-max scaling

## Current Status (December 2025)

### ✅ **Production Ready**
Complete 5-step integrated processing pipeline with adaptive optimization and standardized visualization.

### 🔍 **Known Issues Under Investigation**
None currently under active investigation.

### Recent Key Enhancements
- **Adaptive Bounds Expansion**: Step 3 optimization automatically expands search bounds by 10× when hitting limits
- **Visualization Standardization**: All maps use `pcolormesh` with 3-per-page layouts
- **Enhanced Analysis**: GDP-weighted statistics and objective function metrics in CSV outputs
- **Simplified Weather Filtering**: Time series filtering now detrends relative to reference period mean for all years