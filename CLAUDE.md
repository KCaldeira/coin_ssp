# Claude Code Style Guide for COIN_SSP

## Coding Philosophy
This project prioritizes **elegant, fail-fast code** that surfaces errors quickly rather than hiding them.

<!-- CRITICAL-PRESERVE-START -->
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
- **Compute once, use many times** - move invariant calculations outside loops and create centralized helper functions

### Function Design
- Functions should assume valid inputs and focus on their core mathematical/logical purpose
- Let Python's natural error messages guide debugging rather than custom error handling
- **All function arguments must be explicitly provided** - no default values (=None) or conditional logic
- **Clean fail-fast approach** - if required arguments are not supplied, the code should fail immediately with a clear error

### Configuration Parameter Pattern
**For non-primitive functions requiring configuration data, pass the `config` object directly rather than extracting individual parameters at call sites.**

✅ **Use `config` parameter for:**
- Functions that need 2+ configuration values
- Non-primitive functions (business logic, processing pipelines, optimization)
- Functions called from multiple locations with the same config

❌ **Keep explicit parameters for:**
- Primitive mathematical functions (`calculate_growth_rate(values)`)
- Pure utility functions that don't depend on business configuration
- When only 1 simple parameter is needed and unlikely to grow

## Project Architecture

### Module Organization
- **`coin_ssp_core.py`**: Core economic model functions and parameter classes
- **`coin_ssp_utils.py`**: Consolidated utility functions for mathematical operations, visualization, NetCDF processing
- **`main.py`**: Complete 5-step processing pipeline

### Key Technical Requirements
- **NetCDF Convention**: Arrays follow `[time, lat, lon]` dimension order - always use `data[time_idx, lat_idx, lon_idx]`
- **Grid Cell Validation**: Skip ocean/ice cells where `gdp_value <= 0` or `population_value <= 0`
- **Configuration Management**: Use `resolve_netcdf_filepath()` - NEVER hardcode file prefixes like `gridRaw_`
- **Mathematical Robustness**: Use `np.maximum(0, value)` to prevent negative values rather than conditional checks
- **File Naming Convention**: All output files follow standardized pattern `{prefix}_{json_id}_{model_name}_{ssp_name}_{var_name}.{ext}` with missing fields omitted

### Processing Standards
- **Time Series**: Use LOESS filtering for weather extraction (detrends relative to reference period mean)
- **Interpolation**: Linear interpolation between known data points, preserve exact values at original points
- **Memory Efficiency**: Process large grids with chunking, write outputs after each step completion

### Variability Scaling Framework
- **Mathematical Form**: Climate sensitivity varies with local temperature as `v(T) = v0 + v1*T + v2*T²`
- **Parameter Integration**: Variability parameters (v0, v1, v2) added to ModelParams with defaults (1.0, 0.0, 0.0)
- **Forward Model**: Updated `calculate_coin_ssp_forward_model` to apply variability scaling: `f(T) = v(T) * (f1*T + f2*T²) - v(T_ref) * (f1*T_ref + f2*T_ref²)`
- **Backward Compatibility**: Default values ensure zero behavioral change for `target_type: "damage"`
- **Architecture**: Clean separation between baseline climate parameters (k_tas1, etc.) and variability scaling (v0, v1, v2)

#### Variability Calibration Algorithm
**NEW IMPLEMENTATION (December 2025)**: `calculate_variability_climate_response_parameters` now uses a 4-step calibration process:

**Step 1: Optimization for Uniform 10% GDP Loss**
- Run optimization to find scaling factors that produce uniform 10% GDP loss in target period
- Establishes baseline strength of climate-economy relationship needed for target impact
- Uses dummy target with 10% constant reduction across all grid cells

**Step 2: Forward Model Simulations with Scaled Parameters**
- Take parameters from Step 1, scaled by found factors
- Run forward model simulations for each grid cell using scaled parameters
- Generate economic projections over full time period (historical + future)

**Step 3: Weather Variability Regression Analysis**
- For each grid cell: compute regression `log(y_weather) ~ tas_weather` over historical period
- `y_weather` = weather component of GDP (detrended, LOESS-filtered economic signal)
- `tas_weather` = weather component of temperature (detrended, LOESS-filtered climate signal)
- Regression slope = fractional change in GDP per degree C of weather variability

**Step 4: Parameter Normalization by Regression Slope**
- Divide all climate response parameters from Step 1 by regression slope from Step 3
- Normalizes parameters to represent correct strength per degree of variability
- Final parameters capture both target impact magnitude AND observed weather sensitivity

**Status**: Implementation complete, ready for testing

#### Standard Loop Nesting Orders
Two mandatory patterns for consistency:

**Computational Loop Order (Steps 3-4):**
```python
for target_idx in range(n_targets):           # GDP reduction target (outermost)
    for response_idx in range(n_response_funcs):      # Response function
        for lat_idx in range(nlat):                    # Latitude
            for lon_idx in range(nlon):                    # Longitude
```

**Visualization Loop Order (All PDFs):**
```python
for ssp in ssp_scenarios:                    # SSP scenario (outermost)
    for response_idx in range(n_response_funcs):    # Response function
        for target_idx in range(n_targets):         # GDP target (INNERMOST)
```
**Critical**: Target innermost in visualizations groups all GDP reduction targets on same page

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

<!-- CRITICAL-PRESERVE-END -->

## Current Status (December 2025)

### ✅ **Production Ready**
Complete 5-step integrated processing pipeline with adaptive optimization and standardized visualization.

### Recent Key Enhancements
- **Weather Variables**: Centralized computation and storage in `all_data` structure
- **Reference Baselines**: Pre-computed `tas0_2d`/`pr0_2d` climate baselines stored in `all_data`
- **Function Signatures**: Simplified parameter lists using `all_data` and `config` patterns
- **Import Organization**: All imports moved to top of files for clarity
- **Variable Naming**: Consistent `tas`/`pr` climatological conventions throughout codebase
- **Adaptive Bounds Expansion**: Step 3 optimization automatically expands search bounds by 10× when hitting limits
- **Visualization Standardization**: All maps use `pcolormesh` with adaptive 3-per-page layouts
- **Backward Compatibility Cleanup**: Removed all legacy field name support and optional argument defaults
- **Variability Scaling Framework**: Complete implementation of temperature-dependent climate sensitivity with v0, v1, v2 parameters
- **Function Reorganization**: Renamed and restructured variability functions for clarity (`calculate_variability_climate_response_parameters`, `calculate_variability_scaling_parameters`)