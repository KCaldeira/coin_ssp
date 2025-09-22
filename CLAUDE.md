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

### Configuration Parameter Pattern
**For non-primitive functions requiring configuration data, pass the `config` object directly rather than extracting individual parameters at call sites.**

#### When to Use Config Parameter Pattern
✅ **Use `config` parameter for:**
- Functions that need 2+ configuration values
- Functions that might need additional config in future iterations
- Non-primitive functions (business logic, processing pipelines, optimization)
- Functions called from multiple locations with the same config

❌ **Keep explicit parameters for:**
- Primitive mathematical functions (`calculate_growth_rate(values)`)
- Pure utility functions that don't depend on business configuration
- Functions where config dependency would be artificial/forced
- When only 1 simple parameter is needed and unlikely to grow

#### Benefits
- **Future-proof**: No signature changes when adding config requirements
- **Fail-fast**: Missing config keys cause immediate, clear errors
- **Single source of truth**: Configuration comes directly from authoritative source
- **Cleaner call sites**: No parameter extraction boilerplate
- **Easier testing**: Mock/modify entire config rather than tracking individual parameters

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

### Key Technical Requirements
- **NetCDF Convention**: Arrays follow `[time, lat, lon]` dimension order - always use `data[time_idx, lat_idx, lon_idx]`
- **Grid Cell Validation**: Skip ocean/ice cells where `gdp_value <= 0` or `population_value <= 0`
- **Configuration Management**: Use `resolve_netcdf_filepath()` - NEVER hardcode file prefixes like `gridRaw_`
- **Mathematical Robustness**: Use `np.maximum(0, value)` to prevent negative values rather than conditional checks
- **Post-Processing Data Consistency**: All data objects in post-processing code must use the same format as main processing code to facilitate function reusability
- **File Naming Convention**: All output files follow standardized pattern `{prefix}_{json_id}_{model_name}_{ssp_name}_{var_name}.{ext}` with missing fields omitted
- **Configuration Identity**: Each JSON config must include `json_id` in `run_metadata` section for file naming consistency

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

<!-- CRITICAL-PRESERVE-END -->`

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

<!-- TEMP-DEV-SECTION-START -->
## 🚧 TEMPORARY: Variability Target Implementation Plan

**Status**: Phase 1-3 Complete ✅ - Ready for testing with existing damage targets

### JSON Configuration Updates Required
1. **Add historical_period**:
   ```json
   "historical_period": {
     "start_year": 1861,
     "end_year": 2014,
     "description": "Historical period for variability relationship calibration"
   }
   ```

2. **Add end_year to prediction_period**:
   ```json
   "prediction_period": {
     "start_year": 2015,
     "end_year": 2100,
     "description": "Year before which population and GDP is assumed to grow exponentially"
   }
   ```

3. **GDP Target Types**:
   - `target_type: "damage"` → affects damage in target_period (existing)
   - `target_type: "variability"` → defines GDP variability scaling with temperature variability in historical_period (new)

### Main.py Architecture Changes (around lines 614-700)

**Current Structure**: Single loop calling `optimize_climate_response_scaling` for all targets

**New Structure**: Conditional processing by target type
```python
# Around line 614 - Add early calculations
tas0_2d, pr0_2d = calculate_reference_climate_baselines(temp_data, precip_data, years, config)

variability_reference_scaling = None  # Computed once, reused

for target_idx, gdp_target in enumerate(gdp_targets):
    target_type = gdp_target['target_type']

    if target_type == 'variability':
        if variability_reference_scaling is None:
            # EXPENSIVE: lat-lon loop with optimization to establish reference relationship
            variability_reference_scaling = calculate_reference_gdp_climate_variability(...)

        # CHEAP: multiply reference by target-specific scaling
        results = apply_variability_target_scaling(variability_reference_scaling, gdp_target, ...)

    else:  # target_type == 'damage'
        # EXPENSIVE: separate optimization for each damage target (existing approach)
        results = process_damage_target_optimization(...)  # Extract lines 622-700
```

### New Functions Needed

1. **`calculate_reference_climate_baselines(temp_data, precip_data, years, config)`**
   - Calculate tas0, pr0 as 2D arrays [lat, lon]
   - Based on reference_period mean (like current grid cell optimization)
   - Called once after data loading
   - Returns: (tas0_2d, pr0_2d)

2. **`calculate_reference_gdp_climate_variability(...)`**
   - Does lat-lon loop calling `optimize_climate_response_scaling` (damage-type optimization)
   - Computes linear regression: y_weather ~ tas_weather for each grid cell
   - Returns: reference_slopes [lat, lon] (slope coefficients)
   - Called once per run (expensive)

3. **`apply_variability_target_scaling(reference_slopes, gdp_target, temp_data, precip_data, tas0_2d, pr0_2d, target_idx)`**
   - Uses tas_weather (variability only)
   - Applies: `variability_effect = reference_slope[lat,lon] * f(tas_weather, gdp_target_params)`
   - Returns: results dict for storage in scaling_factors arrays
   - Called once per variability target (cheap)

4. **`process_damage_target_optimization(...)`**
   - Extract existing code from lines 622-700
   - Handle all damage target processing
   - Called once per damage target (expensive)

### Technical Details
- **Climate Data**: Use `tas_weather` (variability component) for variability targets
- **Reference Values**: `tas0_2d`, `pr0_2d` as 2D arrays matching damage optimization approach
- **Performance**: "Calculate once, use multiple times" - expensive reference computation, cheap target applications
- **Storage**: Results stored in same `scaling_factors[lat, lon, damage_idx, target_idx]` arrays for consistency

### Implementation Priority
1. JSON config updates
2. Extract `process_damage_target_optimization` function
3. Add `calculate_reference_climate_baselines`
4. Add stub versions of variability functions
5. Update main.py conditional logic
6. Implement full variability calculation logic

**Key Risk**: Complex refactoring of main optimization loop - test thoroughly with existing damage targets first.
<!-- TEMP-DEV-SECTION-END -->