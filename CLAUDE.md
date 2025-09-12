# Claude Code Style Guide for COIN_SSP

## Coding Philosophy
This project prioritizes **elegant, fail-fast code** that surfaces errors quickly rather than hiding them.

## Code Style Preferences

### Error Handling
- **No input validation** on function parameters (except for command-line interfaces)
- **No defensive programming** - let exceptions bubble up naturally
- **Fail fast** - prefer code that crashes immediately on invalid inputs rather than continuing with bad data
- **No try-catch blocks** unless absolutely necessary for program logic (not error suppression)
- **Assume complete data** - do not check for missing data fields (e.g., precipitation). If required data is missing, let the code fail with natural Python errors rather than handling gracefully

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

## Project Context
This is a climate economics modeling project implementing the Solow-Swan growth model for gridded climate impact assessment. Code should reflect the mathematical elegance of economic models while maintaining computational efficiency for grid-scale analysis.

### Module Organization
- **`coin_ssp_core.py`**: Core economic model functions and parameter classes
- **`coin_ssp_utils.py`**: Consolidated utility functions for mathematical operations, visualization, time series processing, and NetCDF data processing
- **`calculate_target_gdp_reductions.py`**: Standalone tool for spatially-explicit target GDP reduction calculations
- **`main.py`**: Country-level workflow orchestration

Keep helper functions organized in utils to maintain clean separation of concerns. The utilities module now consolidates both mathematical/visualization functions and NetCDF data processing functions to reduce code duplication.

### Time Series Processing
- **LOESS Filtering**: Use locally weighted scatterplot smoothing (LOESS) for separating climate trends from weather variability
- **Trend Preservation**: Maintain long-term climate trends while filtering out interannual variability after reference year (2025)

## Data Processing Philosophy
- **Interpolation over Extrapolation**: Use linear interpolation between known data points rather than extrapolating beyond available data
- **Annual Resolution**: Process all time series at annual resolution for consistency between climate and economic data
- **Country Completeness**: Include countries only when they have complete time series data across all required variables
- **Quality Assurance**: Preserve exact values at original data points when interpolating (e.g., 5-year economic data points)

## Mathematical Robustness
- **Handle Edge Cases Mathematically**: Use `np.maximum(0, value)` to prevent negative capital stock rather than conditional checks
- **L-BFGS-B Optimization**: Prefer gradient-based optimization with proper bounds for parameter calibration
- **Direct Parameter Scaling**: Use `param = scale * base_value` rather than multiplicative scaling to avoid zero-multiplication issues

### NetCDF Data Handling
- **xarray Integration**: Use xarray for elegant NetCDF operations and coordinate handling
- **Vectorized Operations**: Apply economic calculations across entire spatial grids simultaneously
- **Memory Efficiency**: Process large gridded datasets using chunking and lazy evaluation where appropriate
- **Metadata Preservation**: Maintain proper coordinate systems, units, and attributes in output files
- **Unit Conversion**: Convert temperature from Kelvin to Celsius using physical constant 273.15
- **Area Weighting**: Use cosine of latitude for proper global mean calculations
- **Consolidated Utilities**: Keep all NetCDF processing functions in `coin_ssp_utils.py` to avoid duplication
- **Repository Storage**: Small NetCDF files can be stored in git with appropriate .gitignore patterns

## Lessons Learned: Constraint System Design

### Target GDP Reduction Implementation
The implementation of spatially-explicit target reduction functions revealed important lessons about constraint system design in climate economics:

#### Mathematical vs. Physical Consistency
- **Issue**: Constraint systems can be mathematically correct but produce physically unrealistic results
- **Example**: Linear temperature-damage function showing backwards relationship (more damage at colder temperatures)
- **Root Cause**: GDP-weighted constraints combined with spatial temperature patterns can create counterintuitive solutions
- **Solution**: Always verify that mathematical solutions align with physical/economic expectations

#### Debugging Complex Algorithms
- **Document Mathematical Basis**: Extensive comments showing constraint equations, solution methods, and expected behavior
- **Add Debug Output**: Print intermediate calculations (coefficients, constraint values) to verify algorithm correctness  
- **Spatial Diagnosis**: Create diagnostic tools to examine spatial patterns and validate temperature-damage relationships
- **Point-wise Verification**: Test specific temperature points to confirm constraint satisfaction

#### Temperature Dependency in Climate Damage
- **Physical Expectation**: Climate damage should generally increase with temperature
- **Constraint Design**: Ensure constraint systems enforce realistic temperature dependencies
- **Spatial Considerations**: Account for how GDP distribution across temperature gradients affects global constraints
- **Alternative Approaches**: Consider reformulating constraints to use cold-temperature reference points with zero damage

#### Economic Realism vs Mathematical Precision
- **Issue**: Mathematically precise constraint satisfaction can produce economically meaningless results
- **Example**: Quadratic damage function yielding >100% GDP losses (>252% in polar regions)
- **Insight**: Economic bounds are as important as mathematical constraints in climate damage modeling
- **Solution**: Implement realistic bounds (e.g., maximum 80% GDP loss) alongside mathematical precision

#### Function Form Selection
- **Unconstrained quadratics**: Can produce extreme values in temperature ranges outside calibration data
- **Bounded alternatives**: Consider sigmoid functions, piecewise linear, or constrained optimization with bounds
- **Trade-offs**: Balance mathematical tractability with economic plausibility across full spatial temperature range

These lessons emphasize the importance of validating not just mathematical correctness but also physical realism and economic plausibility in climate-economic modeling algorithms.

## Next Development Phase: Grid Cell Processing

### Planned Implementation Strategy
The next major development will transition from the current target GDP reduction utilities to comprehensive grid cell processing for climate-economic modeling:

#### Phase 1: SSP245-Based Scaling Factor Calibration
**Objective**: Calculate scaling factors once per climate model using SSP245 as the calibration scenario
- **Scope**: 6 damage function cases × 3 target GDP cases = 18 scaling factor combinations per model
- **Damage Functions**: Linear/quadratic variants for output, capital stock, and TFP growth mechanisms
- **Target GDP Cases**: Constant, linear (temperature-dependent), quadratic (temperature-dependent)
- **Technical Foundation**: Leverage existing `optimize_climate_response_scaling` function from `main.py:145-147`

#### Phase 2: Multi-SSP TFP Time Series Generation  
**Objective**: Calculate baseline TFP time series for each model and SSP pathway combination
- **Scope**: All available SSP scenarios per climate model
- **Method**: Apply existing `calculate_tfp_coin_ssp` to gridded economic data
- **Output**: Baseline TFP arrays (no climate effects) for forward model initialization

#### Phase 3: Comprehensive Forward Integration
**Objective**: Run climate-integrated forward model for each grid cell across all scenarios
- **Scope**: All SSP pathways using Phase 1 scaling factors
- **Processing**: Apply `calculate_coin_ssp_forward_model` spatially across grids
- **Output**: 18 result cases per grid cell per model (damage function × target GDP combinations)

### Implementation Principles
- **Vectorized Processing**: Maintain computational efficiency for grid-scale calculations
- **Parameter Reuse**: Compute scaling factors once, apply across multiple SSP scenarios
- **Modular Design**: Build on existing economic model core and gridded data infrastructure
- **Quality Assurance**: Extend current validation approaches to spatial processing

### Technical Requirements
- **Memory Management**: Handle large gridded arrays efficiently with chunking strategies
- **Parallel Processing**: Consider grid cell independence for computational optimization  
- **Output Structure**: Design NetCDF output schemas for multi-scenario, multi-damage function results
- **Validation Tools**: Extend diagnostic capabilities to spatial patterns and cross-scenario consistency