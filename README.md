# COIN_SSP: Gridded Climate-Economic Impact Model

A spatially-explicit implementation of the Solow-Swan growth model for assessing climate impacts on economic growth at the grid cell level.

## Overview

COIN_SSP processes gridded NetCDF climate and economic data to quantify how climate change affects economic growth through response functions applied to capital stock, productivity, and output. The complete mathematical framework and implementation details are documented in `METHODS.md`.

### Core Model
- **Economic Framework**: Solow-Swan growth model with DICE-derived parameters
- **Climate Integration**: Temperature and precipitation response functions (linear/quadratic)
- **Variability Scaling**: Temperature-dependent climate sensitivity with g(T) = g0 + g1*T + g2*T²
- **Spatial Processing**: Grid cell-level optimization and forward modeling
- **Scenario Support**: Multiple SSP economic scenarios and climate projections

## Function Calling Tree

The COIN_SSP pipeline follows a structured calling hierarchy organized into five main processing steps. Understanding this structure is essential for navigating the codebase.

### Pipeline Entry Points

**`main.py`** - Single configuration execution
- `run_pipeline()` → Main execution orchestrator that coordinates all 5 processing steps

**`workflow_manager.py`** - Multi-stage workflow orchestration
- `WorkflowManager` → Three-stage pipeline manager for parameter sensitivity analysis
  - `run_stage1()` → Individual response function assessments (6-12 separate runs)
  - `analyze_stage1_results()` → Extract GDP-weighted parameter means from Stage 1 CSV outputs
  - `generate_stage2_config()` → Create multi-variable configuration combining best parameters
  - `run_stage3()` → Execute final simulations with combined response functions

### Data Loading and Preprocessing

Called at pipeline initialization before Step 1:

**`load_all_data()`** → **[coin_ssp_utils.py]**
- Loads and concatenates all NetCDF input files (climate, GDP, population)
- Returns unified `all_data` dictionary used throughout pipeline
- Sub-functions:
  - `load_and_concatenate_climate_data()` → Loads temperature/precipitation from historical + SSP files
  - `load_and_concatenate_pop_data()` → Loads population from historical + SSP files
  - `load_and_concatenate_gdp_data()` → Loads GDP density from SSP-specific files
  - `resolve_netcdf_filepath()` → Resolves file paths using configured prefixes
  - `get_grid_metadata()` → Extracts spatial/temporal coordinates from NetCDF files

**`calculate_weather_vars()`** → **[coin_ssp_utils.py]**
- Computes weather variability components (LOESS-filtered climate signals)
- Separates short-term variability from long-term climate trends
- Sub-functions:
  - `apply_time_series_filter()` → **[coin_ssp_math_utils.py]** LOESS filtering with 30-year window
  - `calculate_area_weights()` → **[coin_ssp_math_utils.py]** Cosine-latitude weighting

### Step 1: Target GDP Changes

**`step1_calculate_target_gdp_changes()`** → **[main.py]**
- Calculates spatial patterns of target GDP reductions that optimization will try to achieve
- Supports multiple target types: constant, linear, quadratic temperature relationships
- Called once per pipeline run using reference SSP

Key functions:
- `calculate_all_target_reductions()` → **[coin_ssp_target_calculations.py]**
  - Computes target reduction patterns for all configured GDP targets
  - Sub-functions:
    - `calculate_constant_target_reduction()` → Uniform spatial targets (e.g., -5% everywhere)
    - `calculate_linear_target_reduction()` → Temperature-dependent linear targets
    - `calculate_quadratic_target_reduction()` → Temperature-dependent quadratic targets
- `save_step1_results_netcdf()` → **[coin_ssp_netcdf.py]**
  - Writes target patterns to NetCDF with full metadata
- `create_target_gdp_visualization()` → **[coin_ssp_reporting.py]**
  - Generates PDF maps showing spatial distribution of targets

**Outputs:**
- `step1_{json_id}_{model}_{ssp}_target_gdp.nc` (~220 KB)
- `step1_{json_id}_{model}_{ssp}_target_gdp_visualization.pdf` (~125 KB)

### Step 2: Baseline TFP

**`step2_calculate_baseline_tfp()`** → **[main.py]**
- Calculates baseline economic variables (TFP, capital) without any climate effects
- Provides counterfactual "what growth would be without climate change"
- Called once per SSP scenario (typically 1-2 SSPs)

Key functions:
- `calculate_tfp_coin_ssp()` → **[coin_ssp_core.py]**
  - Runs Solow-Swan growth model with zero climate response parameters
  - Sub-functions:
    - `calculate_coin_ssp_forward_model()` → Core economic model integration
      - Solves differential equations for capital accumulation
      - Computes TFP from GDP, capital, and population
      - Returns time series of economic variables
- `save_step2_results_netcdf()` → **[coin_ssp_netcdf.py]**
  - Writes baseline TFP and capital to NetCDF
- `create_baseline_tfp_visualization()` → **[coin_ssp_reporting.py]**
  - Generates PDF time series plots and percentile analysis

**Outputs:**
- `step2_{json_id}_{model}_{ssp}_baseline_tfp.nc` (~30 MB per SSP)
- `step2_{json_id}_{model}_baseline_tfp_visualization.pdf` (~470 KB)

### Step 3: Scaling Factor Optimization

**`step3_calculate_scaling_factors_per_cell()`** → **[main.py]**
- Most computationally expensive step (~5-10 minutes for full grid)
- Optimizes scaling factors for each grid cell to match target GDP patterns
- Processes all combinations of response functions × GDP targets
- Uses reference SSP only for calibration

Key functions:

**For damage-type targets:**
- `process_response_target_optimization()` → **[coin_ssp_core.py]**
  - Loops over all grid cells running optimization for each
  - Sub-functions:
    - `optimize_climate_response_scaling()` → Per-grid-cell constrained optimization
      - Uses scipy.optimize.minimize with constraint satisfaction
      - Objective: minimize squared error between simulated and target GDP
      - Calls `calculate_coin_ssp_forward_model()` repeatedly during optimization
      - Returns scaling factor that best matches target for this grid cell

**For variability-type targets:**
- `calculate_variability_climate_response_parameters()` → **[coin_ssp_core.py]** (NEW December 2025)
  - 4-step calibration process for variability targets:
    1. **Phase 1**: Optimize for uniform 10% GDP loss (establishes baseline strength)
    2. **Phase 2**: Run forward model with weather components to isolate variability effects
    3. **Phase 3**: Compute regression slopes (GDP_weather ~ TAS_weather) over historical period
    4. **Phase 4**: Normalize parameters by regression slope to match observed sensitivity
  - Returns calibrated parameters for all response functions
  - Outputs per-response-function regression slope statistics to CSV

- `calculate_variability_scaling_parameters()` → **[coin_ssp_core.py]**
  - Applies variability parameters with target-specific scaling

**Analysis and reporting:**
- `calculate_weather_gdp_regression_slopes()` → **[coin_ssp_core.py]**
  - Analyzes historical weather-GDP relationships for all response functions
  - Computes regression slopes for each grid cell
  - Returns GDP-weighted statistics
- `save_step3_results_netcdf()` → **[coin_ssp_netcdf.py]**
  - Writes scaling factors, parameters, convergence flags to NetCDF
- `create_scaling_factors_visualization()` → **[coin_ssp_reporting.py]**
  - Generates PDF maps of optimized scaling factors
- `create_objective_function_visualization()` → **[coin_ssp_reporting.py]**
  - Generates PDF maps of optimization errors
- `create_regression_slopes_visualization()` → **[coin_ssp_reporting.py]**
  - Generates PDF maps of weather-GDP regression slopes
- `print_gdp_weighted_scaling_summary()` → **[coin_ssp_reporting.py]**
  - Computes and writes GDP-weighted statistics to CSV
- `write_variability_calibration_summary()` → **[coin_ssp_reporting.py]**
  - Writes per-response-function variability calibration results to CSV

**Outputs:**
- `step3_{json_id}_{model}_{ssp}_scaling_factors.nc` (~3-6 MB)
- `step3_{json_id}_{model}_{ssp}_scaling_factors_summary.csv` (~1-2 KB)
- `step3_{json_id}_{model}_{ssp}_variability_calibration_summary.csv` (~1 KB, if variability targets exist)
- `step3_{json_id}_{model}_{ssp}_scaling_factors_visualization.pdf` (~700 KB)
- `step3_{json_id}_{model}_{ssp}_objective_function_visualization.pdf` (~700 KB)
- `step3_{json_id}_{model}_{ssp}_regression_slopes_visualization.pdf` (~460 KB)

### Step 4: Forward Integration

**`step4_forward_integration_all_ssps()`** → **[main.py]**
- Runs forward model for all configured SSP scenarios using calibrated parameters
- Generates climate-integrated economic projections (GDP, capital, TFP)
- Only executed if `forward_simulation_ssps` specified in configuration

Key functions:
- `calculate_coin_ssp_forward_model()` → **[coin_ssp_core.py]**
  - Core Solow-Swan model with climate response functions applied
  - Called for every grid cell × response function × target × SSP combination
  - Uses calibrated scaling factors from Step 3
- `save_step4_results_netcdf_split()` → **[coin_ssp_netcdf.py]**
  - Writes separate NetCDF files for each SSP and variable type
  - Files: `step4_{json_id}_{model}_{ssp}_forward_{gdp|tfp|capital}.nc`
- `create_forward_model_visualization()` → **[coin_ssp_reporting.py]**
  - Generates time series line plots comparing scenarios
- `create_forward_model_maps_visualization()` → **[coin_ssp_reporting.py]**
  - Generates spatial impact maps (both linear and log10 scales)
- `create_forward_model_ratio_visualization()` → **[coin_ssp_reporting.py]**
  - Generates ratio maps (climate/weather effects)

**Outputs:**
- `step4_{json_id}_{model}_{ssp}_forward_{variable}.nc` (~65-70 MB per SSP×variable)
- `step4_{json_id}_{model}_forward_model_lineplots.pdf` (~55-125 KB)
- `step4_{json_id}_{model}_forward_model_maps.pdf` (~900 KB - 2.6 MB)
- `step4_{json_id}_{model}_forward_model_maps_log10.pdf` (~900 KB - 2.7 MB)
- `step4_{json_id}_{model}_forward_model_ratios.pdf` (~70-180 KB)

### Step 5: Processing Summary

**`step5_processing_summary()`** → **[main.py]**
- Prints final statistics and timing information
- Currently minimal - placeholder for future summary analysis

### Utility Functions

**Mathematical Operations** → **[coin_ssp_math_utils.py]**
- `calculate_global_mean()` → Area-weighted spatial averages with masking
- `calculate_area_weights()` → Cosine-latitude area weighting
- `calculate_zero_biased_range()` → Visualization range calculation (extends to include zero)
- `calculate_time_means()` → Temporal averaging over specified periods
- `apply_loess_subtract()` → Degree-2 LOESS smoothing with trend subtraction and reference period mean addition
- `apply_loess_divide()` → Degree-2 LOESS smoothing applied to log-transformed data (difference of logs for quasi-exponential series like GDP)

**Data I/O Operations** → **[coin_ssp_netcdf.py]**
- `create_serializable_config()` → Converts config dict to JSON-safe format for NetCDF attributes
- `extract_year_coordinate()` → Extracts time coordinates from NetCDF files
- `interpolate_to_annual_grid()` → Temporal interpolation to annual resolution
- `resolve_netcdf_filepath()` → Constructs file paths using configured prefixes and naming conventions

**Visualization Utilities** → **[coin_ssp_reporting.py]**
- `get_adaptive_subplot_layout()` → Calculates optimal subplot arrangement based on number of targets
- `add_extremes_info_box()` → Adds min/max value boxes to map visualizations
- All visualization functions use consistent styling and adaptive layouts

## Installation

```bash
git clone https://github.com/KCaldeira/coin_ssp.git
cd coin_ssp
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Quick Start - Primary Test

**Test the complete workflow with this single command:**
```bash
python workflow_manager.py coin_ssp_config_linear_parameter_sensitivity.json coin_ssp_config_test_template.json
```

This executes a 3-stage sensitivity analysis workflow combining linear parameter variations with test target configurations.

### Complete Processing Pipeline

**Single Configuration Run:**
```bash
# Run full 5-step integrated workflow
python main.py coin_ssp_config_0008.json

# Skip Step 3 optimization (faster development)
python main.py coin_ssp_config_0008.json --step3-file previous_step3_results.nc
```

**Multi-Stage Workflow:**
```bash
# Full 3-stage workflow: parameter assessment → analysis → multi-variable simulation
python workflow_manager.py coin_ssp_config_parameter_sensitivity.json coin_ssp_config_response_functions_template.json

# Start from specific stage
python workflow_manager.py config_sensitivity.json config_template.json --start-stage 2 --stage1-output ./output_dir/
python workflow_manager.py config_sensitivity.json config_template.json --start-stage 3 --stage2-config ./configs/stage2_generated.json
```

### Processing Steps Overview
1. **Target GDP Changes**: Calculate spatially-explicit economic impact targets
2. **Baseline TFP**: Compute total factor productivity without climate effects
3. **Scaling Optimization**: Per-grid-cell parameter calibration for response functions
4. **Forward Integration**: Climate-integrated economic projections for all SSPs
5. **Summary Generation**: Aggregate results and visualization

## Configuration

See the full README for detailed configuration documentation. Key sections:
- `run_metadata`: Identifies the configuration
- `climate_model`: NetCDF file patterns and variable names
- `ssp_scenarios`: Reference and forward SSP scenarios
- `time_periods`: Reference, historical, target, and prediction periods
- `gdp_targets`: Economic impact targets (damage or variability types)
- `model_params`: Solow-Swan model parameters
- `response_function_scalings`: Climate response function configurations

## Output Structure

Results are organized in timestamped directories:
```
data/output/output_{model}_{timestamp}/{json_id}_{timestamp}/
├── coin_ssp_config_{json_id}.json (configuration copy)
├── all_loaded_data_{json_id}_{model}.nc (all input data)
├── step1_{json_id}_{model}_{ssp}_target_gdp.* (target patterns)
├── step2_{json_id}_{model}_{ssp}_baseline_tfp.* (baseline economics)
├── step3_{json_id}_{model}_{ssp}_scaling_factors.* (optimization results)
├── step3_{json_id}_{model}_{ssp}_*_summary.csv (GDP-weighted statistics)
└── step4_{json_id}_{model}_{ssp}_forward_*.* (climate projections, if configured)
```

## Key Features

✅ **Production Ready**: Complete 5-step processing pipeline
✅ **Adaptive Optimization**: Automatic bounds expansion when hitting limits
✅ **Variability Calibration**: 4-step algorithm for variability-type targets
✅ **Weather Analysis**: Pre-computed LOESS-filtered climate variability
✅ **Fail-Fast Design**: Clean error handling without defensive programming
✅ **Comprehensive Visualization**: Multi-page PDFs with adaptive layouts

## Documentation

- **`METHODS.md`**: Complete mathematical formulation and academic methods
- **`CLAUDE.md`**: Code style guide and architecture decisions
- **`README.md`**: This file - usage and reference documentation

## Next Steps

### Immediate Priorities

1. **Complete xarray DataArray Migration**
   - Finish removing all legacy code that used to work with numpy arrays
   - Convert remaining numpy array initializations to xarray DataArrays (e.g., regression slopes in coin_ssp_netcdf.py)
   - All arrays with time, lat, or lon dimensions MUST be xarray DataArrays with properly labeled coordinates
   - Remove any remaining numpy fallback branches

2. **Step 1 Mean Reduction Verification**
   - Investigate why mean reductions printed on Step 1 reports are not consistent with values requested in config JSON file
   - Verify target calculation functions are correctly implementing requested reduction percentages
   - Check GDP-weighted mean calculations for accuracy

## Contributing

This project follows elegant, fail-fast coding principles:
- No input validation on function parameters
- Let exceptions bubble up naturally
- Prefer mathematical clarity over defensive checks
- Use numpy vectorization instead of loops

## License

MIT License - See LICENSE file for details.

## Citation

If you use COIN_SSP in your research, please cite:
```
COIN_SSP: A spatially-explicit climate-economic impact model implementing
the Solow-Swan growth model with gridded climate response functions.
```
