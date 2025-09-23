# COIN_SSP: Gridded Climate-Economic Impact Model

A spatially-explicit implementation of the Solow-Swan growth model for assessing climate impacts on economic growth at the grid cell level.

## Overview

COIN_SSP processes gridded NetCDF climate and economic data to quantify how climate change affects economic growth through response functions applied to capital stock, productivity, and output.

### Core Model
- **Economic Framework**: Solow-Swan growth model with DICE-derived parameters
- **Climate Integration**: Temperature and precipitation response functions (linear/quadratic)
- **Spatial Processing**: Grid cell-level optimization and forward modeling
- **Scenario Support**: Multiple SSP economic scenarios and climate projections

## Installation

```bash
git clone https://github.com/KCaldeira/coin_ssp.git
cd coin_ssp
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Complete Processing Pipeline
```bash
# Run full 5-step integrated workflow
python main.py coin_ssp_config_0008.json

# Skip Step 3 optimization (faster development)
python main.py coin_ssp_config_0008.json --step3-file previous_step3_results.nc
```

### Processing Steps
1. **Target GDP Changes**: Calculate spatially-explicit economic impact targets
2. **Baseline TFP**: Compute total factor productivity without climate effects
3. **Scaling Optimization**: Per-grid-cell parameter calibration for response functions
4. **Forward Integration**: Climate-integrated economic projections for all SSPs
5. **Summary Generation**: Aggregate results and visualization

## Configuration

JSON configuration files control all aspects of the COIN_SSP processing pipeline. Below are all available parameters organized by configuration section:

### `run_metadata` (Required)
Identifies the configuration for file naming and reproducibility:
- **`json_id`** (string): Unique identifier used in output filenames
- **`run_name`** (string): Descriptive name for this model run
- **`description`** (string): Detailed description of the experiment
- **`created_date`** (string): Creation date for tracking

### `climate_model` (Required)
Specifies NetCDF file patterns and variable names:
- **`model_name`** (string): Climate model identifier (e.g., "CanESM5")
- **`input_directory`** (string): Base directory for input data files
- **`file_prefixes`** (object): File naming patterns for each data type
  - `tas_file_prefix`: Temperature data file prefix
  - `pr_file_prefix`: Precipitation data file prefix
  - `gdp_file_prefix`: GDP data file prefix
  - `pop_file_prefix`: Population data file prefix
  - `target_reductions_file_prefix`: Target GDP amounts file prefix (optional)
- **`variable_names`** (object): NetCDF variable names within files
  - `tas_var_name`: Temperature variable name (default: "tas")
  - `pr_var_name`: Precipitation variable name (default: "pr")
  - `gdp_var_name`: GDP variable name (default: "gdp_density")
  - `pop_var_name`: Population variable name (default: "pop_density")

File naming convention: `{prefix}_{model_name}_historical.nc`, `{prefix}_{model_name}_{ssp}.nc`

### `ssp_scenarios` (Required)
Controls which SSP scenarios are processed:
- **`reference_ssp`** (string): SSP used for optimization and calibration (e.g., "ssp245")
- **`forward_simulation_ssps`** (array): List of SSPs for forward projections (e.g., ["ssp245", "ssp585"])

### `time_periods` (Required)
Defines temporal windows for different calculations:
- **`reference_period`** (object): Zero-climate change baseline period
  - `start_year`: Start year (e.g., 1861)
  - `end_year`: End year (e.g., 1910)
  - `description`: Human-readable description
- **`historical_period`** (object): Historical period for variability calibration
  - `start_year`: Start year (e.g., 1861)
  - `end_year`: End year (e.g., 2014)
  - `description`: Human-readable description
- **`target_period`** (object): Period for impact assessment
  - `start_year`: Start year (e.g., 2080)
  - `end_year`: End year (e.g., 2100)
  - `description`: Human-readable description
- **`prediction_period`** (object): Forward model projection period
  - `start_year`: Start year (e.g., 2015)
  - `end_year`: End year (e.g., 2100)
  - `description`: Human-readable description

### `gdp_targets` (Required Array)
Defines economic impact targets for optimization. Each target object contains:
- **`target_name`** (string): Unique identifier for this target
- **`target_shape`** (string): Spatial pattern type
  - `"constant"`: Uniform reduction across all grid cells
  - `"linear"`: Linear relationship with temperature
  - `"quadratic"`: Quadratic relationship with temperature
- **`target_type`** (string): Impact mechanism type
  - `"damage"`: Direct economic damage in target period
  - `"variability"`: GDP variability scaling with climate variability
- **`gdp_amount`** (number): Target reduction amount (negative for damage, e.g., -0.05 for 5% reduction)
- **`description`** (string): Human-readable description

For linear targets, additional parameters:
- **`global_mean_amount`** (number): Global mean reduction amount
- **`reference_temperature`** (number): Reference temperature for scaling
- **`amount_at_reference_temp`** (number): GDP amount at reference temperature

For quadratic targets, additional parameters:
- **`global_mean_amount`** (number): Global mean reduction amount
- **`zero_amount_temperature`** (number): Temperature where impact is zero
- **`derivative_at_zero_amount_temperature`** (number): Slope at zero temperature

### `model_params` (Required)
Core Solow-Swan economic model parameters:
- **`s`** (number): Savings rate (typically 0.3)
- **`alpha`** (number): Capital elasticity (typically 0.3)
- **`delta`** (number): Depreciation rate (typically 0.1)

*Note: Baseline temperature (`tas0`) and precipitation (`pr0`) are automatically computed from climate data during the reference period and do not need to be specified in the configuration.*

Climate response parameters (typically 0.0 for base model):
- **`k_tas1`** (number): Capital linear temperature response
- **`k_tas2`** (number): Capital quadratic temperature response
- **`k_pr1`** (number): Capital linear precipitation response
- **`k_pr2`** (number): Capital quadratic precipitation response
- **`tfp_tas1`** (number): TFP linear temperature response
- **`tfp_tas2`** (number): TFP quadratic temperature response
- **`tfp_pr1`** (number): TFP linear precipitation response
- **`tfp_pr2`** (number): TFP quadratic precipitation response
- **`y_tas1`** (number): Output linear temperature response
- **`y_tas2`** (number): Output quadratic temperature response
- **`y_pr1`** (number): Output linear precipitation response
- **`y_pr2`** (number): Output quadratic precipitation response

### `response_function_scalings` (Required Array)
Defines response function configurations for optimization. Each scaling object contains:
- **`scaling_name`** (string): Unique identifier for this response function
- **`description`** (string): Human-readable description
- Climate response parameter overrides (any subset of the parameters from `model_params`):
  - `k_tas1`, `k_tas2`, `k_pr1`, `k_pr2`: Capital stock responses
  - `tfp_tas1`, `tfp_tas2`, `tfp_pr1`, `tfp_pr2`: Total factor productivity responses
  - `y_tas1`, `y_tas2`, `y_pr1`, `y_pr2`: Direct output responses

Example configurations:
```json
{
  "scaling_name": "output_linear",
  "description": "Linear temperature sensitivity for output",
  "y_tas1": 1.0
},
{
  "scaling_name": "capital_quadratic",
  "description": "Quadratic temperature sensitivity for capital",
  "k_tas2": 1.0
}
```

### `processing_options` (Optional)
Advanced processing control options:
- **`grid_cell_processing`** (object): Grid processing settings
  - `enabled`: Enable/disable grid cell processing
  - `parallel_processing`: Enable parallel processing
  - `chunk_size`: Number of cells to process in each chunk
- **`output_formats`** (object): Control output file generation
  - `netcdf`: Generate NetCDF output files
  - `csv`: Generate CSV summary files
  - `pdf_maps`: Generate PDF maps
  - `pdf_line_plots`: Generate PDF time series plots
- **`validation`** (object): Validation and checking options
  - `constraint_verification`: Verify optimization constraints
  - `economic_bounds_check`: Check economic parameter bounds
  - `temperature_damage_relationship`: Validate temperature-damage relationships

### `output_configuration` (Optional)
Output file naming and organization:
- **`output_directory`** (string): Base output directory path
- **`filename_patterns`** (object): Template patterns for output files
- **`include_timestamp`** (boolean): Include timestamps in filenames

**Example Configuration:**
```json
{
  "run_metadata": {
    "json_id": "0008",
    "run_name": "CanESM5_scale-linear-impacts-test",
    "description": "5% and 10% linear damage test",
    "created_date": "2025-09-21"
  },
  "climate_model": {
    "model_name": "CanESM5",
    "input_directory": "data/input",
    "file_prefixes": {
      "tas_file_prefix": "CLIMATE",
      "pr_file_prefix": "CLIMATE",
      "gdp_file_prefix": "GDP",
      "pop_file_prefix": "POP"
    }
  },
  "gdp_targets": [
    {
      "target_name": "const_5%",
      "target_shape": "constant",
      "target_type": "damage",
      "gdp_amount": -0.05
    }
  ]
}
```

## Key Features

✅ **Production Ready**: Complete 5-step processing pipeline
✅ **Adaptive Optimization**: Automatic bounds expansion when hitting optimization limits
✅ **Weather Variables**: Pre-computed LOESS-filtered climate variability
✅ **Variability Targets**: Support for both damage and variability target types
✅ **Fail-Fast Design**: Clean error handling without defensive programming
✅ **Comprehensive Visualization**: Multi-page PDFs with adaptive layouts

## Output Files

Results saved to timestamped directories:
- **NetCDF Data**: Complete step-by-step processing results
- **PDF Visualizations**: Maps, time series, and diagnostic plots
- **CSV Summaries**: GDP-weighted statistics and analysis
- **Configuration Archive**: Complete reproducibility information

## Data Requirements

- **Climate Data**: Gridded NetCDF temperature/precipitation (historical + projections)
- **Economic Data**: Gridded NetCDF GDP/population (SSP scenarios)
- **File Structure**: Standardized naming convention with model/scenario identifiers
- **Storage**: Files located in `./data/input/` directory

## Recent Updates

- **Weather Variables**: Centralized computation and storage in `all_data` structure
- **Reference Baselines**: Pre-computed `tas0_2d`/`pr0_2d` climate baselines
- **Function Signatures**: Simplified parameter lists using `all_data` and `config`
- **Import Organization**: All imports moved to top of files
- **Variable Naming**: Consistent `tas`/`pr` climatological conventions

## Architecture

### Core Modules
- **`coin_ssp_core.py`**: Core economic model and optimization functions
- **`main.py`**: Integrated 5-step processing pipeline
- **`coin_ssp_models.py`**: Data classes and model parameter structures

### Specialized Utility Modules
- **`coin_ssp_math_utils.py`**: Mathematical utilities and helper functions
- **`coin_ssp_netcdf.py`**: NetCDF input/output and serialization functions
- **`coin_ssp_target_calculations.py`**: GDP target reduction calculations
- **`coin_ssp_reporting.py`**: Visualization and PDF generation functions
- **`coin_ssp_utils.py`**: Mathematical utilities and data processing functions

### Configuration
- **JSON-based workflow control**: Unified schema with standardized file naming

## Function Calling Tree

The COIN_SSP pipeline follows a structured calling hierarchy. Below shows the main functions called by `main.py` and the key functions they call (2 levels deep):

### Pipeline Entry Point
**`main.py`**
- `run_pipeline()` → Main execution orchestrator
  - `load_config()` → Configuration validation and setup
  - `setup_output_directory()` → Output directory creation
  - `load_all_data()` → **[coin_ssp_utils.py]** NetCDF data loading
    - `resolve_netcdf_filepath()` → File path resolution
    - `get_grid_metadata()` → Extract spatial/temporal coordinates
  - `calculate_weather_vars()` → **[coin_ssp_utils.py]** Weather variable computation
    - `apply_time_series_filter()` → **[coin_ssp_math_utils.py]** LOESS filtering
    - `calculate_area_weights()` → **[coin_ssp_math_utils.py]** Area weighting

### Processing Steps
**Step 1: Target GDP Changes**
- `step1_calculate_target_gdp_changes()`
  - `calculate_all_target_reductions()` → **[coin_ssp_target_calculations.py]** Target computation
    - `calculate_constant_target_reduction()` → Uniform spatial targets
    - `calculate_linear_target_reduction()` → Temperature-dependent targets
    - `calculate_quadratic_target_reduction()` → Quadratic temperature targets
  - `save_step1_results_netcdf()` → **[coin_ssp_netcdf.py]** NetCDF output
  - `create_target_gdp_visualization()` → **[coin_ssp_reporting.py]** PDF generation

**Step 2: Baseline TFP**
- `step2_calculate_baseline_tfp()`
  - `calculate_tfp_coin_ssp()` → **[coin_ssp_core.py]** Economic model computation
    - `calculate_coin_ssp_forward_model()` → Solow-Swan model integration
  - `save_step2_results_netcdf()` → **[coin_ssp_netcdf.py]** NetCDF output
  - `create_baseline_tfp_visualization()` → **[coin_ssp_reporting.py]** PDF generation

**Step 3: Scaling Optimization**
- `step3_calculate_scaling_factors_per_cell()`
  - `optimize_climate_response_scaling()` → **[coin_ssp_core.py]** Per-grid optimization
    - `calculate_coin_ssp_forward_model()` → Forward model evaluation
    - Scipy optimization routines → Constraint satisfaction
  - `save_step3_results_netcdf()` → **[coin_ssp_netcdf.py]** NetCDF output
  - `create_scaling_factors_visualization()` → **[coin_ssp_reporting.py]** PDF generation

**Step 4: Forward Integration**
- `step4_forward_integration_all_ssps()`
  - `calculate_coin_ssp_forward_model()` → **[coin_ssp_core.py]** Climate-integrated projections
    - Economic model with calibrated response functions
  - `save_step4_results_netcdf_split()` → **[coin_ssp_netcdf.py]** Multi-SSP NetCDF output
  - `create_forward_model_visualization()` → **[coin_ssp_reporting.py]** Time series PDFs
  - `create_forward_model_maps_visualization()` → **[coin_ssp_reporting.py]** Spatial maps PDFs

**Step 5: Processing Summary**
- `step5_processing_summary()`
  - Summary statistics and final diagnostics
  - `create_forward_model_ratio_visualization()` → **[coin_ssp_utils.py]** Ratio analysis PDFs

### Utility Functions
**Mathematical Operations** → **[coin_ssp_math_utils.py]**
- `calculate_global_mean()` → Area-weighted spatial averages
- `calculate_zero_biased_range()` → Visualization range calculation
- `calculate_time_means()` → Temporal averaging

**Data I/O Operations** → **[coin_ssp_netcdf.py]**
- `create_serializable_config()` → JSON metadata for NetCDF
- `extract_year_coordinate()` → Time coordinate extraction
- `interpolate_to_annual_grid()` → Temporal interpolation

## Post-Processing Analysis

**`main_postprocess.py`** provides comprehensive analysis of results generated by the main pipeline.

### Purpose
Analyzes output directories created by `main.py` to extract insights, validate results, and generate additional diagnostics beyond the standard pipeline outputs.

### Usage
```bash
# Analyze complete pipeline output
python main_postprocess.py /path/to/output_integrated_CanESM5_20250919_123456/

# Filter analysis to specific SSPs
python main_postprocess.py output_dir/ --ssps ssp245 ssp585
```

### Key Functions
- **`validate_output_directory()`** → Verify required files exist
- **`obtain_configuration_information()`** → Extract metadata from results
- **`analysis_tas_gdp_correlations()`** → Temperature-GDP relationship analysis
- **`run_all_analyses()`** → Execute complete analysis suite

### Analysis Types
1. **Correlation Analysis**: Temperature-GDP relationships across grid cells and time
2. **Statistical Validation**: Model performance and constraint satisfaction metrics
3. **Sensitivity Analysis**: Response function parameter effectiveness
4. **Comparative Analysis**: Multi-SSP scenario comparisons

### Output
- **Analysis reports**: Statistical summaries and validation metrics
- **Diagnostic plots**: Enhanced visualizations for model validation
- **CSV exports**: Detailed numerical results for further analysis

The post-processing framework is designed to be modular and extensible, allowing researchers to add custom analysis functions for specific research questions.

## Contributing

This project follows elegant, fail-fast coding principles:
- No input validation on function parameters
- Let exceptions bubble up naturally
- Prefer mathematical clarity over defensive checks
- Use numpy vectorization instead of loops

## License

COIN_SSP is open source software designed for scientific research and is distributed under the MIT License. This license allows for free use, modification, and distribution while maintaining attribution to the original authors.

The MIT License is widely adopted in the scientific computing community as it provides maximum flexibility for researchers while ensuring proper attribution. Users are free to:
- Use the software for any purpose, including commercial applications
- Modify and adapt the code for specific research needs
- Distribute modified versions with appropriate attribution
- Incorporate the software into larger research frameworks

For the complete license text, see the LICENSE file in the project repository.

## Citation

If you use COIN_SSP in your research, please cite:
```
COIN_SSP: A spatially-explicit climate-economic impact model implementing the Solow-Swan growth model with gridded climate response functions.
```

This software is intended to advance scientific understanding of climate-economic interactions and is provided as a resource to the research community.