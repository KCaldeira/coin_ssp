# COIN_SSP: Gridded Climate-Economic Impact Model

A spatially-explicit implementation of the Solow-Swan growth model that processes gridded climate and economic data to assess climate impacts on economic growth at the grid cell level under various SSP scenarios and RCP climate trajectories.

## Project Overview

This project implements a forward-looking Solow-Swan growth model that integrates:

- **SSP Economic Scenarios** (gridded GDP and population projections)
- **NetCDF Climate Data** (gridded temperature and precipitation time series)  
- **Climate Damage Functions** with multiple response mechanisms
- **Total Factor Productivity** calculations from observed economic data

The model processes gridded NetCDF data to understand how climate change affects economic growth at the spatial scale of climate models through impacts on capital stock, productivity, and output.

## Model Framework

### Core Economic Model
Based on the Solow-Swan growth model with parameters derived from DICE:

```
Y(t) = A(t) * K(t)^α * L(t)^(1-α)
dK/dt = s * Y(t) - δ * K(t)
```

Where:
- `Y(t)` = Output (GDP)
- `A(t)` = Total Factor Productivity 
- `K(t)` = Capital Stock
- `L(t)` = Labor/Population
- `s` = Savings Rate (30%)
- `α` = Capital Elasticity (0.3) 
- `δ` = Depreciation Rate (10%/year)

### Climate Response Functions

Three types of climate damage mechanisms:

1. **Capital Stock Damage**: `k_climate = 1 + k_tas1*(T-T0) + k_tas2*(T-T0)²`
2. **TFP Growth Damage**: `tfp_climate = 1 + tfp_tas1*(T-T0) + tfp_tas2*(T-T0)²`
3. **Output Damage**: `y_climate = 1 + y_tas1*(T-T0) + y_tas2*(T-T0)²`

All damage functions support both temperature and precipitation sensitivities.

## Data Processing Pipeline

### 1. Raw Data Sources
- **Climate Data**: Gridded NetCDF files with historical (1850-2014) + RCP scenarios (2015-2100) annual temperature/precipitation
- **Economic Data**: Gridded NetCDF files with SSP scenarios for GDP and population (annual resolution, 1950-2100)
- **Storage**: NetCDF input files located in `./data/input/`
- **Coverage**: Global grid coverage at climate model resolution

### 2. Data Harmonization
- **Spatial Alignment**: Ensures consistent grid cell indexing between climate and economic NetCDF files
- **Temporal Alignment**: Synchronizes annual time series across all data sources
- **Quality Assurance**: Validates grid cell completeness and data continuity

### 3. Grid Cell Processing

The following pipeline will be done for each climate model under consideration

**Step 1: Develop target gdp changes using the SSP245 scenario**
```python
# Vectorized across all grid cells
calculate_target_reductions(config_file)
```

**Step 2: Grid Cell TFP Calculation for each of the SSP scenarios, without consideration of climate change**
```python
# Applied to each grid cell independently
tfp_baseline, k_baseline = calculate_tfp_coin_ssp(population_grid, gdp_grid, params)
```

**Step 3: Calculate the scaling factors for each grid cell for each damage function case and each target gdp change for SSP245**

Note that as the json file is now configured we are considering 6 damage function cases (linear and quadratic on output, capital stock and growth rate in total factor productivity) and 3 target gdp changes (constant, linear, and quadratic).

```python
# Applied to each grid cell independently
optimal_scale, final_error, params_scaled = optimize_climate_response_scaling(country_data, params, scaling_params)
```

**Step 4: Climate-Integrated Forward Model**  

Run forward case for all available SSPs for this model using the scaling results from the previous step
```python
# Vectorized across all grid cells
results = calculate_coin_ssp_forward_model(tfp_baseline, population_grid, gdp_grid, temperature_grid, params)
```

### Standard Processing Loop Structure

To maintain consistency and efficiency across all steps, the following loop hierarchy is used throughout the processing pipeline:

```python
# STANDARD LOOP HIERARCHY (outermost to innermost):
for ssp_scenario in ssp_scenarios:           # 1. SSP scenario (when multiple)
    for target_reduction in target_reductions:   # 2. Target GDP reduction  
        for damage_function in damage_functions:     # 3. Damage function
            for lat_idx in range(nlat):                  # 4. Latitude
                for lon_idx in range(nlon):                  # 5. Longitude  
                    for time_idx in range(ntime):               # 6. Time (when needed)
                        # Core computation here
```

**Design Rationale:**
- **SSP Outermost**: Minimizes NetCDF file loading, enables processing one scenario at a time
- **Target/Damage Functions**: Logical grouping of related parameter combinations
- **Spatial Loops**: Grid cell independence enables parallelization
- **Time Innermost**: Optimizes memory locality for time series operations

**Step-Specific Adaptations:**
- **Step 1**: `target_reduction` loop only (single reference SSP, no time loop)
- **Step 2**: `ssp_scenario → spatial` loops (time handled within TFP calculation)
- **Step 3**: `target_reduction → damage_function → spatial` loops (single reference SSP)
- **Step 4**: Full hierarchy for complete scenario × parameter × spatial processing

**Step 5: NetCDF Output Generation**
```python
# Save gridded results to NetCDF with proper metadata
save_gridded_results(results, output_path, grid_metadata)
```

## Implementation Status

### ✅ Completed Components

#### Foundation Economic Model  
- **TFP Calculation Module** (`coin_ssp_determine_tfp.py`): Core Solow-Swan TFP calculation from GDP/population time series
- **Test Framework** (`test_tfp.py`): Validation testing with synthetic 20-year scenarios  
- **Economic Model Core**: Functional implementation of baseline economic calculations

#### Gridded Data Infrastructure
- **NetCDF Data Loaders** (`coin_ssp_utils.py`): Consolidated utilities for gridded climate and economic data from `./data/input/`
  - Functions: `load_gridded_data()`, `calculate_area_weights()`, `calculate_time_means()`, `calculate_global_mean()`
  - Handles 4 NetCDF files: temperature, precipitation, GDP, population
  - Automatic Kelvin to Celsius conversion using physical constant 273.15
  - Area-weighted global means using cosine of latitude
  - Temporal averaging over user-specified year ranges
  - NetCDF files now stored in repository (small file sizes, updated .gitignore)

#### Target GDP Reduction System
- **Target Calculation Utility** (`calculate_target_gdp_reductions.py`): Standalone tool for creating spatially-explicit economic impact targets
  - JSON configuration system (`target_gdp_config_0000.json`)
  - Three reduction types: constant, linear (temperature-dependent), quadratic (temperature-dependent)
  - GDP-weighted global constraint satisfaction using weighted least squares (OLS)
  - NetCDF output (3×lat×lon array) saved to `./data/output/`
  - Multi-page PDF visualization: global maps + temperature-damage function plots
  - Global map headers show GDP-weighted global means (not area-weighted)
  - Fixed color scale (-1 to +1) with actual range annotations
  - Comprehensive constraint verification with 10+ decimal precision

#### Integrated Configuration System
- **Unified JSON Schema** (`coin_ssp_integrated_config_example.json`): Complete workflow configuration combining all processing components
  - **Climate Models**: Flexible NetCDF file naming convention `{prefix}_{model_name}_{ssp_name}.nc`
  - **Damage Function Scalings**: Enhanced scaling parameters with optimization targets from `coin_ssp*.json` files
  - **GDP Reduction Targets**: Integrated target specifications from `target_gdp_config*.json` files with bounded options
  - **SSP Scenarios**: Reference SSP for calibration + list of forward simulation scenarios
  - **Processing Control**: Grid processing options, validation settings, output formats

#### ✅ Complete Integrated Processing Pipeline Implementation
- **Main Integrated Pipeline** (`main_integrated.py`): **COMPLETE** 5-step processing workflow following README Section 3
  - **Step 1**: ✅ **Calculate target GDP changes** using reference SSP (global constraint satisfaction)
  - **Step 2**: ✅ **Generate baseline TFP** for all SSPs (per grid cell, no climate effects)
  - **Step 3**: ✅ **Per-grid-cell scaling factor optimization** - runs `optimize_climate_response_scaling` for each grid cell
  - **Step 4**: ✅ **Forward model integration** for all SSPs using per-cell scaling factors with climate and weather scenarios
  - **Step 5**: ✅ **Processing summary generation** with step-by-step NetCDF output completed by each step
  - **Flexible Dimensions**: All arrays sized dynamically based on JSON configuration
  - **Status**: **ENTIRE WORKFLOW IMPLEMENTED** - Ready for testing, validation, and scientific study runs

### ⚠️ Known Issues

#### Target GDP Reduction Algorithm
- **Quadratic function unrealistic values**: Quadratic reduction shows extreme negative values (>100% GDP loss) in polar regions
- **Root cause**: Unconstrained quadratic function can produce economically meaningless results (GDP reductions >100%)
- **Linear function issue**: Mathematical solution can show counterintuitive temperature-damage relationships in some regions
- **Mathematical accuracy**: Constraint satisfaction is precise (10+ decimals) but economic realism needs bounds
- **Status**: Requires reformulation with realistic bounds or alternative functional forms
- **Documentation**: Extensively documented constraint equations and verification methods in code

### ✅ Recent Fixes and Improvements

#### Critical NaN/Division by Zero Resolution (December 2024)
The integrated processing pipeline encountered NaN generation during optimization due to several interconnected issues with NetCDF data processing:

**Root Causes Identified and Fixed**:
1. **Temporal Interpolation Issues**: Different NetCDF files had mismatched time dimensions (GDP: 18 points, Population: 95 points, Temperature: 86 points)
   - **Solution**: Implemented temporal alignment utilities (`extract_year_coordinate()`, `interpolate_to_annual_grid()`) to harmonize all variables to common annual resolution

2. **Grid Cell Screening Inadequacy**: Original screening only checked first time point, allowing cells with zeros in middle of time series
   - **Solution**: Enhanced screening to validate all time points with centralized `valid_mask` computation

3. **NetCDF Dimension Order Confusion**: Code incorrectly assumed `[lat, lon, time]` when climate model convention is `[time, lat, lon]`
   - **Solution**: Fixed dimension unpacking throughout pipeline: `ntime, nlat, nlon = data.shape` and array indexing `data[:, lat_idx, lon_idx]`

4. **Division by Zero in Optimization**: Negative GDP values constrained to zero caused `y_weather = 0` in ratio calculations
   - **Solution**: Added `RATIO_EPSILON = 1e-20` constant and changed ratio calculation to `y_climate[idx] / (y_weather[idx] + RATIO_EPSILON)`

**Technical Improvements**:
- **Variable Naming**: Renamed `country_data` to `gridcell_data` throughout codebase for clarity in gridded context
- **Progress Indicators**: Added dot-per-latitude-band progress display during optimization step with organized output
- **Enhanced Debugging**: Comprehensive NaN diagnostic output showing complete gridcell data, parameters, and time series when errors occur
- **Error Termination**: RuntimeError with full diagnosis when NaN/Inf ratios detected, enabling root cause analysis

**Code Quality Enhancements**:
- **Centralized Constants**: Module-level `RATIO_EPSILON` for maintainable division-by-zero prevention
- **Improved Warning Messages**: Clear "filtered out N time values with incomplete data" messages instead of confusing warnings
- **Output Cleanup**: Commented out verbose objective function output to reduce console noise during optimization
- **Documentation Updates**: Enhanced CLAUDE.md with NetCDF conventions and debugging lessons learned

**Pipeline Status**: The complete 5-step integrated processing pipeline now runs to completion without NaN errors and provides robust handling of edge cases in gridded climate-economic data processing.

### 🚧 Next Development Phase

#### Integrated Processing Pipeline Implementation Status

**✅ Phase 1: Steps 1-3 Complete**

**Step 1: Target GDP Changes (Complete)**
- **Hybrid Approach Success**: Extracted core functions to `coin_ssp_utils.py` for code reuse
- **Full Integration**: Target GDP calculation with dynamic file resolution and JSON configuration
- **Backward Compatibility**: Existing `calculate_target_gdp_reductions.py` works unchanged  
- **NetCDF Output**: Enhanced NetCDF files with coordinate systems and constraint verification metadata
- **Multi-Target Processing**: Handles any number of constant/linear/quadratic targets with comprehensive results

**Step 2: Baseline TFP Calculation (Complete)**
- **Centralized Data Loading**: Efficient NetCDF data loading for all SSP scenarios upfront using `load_all_netcdf_data()`
- **Grid Cell Processing**: Applies `calculate_tfp_coin_ssp` to each valid grid cell (GDP > 0, population > 0)
- **Multi-SSP Support**: Calculates baseline TFP for all forward simulation SSP scenarios
- **NetCDF Output**: 4D arrays `[ssp, lat, lon, time]` with TFP and capital stock time series
- **Validation Integration**: Grid cell validity masks and processing statistics included

**Step 3: Per-Grid-Cell Scaling Factor Optimization (Complete)**
- **Individual Grid Cell Optimization**: Runs `optimize_climate_response_scaling` for each valid grid cell
- **Complete Parameter Integration**: Uses Step 1 target reductions and Step 2 baseline TFP as inputs
- **Standard Loop Structure**: Follows `target → damage_function → spatial` hierarchy for consistency
- **Enhanced NetCDF Output**: 5D scaled parameter arrays `[lat, lon, damage_func, target, param]` including all 12 climate damage parameters
- **Weather Filtering Integration**: Applies LOESS filtering to separate climate trends from variability
- **Comprehensive Results**: Scaling factors, optimization errors, convergence flags, and scaled damage function parameters
- **Performance Tracking**: Success rates, processing statistics, and error handling with graceful degradation

**✅ Implementation Complete: All Processing Steps Functional**

**Technical Achievements Completed**:
- **Step-by-Step Data Persistence**: Results written to NetCDF after each step completion for debugging and resumability
- **Enhanced NetCDF Output**: Complete coordinate systems, comprehensive metadata, and processing provenance
- **Memory Management**: Large gridded arrays written to disk immediately after processing to free memory
- **Standard Loop Structure**: Consistent `target → damage_function → spatial` hierarchy across all steps
- **Error Handling**: Graceful degradation with detailed success rate reporting for grid cell processing
- **Hybrid Architecture**: Maintains backward compatibility while building integrated functionality

## 🚀 Next Steps: Scientific Study Preparation

### **Phase 1: Testing and Validation**
With the complete integrated workflow now implemented, the next priority is thorough testing and validation:

#### **Pipeline Testing**
- **Small Grid Testing**: Test complete workflow with small grid subsets to validate functionality
- **Integration Testing**: Verify data flow between all steps and NetCDF output integrity
- **Performance Testing**: Assess memory usage, processing time, and scalability for full grids
- **Error Handling Validation**: Test robustness with edge cases and optimization failures

#### **Scientific Validation**
- **Constraint Verification**: Validate that Step 1 target GDP constraints are properly satisfied in Step 4 forward results
- **Economic Bounds Checking**: Ensure Step 4 results show realistic economic projections (no extreme GDP losses)
- **Climate vs Weather Comparison**: Verify that climate scenarios show appropriate differences from weather-only scenarios
- **Cross-SSP Consistency**: Check that relative differences between SSP scenarios are economically sensible

### **Phase 2: Enhanced Data Visualization**
- **NetCDF Analysis Tools**: Develop utilities for exploring and analyzing the 6D output arrays from Step 4
- **Comparative Visualization**: Create tools to visualize climate vs weather scenarios and SSP differences
- **Statistical Summary Tools**: Generate summary statistics across damage functions, targets, and spatial regions
- **Diagnostic Plots**: Extend existing PDF generation to include forward model results and optimization diagnostics

### **Phase 3: Production Scientific Runs**
Once testing and visualization are complete, the pipeline will be ready for full scientific study runs:

#### **Study Design**
- **Climate Model Selection**: Determine which climate models to process for comprehensive analysis
- **Scenario Coverage**: Select appropriate SSP scenarios for climate impact assessment
- **Damage Function Analysis**: Run all damage function configurations to assess uncertainty ranges
- **Regional Analysis**: Process multiple grid resolutions and regional subsets as needed

#### **Production Processing**
- **High-Performance Computing**: Optimize pipeline for cluster/HPC environments for large-scale processing
- **Batch Processing**: Implement scripts for processing multiple climate models and scenarios systematically
- **Quality Assurance**: Establish protocols for validating production run results and identifying processing errors
- **Data Management**: Organize and archive large-scale NetCDF outputs for scientific analysis

#### Core Model
- **Economic Core** (`coin_ssp_core.py`): 
  - `ModelParams` and `ScalingParams` dataclasses with all economic and climate sensitivity parameters
  - `calculate_tfp_coin_ssp()` - baseline TFP from observed GDP/population
  - `calculate_coin_ssp_forward_model()` - climate-integrated economic projections with robust capital stock handling
  - `optimize_climate_response_scaling()` - L-BFGS-B optimization for climate parameter calibration
- **Utilities** (`coin_ssp_utils.py`): 
  - `apply_time_series_filter()` - LOESS filtering for climate trend separation
  - `create_scaled_params()` - centralized parameter scaling (compute once, use many times)
  - `create_country_scaling_page()` - three-panel visualization generation
  - `create_country_pdf_books()` - automated PDF report generation
- **Main Pipeline** (`main.py`): JSON-configured workflow with timestamped outputs

#### Optimization & Calibration
- **Climate Parameter Scaling**: Automated optimization to achieve target economic impacts
- **Multiple Damage Functions**: Capital, TFP, and output damage mechanisms with linear/quadratic temperature and precipitation responses
- **Robust Optimization**: L-BFGS-B with proper bounds and convergence handling
- **Flexible Scaling Modes**: Both optimization-based and direct parameter specification

### 📊 Current Datasets

| File | Countries | Years | Resolution | Size |
|------|-----------|-------|------------|------|
| `Historical_SSP1_annual.csv` | 145 | 1980-2100 | Annual | 1.2MB |
| `Historical_SSP2_annual.csv` | 146 | 1980-2100 | Annual | 1.3MB |
| `Historical_SSP3_annual.csv` | 146 | 1980-2100 | Annual | 1.3MB |
| `Historical_SSP4_annual.csv` | 146 | 1980-2100 | Annual | 1.3MB |
| `Historical_SSP5_annual.csv` | 146 | 1980-2100 | Annual | 1.3MB |

**Columns**: `country, year, population, GDP, tas, pr`

## 📈 Enhanced Visualization and Analysis Framework

### Overview
With the complete integrated processing pipeline now functional, the next development priority is comprehensive visualization and numerical analysis capabilities to evaluate and validate the climate-economic modeling results. This framework will enable systematic assessment of model outputs across multiple dimensions: spatial patterns, temporal evolution, scenario comparisons, and damage function sensitivity.

### Planned Visualization Capabilities

#### **1. Spatial Analysis and Mapping**
- **Global Impact Maps**: Visualize climate damage patterns across grid cells for different scenarios and damage functions
- **Regional Comparison Tools**: Focus analysis on specific geographic regions (continents, countries, climate zones)
- **Constraint Verification Maps**: Validate that target GDP reductions are spatially consistent and economically realistic
- **Optimization Success Analysis**: Map regions where scaling factor optimization succeeded vs. failed, with diagnostic information

#### **2. Multi-Dimensional Output Analysis**
The integrated pipeline produces complex 6D output arrays `[ssp, lat, lon, damage_func, target, time]` requiring specialized analysis tools:

- **SSP Scenario Comparison**: Side-by-side analysis of economic impacts across different socio-economic pathways
- **Damage Function Sensitivity**: Visualize uncertainty ranges across linear/quadratic capital/TFP/output damage mechanisms
- **Target GDP Reduction Analysis**: Compare spatial patterns for constant, linear, and quadratic reduction targets
- **Climate vs Weather Separation**: Quantify and visualize the pure climate signal vs. weather variability impacts

#### **3. Time Series Analysis and Trends**
- **Economic Trajectory Visualization**: Plot GDP, TFP, and capital stock evolution for selected regions and scenarios
- **Damage Function Convergence**: Visualize how different damage mechanisms lead to varying long-term economic outcomes
- **Constraint Tracking**: Monitor whether target GDP reductions are maintained throughout forward model projections
- **Baseline vs Climate-Integrated Comparison**: Three-panel plots showing baseline, weather-only, and climate-integrated scenarios

#### **4. Statistical Summary and Validation**
- **Global and Regional Aggregation**: Calculate area-weighted and GDP-weighted means for validation against literature
- **Uncertainty Quantification**: Statistical distributions across damage functions and target reduction scenarios
- **Optimization Diagnostics**: Success rates, convergence properties, and parameter scaling factor distributions
- **Economic Bounds Validation**: Verify that projected impacts remain within economically realistic ranges (no extreme GDP losses >80%)

### Implementation Strategy

#### **Phase 1: NetCDF Analysis Infrastructure**
```python
# Specialized utilities for exploring high-dimensional output arrays
analyze_6d_output(netcdf_path, analysis_type='spatial_pattern')
compare_ssp_scenarios(results_dict, metric='gdp_damage', year=2100)
extract_regional_timeseries(netcdf_path, region='North_America')
validate_constraint_satisfaction(step1_targets, step4_results)
```

#### **Phase 2: Advanced Visualization Tools**
```python
# Multi-panel comparative visualization
create_scenario_comparison_plots(results, scenarios=['ssp245', 'ssp585'])
generate_damage_function_uncertainty_fans(results, region='global')
plot_spatial_optimization_diagnostics(scaling_results, success_threshold=0.8)
create_climate_vs_weather_difference_maps(forward_results, year=2100)
```

#### **Phase 3: Automated Report Generation**
```python
# Comprehensive analysis reports
generate_model_validation_report(all_results, output_path='validation_report.pdf')
create_regional_analysis_book(results, regions=['Africa', 'Asia', 'Europe'])
produce_damage_function_comparison_summary(results, save_path='damage_comparison.pdf')
```

### Proposed Output Formats

#### **Interactive Analysis**
- **Jupyter Notebooks**: Template notebooks for exploring results interactively
- **Web-based Dashboards**: Interactive tools for stakeholders to explore scenarios
- **Parameterized Analysis**: Configurable analysis scripts for different research questions

#### **Publication-Ready Figures**
- **High-resolution Maps**: Global and regional impact visualizations with proper cartographic projections
- **Multi-panel Time Series**: Standardized plots for comparing scenarios and damage functions
- **Statistical Summary Tables**: Formatted results tables for academic publications
- **Uncertainty Visualization**: Error bars, confidence intervals, and sensitivity analysis plots

#### **Quality Assurance Reports**
- **Model Validation Summaries**: Systematic checks of economic realism and constraint satisfaction
- **Processing Diagnostics**: Success rates, computational performance, and optimization convergence
- **Data Quality Reports**: Grid cell coverage, temporal completeness, and interpolation quality assessment

### Scientific Applications

#### **Climate Impact Assessment**
- **Damage Function Comparison**: Evaluate which climate response mechanisms (capital vs TFP vs output) produce most realistic results
- **Scenario Analysis**: Quantify economic differences between SSP245 and SSP585 pathways under different damage assumptions
- **Regional Vulnerability**: Identify geographic regions most susceptible to climate-economic impacts
- **Temporal Analysis**: Understand how climate damages evolve over the 21st century

#### **Policy Analysis Support**
- **Target Setting Validation**: Verify that spatially-explicit GDP reduction targets are economically consistent
- **Mitigation Scenario Comparison**: Compare economic outcomes under different emissions pathways
- **Adaptation Strategy Assessment**: Identify regions where climate adaptation investments would be most effective
- **Uncertainty Communication**: Provide robust uncertainty ranges for policy-relevant economic projections

### Technical Requirements

#### **Data Management**
- **Large Array Handling**: Efficient processing of multi-gigabyte 6D NetCDF arrays
- **Memory Optimization**: Chunked analysis for processing larger-than-memory datasets
- **Parallel Processing**: Distribute analysis across multiple cores/nodes for large-scale studies
- **Archive Integration**: Seamless access to processed results from different model runs and timestamps

#### **Visualization Performance**
- **Interactive Responsiveness**: Fast rendering for exploratory analysis
- **High-Resolution Output**: Publication-quality figure generation
- **Batch Processing**: Automated generation of large numbers of comparative plots
- **Customization Framework**: Flexible styling and layout options for different audiences

This enhanced visualization and analysis framework will transform the COIN-SSP pipeline from a data processing tool into a comprehensive climate-economic assessment platform, enabling rigorous scientific analysis and policy-relevant insights from gridded climate-economic modeling results.

## Getting Started

### Installation
```bash
# Clone repository
git clone https://github.com/KCaldeira/coin_ssp.git
cd coin_ssp

# Create virtual environment  
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Model
The model now uses JSON configuration files for complete workflow definition:

```bash
# Process all countries using JSON configuration
python main.py coin_ssp_experiment1.json

# Process limited number of countries for testing
python main.py coin_ssp_experiment1.json --max-countries 5

# Results saved to: ./data/output/run_{run_name}_{timestamp}/
# Individual CSV files: [country]_results_{run_name}_{timestamp}.csv
# PDF books: COIN_SSP_{country}_Book_{run_name}_{timestamp}.pdf
```

### JSON Configuration Structure

The model uses JSON files named `coin_ssp_*.json` containing two main sections:

```json
{
  "model_params": {
    "year_diverge": 2025,
    "year_scale": 2100,
    "amount_scale": -0.05,
    "s": 0.3,
    "alpha": 0.3,
    "delta": 0.1
  },
  "scaling_params": [
    {
      "scaling_name": "capital_optimized",
      "k_tas2": 1.0
    },
    {
      "scaling_name": "tfp_direct", 
      "tfp_tas2": 1.0,
      "scale_factor": -0.005
    },
    {
      "scaling_name": "output_optimized",
      "y_tas2": 1.0
    }
  ]
}
```

**Model Parameters** (optional - uses defaults if not specified):
- `year_diverge`: Year when climate effects begin (default: 2025)
- `year_scale`: Target year for optimization (default: 2100)  
- `amount_scale`: Target climate impact on GDP (e.g., -0.05 for 5% loss)
- Economic parameters: `s`, `alpha`, `delta`

**Scaling Parameters** (required list):
- `scaling_name`: Unique identifier for this parameter set
- `scale_factor`: Optional - if provided, uses this value directly instead of optimization
- Climate sensitivity parameters: `k_tas1/2`, `tfp_tas1/2`, `y_tas1/2`, `k_pr1/2`, `tfp_pr1/2`, `y_pr1/2`

### Output Files
The model generates timestamped output in `./data/output/run_{run_name}_{timestamp}/`:

1. **Individual CSV files**: `[Country_Name]_results_{run_name}_{timestamp}.csv` - Complete time series data for each country:
   - **Common columns**: `year`, `population`, `gdp_observed`, `tas`, `pr`, `tas_weather`, `pr_weather`, `tfp_baseline`, `k_baseline`
   - **Per-scaling columns**: `gdp_climate_{scaling_name}`, `tfp_climate_{scaling_name}`, `k_climate_{scaling_name}`, `gdp_weather_{scaling_name}`, etc.

2. **PDF Books**: `COIN_SSP_{Country}_Book_{run_name}_{timestamp}.pdf` - One book per country:
   - **One page per scaling set** within each country book
   - Each page shows three panels: GDP, TFP, Capital Stock
   - Each panel shows: Baseline vs Climate vs Weather projections
   - Page title: `{Country} - {scaling_name}`
   - Info box shows optimization results and target parameters

### Scaling Modes

The model supports two modes for determining scale factors:

**1. Optimization Mode** (when `scale_factor` is not provided):
- Automatically finds the optimal scale factor to achieve target climate impact
- Uses `year_scale` and `amount_scale` from model parameters
- Shows: `"Optimizing climate response scaling..." → "Scale factor: X, error: Y"`

**2. Direct Mode** (when `scale_factor` is provided):
- Uses the specified scale factor directly, skipping optimization
- Faster execution for testing or when optimal values are known
- Shows: `"Using provided scale factor: X"`

### Workflow Structure
The model processes data with the following structure:
- **Outer loop**: Countries (optimized for comparing scaling sets within countries)
- **Inner loop**: Scaling parameter sets
- **Scale determination**: Either optimized or directly specified per scaling set
- **Three scenarios**: Baseline (no climate), Climate (full trends), Weather (variability only)

### Usage Example
```python
from coin_ssp_core import ModelParams, calculate_tfp_coin_ssp, calculate_coin_ssp_forward_model
import pandas as pd

# Load country data
data = pd.read_csv('./data/input/Historical_SSP2_annual.csv')
country_data = data[data.country == 'United States'].sort_values('year')

# Set up parameters
params = ModelParams(
    s=0.3,          # 30% savings rate
    alpha=0.3,      # Capital elasticity  
    delta=0.1,      # 10% depreciation
    tas0=20.0,      # Reference temperature
    tfp_tas2=-0.01  # Quadratic temperature sensitivity for TFP
)

# Calculate baseline TFP (no climate effects)
tfp, capital = calculate_tfp_coin_ssp(
    country_data.population.values,
    country_data.GDP.values, 
    params
)

# Run climate-integrated model
gdp_climate, tfp_climate, k_climate, climate_factors = calculate_coin_ssp_forward_model(
    tfp,
    country_data.population.values,
    country_data.GDP.values, 
    country_data.tas.values,
    params
)
```

## Model Features

### ✅ Current Capabilities
- **Country-level Analysis**: 146 countries with complete 1980-2100 time series
- **Multi-scenario Support**: All 5 SSP economic scenarios
- **Climate Integration**: Temperature and precipitation damage functions with automatic parameter calibration
- **Gridded Data Processing**: NetCDF utilities for spatial climate and economic data with standardized naming conventions
- **Target GDP Reduction System**: Spatially-explicit constraint satisfaction with GDP-weighted global means
- **Unified Configuration**: Integrated JSON schema combining climate models, damage functions, targets, and SSP scenarios
- **Integrated Pipeline Step 1**: Complete target GDP calculation with dynamic file resolution and JSON integration
- **Hybrid Code Architecture**: Extracted reusable functions in `coin_ssp_utils.py` with preserved standalone functionality
- **Flexible Configuration**: Multiple scaling parameter sets with both optimization and direct scaling modes
- **Robust Economic Modeling**: Negative capital stock protection and fail-fast error handling
- **Automated Visualization**: Multi-page PDF books with spatial maps and function plots
- **Optimization Framework**: L-BFGS-B optimization for achieving target economic impacts
- **Quality Assurance**: Comprehensive data validation, constraint verification, and economic bounds checking

### 🔄 Research Applications
- **Climate Impact Assessment**: Quantify economic losses under different warming scenarios with optimized damage functions
- **Policy Analysis**: Compare adaptation strategies across SSP pathways and scaling mechanisms
- **Parameter Sensitivity**: Automated scaling to achieve specific economic impact targets (e.g., 5%, 10%, 20% GDP losses)
- **Counterfactual Studies**: Climate vs. weather-only vs. baseline economic projections with visual comparison
- **Damage Function Comparison**: Capital, TFP, and output damage mechanisms with linear and quadratic responses

## Data Quality

### Economic Data Interpolation
- **Method**: Linear interpolation between 5-year SSP data points
- **Validation**: Original 5-year values preserved exactly (0.000 difference)
- **Coverage**: Historical Reference (1950-2019) + SSP scenarios (2020-2100)

### Climate Data
- **Resolution**: Annual temperature (°C) and precipitation (mm/day)
- **Sources**: Historical observations (1850-2014) + CMIP6 projections (2015-2100)
- **Processing**: Country-level spatial averages

### Country Coverage
- **Total**: 146 countries with complete data
- **Additions**: Enhanced mapping resolved 9 country name mismatches
- **Quality**: All countries have continuous annual time series across all variables

## File Structure
```
coin_ssp/
├── coin_ssp_core.py                    # Main economic model functions
├── coin_ssp_utils.py                   # Consolidated utilities (LOESS filtering + NetCDF processing)
├── calculate_target_gdp_reductions.py  # Standalone tool for gridded target reduction calculations  
├── test_target_gdp.py                  # Test script for target GDP reductions
├── main.py                             # Country-level processing pipeline
├── main_integrated.py                  # Integrated grid-cell processing pipeline (5-step workflow)
├── coin_ssp_integrated_config_example.json  # Unified configuration for complete processing workflow
├── target_gdp_config_0000.json         # Configuration for target GDP reduction calculations
├── data/
│   ├── input/                          # NetCDF gridded climate/economic data + country datasets
│   │   ├── gridRaw_tas_CanESM5_ssp585.nc      # Gridded temperature data
│   │   ├── gridRaw_pr_CanESM5_ssp585.nc       # Gridded precipitation data
│   │   ├── gridded_gdp_regrid_CanESM5.nc      # Gridded GDP projections
│   │   ├── gridded_pop_regrid_CanESM5.nc      # Gridded population projections
│   │   ├── Historical_SSP1_annual.csv
│   │   ├── Historical_SSP2_annual.csv
│   │   └── ...
│   └── output/                         # Timestamped model run results
│       ├── run_20250903_143052/        # Country-level results
│       │   ├── [country]_results_20250903_143052.csv
│       │   └── COIN_SSP_Results_Book_20250903_143052.pdf
│       ├── target_gdp_reductions.nc    # Gridded target reduction results
│       └── target_gdp_reductions_maps.pdf  # Global maps + function plots
```

## Contributing

This project follows the coding philosophy outlined in `CLAUDE.md`:
- **Elegant, fail-fast code** that surfaces errors quickly
- **Minimal input validation** - let exceptions bubble up naturally  
- **Mathematical clarity** over defensive programming
- **Numpy vectorization** preferred over loops and conditionals

## License

This project is designed for climate economics research and follows open science principles.
