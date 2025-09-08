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
**Step 1: Grid Cell TFP Calculation**
```python
# Applied to each grid cell independently
tfp_baseline, k_baseline = calculate_tfp_coin_ssp(population_grid, gdp_grid, params)
```

**Step 2: Climate-Integrated Forward Model**  
```python
# Vectorized across all grid cells
results = calculate_coin_ssp_forward_model(tfp_baseline, population_grid, gdp_grid, temperature_grid, params)
```

**Step 3: NetCDF Output Generation**
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
- **NetCDF Data Loaders** (`coin_ssp_netcdf.py`): Reads gridded climate and economic data from `./data/input/`
  - Handles 4 NetCDF files: temperature, precipitation, GDP, population
  - Automatic Kelvin to Celsius conversion using physical constant
  - Area-weighted global means using cosine of latitude
  - Temporal averaging over user-specified year ranges

#### Target GDP Reduction System
- **Target Calculation Utility** (`calculate_target_gdp_reductions.py`): Standalone tool for creating spatially-explicit economic impact targets
  - JSON configuration system (`target_gdp_config_example.json`)
  - Three reduction types: constant, linear (temperature-dependent), quadratic (temperature-dependent)
  - GDP-weighted global constraint satisfaction 
  - NetCDF output (3×lat×lon array) saved to `./data/output/`
  - Global map visualization with red-white-blue colormap
  - Fixed color scale (-1 to +1) with actual range annotations

### ⚠️ Known Issues

#### Target GDP Reduction Algorithm
- **Temperature dependency issue**: Linear function currently shows backwards relationship (more damage at colder temperatures)
- **Root cause**: Constraint system mathematically correct but produces physically unrealistic temperature dependence  
- **Status**: Requires reformulation of constraint system to ensure damage increases with temperature
- **Mathematical details**: Extensively documented in code comments showing constraint equations and solution methods

### 🚧 Future Development

#### Next Priority Items
- **Fix temperature dependency**: Redesign linear/quadratic constraint systems for realistic climate damage patterns
- **Grid Cell Economic Model**: Extend TFP calculation to vectorized grid processing
- **Climate-Integrated Forward Model**: Apply damage functions spatially across grid cells
- **Full Pipeline Integration**: Connect target reduction tools with main economic modeling workflow

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
- **Flexible Configuration**: JSON-based workflow definition with multiple scaling parameter sets
- **Robust Economic Modeling**: Negative capital stock protection and fail-fast error handling
- **Automated Visualization**: Multi-page PDF books with three-panel charts per scaling scenario
- **Optimization Framework**: L-BFGS-B optimization for achieving target economic impacts
- **Quality Assurance**: Comprehensive data validation and interpolation

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
├── coin_ssp_core.py          # Main economic model functions
├── coin_ssp_utils.py         # LOESS time series filtering utilities  
├── main.py                   # Complete processing pipeline
├── data/
│   ├── input/                # Processed country-level datasets
│   │   ├── Historical_SSP1_annual.csv
│   │   ├── Historical_SSP2_annual.csv
│   │   └── ...
│   └── output/               # Timestamped model run results
│       ├── run_20250903_143052/
│       │   ├── [country]_results_20250903_143052.csv
│       │   └── COIN_SSP_Results_Book_20250903_143052.pdf
│       └── run_20250903_151234/
│           └── ...
```

## Contributing

This project follows the coding philosophy outlined in `CLAUDE.md`:
- **Elegant, fail-fast code** that surfaces errors quickly
- **Minimal input validation** - let exceptions bubble up naturally  
- **Mathematical clarity** over defensive programming
- **Numpy vectorization** preferred over loops and conditionals

## License

This project is designed for climate economics research and follows open science principles.