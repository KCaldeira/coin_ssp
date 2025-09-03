# COIN_SSP: Climate-Integrated Economic Growth Model

A country-level implementation of the Solow-Swan growth model designed to assess climate impacts on economic growth under various SSP scenarios and climate trajectories.

## Project Overview

This project implements a forward-looking Solow-Swan growth model that integrates:

- **SSP Economic Scenarios** (GDP and population projections)
- **Annual Climate Data** (temperature and precipitation time series)  
- **Climate Damage Functions** with multiple response mechanisms
- **Total Factor Productivity** calculations from observed economic data

The model processes country-level data to understand how climate change affects economic growth through impacts on capital stock, productivity, and output.

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
- **Climate Data**: Historical (1850-2014) + SSP585 (2015-2100) annual temperature/precipitation
- **Economic Data**: SSP scenarios with GDP and population (5-year intervals, 1950-2100)
- **Coverage**: 146 countries with complete data across all sources

### 2. Data Harmonization
- **Country Name Mapping**: Resolves naming differences between climate and economic datasets
- **Annual Interpolation**: Linear interpolation of economic data to match annual climate resolution
- **Quality Assurance**: Removes countries with incomplete time series

### 3. Economic Processing
**Step 1: Baseline TFP Calculation**
```python
tfp_baseline, k_baseline = calculate_tfp_coin_ssp(population, gdp, params)
```

**Step 2: Climate-Integrated Forward Model**  
```python
results = calculate_coin_ssp_forward_model(tfp_baseline, population, gdp, temperature, params)
```

## Implementation Status

### ✅ Completed Components

#### Data Infrastructure
- **SSP Data Download** (`download_ssp_data.py`): Automated retrieval of economic scenarios
- **Data Harmonization** (`create_annual_datasets.py`): Country mapping and annual interpolation
- **Quality Datasets**: 5 Historical/SSP scenario files with 146 countries, annual resolution (1980-2100)

#### Core Model
- **Economic Core** (`coin_ssp_core.py`): 
  - `ModelParams` dataclass with all economic and climate sensitivity parameters
  - `calculate_tfp_coin_ssp()` - baseline TFP from observed GDP/population
  - `calculate_coin_ssp_forward_model()` - climate-integrated economic projections
- **Utilities** (`coin_ssp_utils.py`): Time series filtering (LOESS) for climate smoothing
- **Main Pipeline** (`main.py`): Complete country-by-country processing workflow

#### Testing & Validation
- **TFP Testing** (`test_tfp.py`): Synthetic 20-year validation scenarios
- **Data Verification**: Interpolation quality checks, country coverage analysis

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
```bash
# Process all countries for SSP5 scenario (default: no climate effects)
python main.py

# Process limited number of countries for testing
python main.py --max-countries 1          # Process only first country
python main.py --max-countries 5          # Process first 5 countries

# Run with climate sensitivity parameters
python main.py --tfp_tas2 -0.01           # Quadratic temperature sensitivity for TFP
python main.py --k_tas2 -0.005 --y_tas2 -0.02  # Multiple climate parameters

# See all available options
python main.py --help

# Results saved to timestamped directory: ./data/output/run_YYYYMMDD_HHMMSS/
# Individual CSV files: [country]_results_YYYYMMDD_HHMMSS.csv
# PDF summary book: COIN_SSP_Results_Book_YYYYMMDD_HHMMSS.pdf
```

### Climate Sensitivity Parameters
All climate damage parameters default to 0.0 (no climate effects). Available options:

**Temperature Sensitivity:**
- `--k_tas1`, `--k_tas2`: Linear/quadratic temperature effects on capital stock
- `--tfp_tas1`, `--tfp_tas2`: Linear/quadratic temperature effects on TFP growth  
- `--y_tas1`, `--y_tas2`: Linear/quadratic temperature effects on output

**Precipitation Sensitivity:**
- `--k_pr1`, `--k_pr2`: Linear/quadratic precipitation effects on capital stock
- `--tfp_pr1`, `--tfp_pr2`: Linear/quadratic precipitation effects on TFP growth
- `--y_pr1`, `--y_pr2`: Linear/quadratic precipitation effects on output

### Output Files
The model generates timestamped output in `./data/output/run_YYYYMMDD_HHMMSS/`:

1. **Individual CSV files**: `[Country_Name]_results_YYYYMMDD_HHMMSS.csv` - Complete time series data for each country containing:
   - Baseline economic variables (GDP, TFP, capital stock)
   - Climate-affected projections (with full climate trends)
   - Weather-only projections (interannual variability, trends removed after 2025)
   - Climate effect factors and reference climate data

2. **PDF Summary Book**: `COIN_SSP_Results_Book_YYYYMMDD_HHMMSS.pdf` - Visual summary with one page per country showing:
   - **GDP Panel**: Baseline vs Climate vs Weather projections
   - **TFP Panel**: Baseline vs Climate vs Weather projections  
   - **Capital Stock Panel**: Baseline vs Climate vs Weather projections

Each model run creates a separate timestamped subdirectory, allowing you to:
- Compare results from different parameter configurations
- Maintain a complete history of model runs  
- Easily organize and archive results

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
- **Climate Integration**: Temperature and precipitation damage functions
- **Flexible Parameters**: Configurable economic and climate sensitivities
- **Quality Assurance**: Comprehensive data validation and interpolation

### 🔄 Research Applications
- **Climate Impact Assessment**: Quantify economic losses under different warming scenarios
- **Policy Analysis**: Compare adaptation strategies across SSP pathways  
- **Uncertainty Quantification**: Sensitivity analysis of climate damage parameters
- **Counterfactual Studies**: Climate vs. weather-only vs. baseline economic projections

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