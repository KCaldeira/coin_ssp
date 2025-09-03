# COIN_SSP: Forward Solow-Swan Growth Model for Climate Impact Assessment

A gridded implementation of the Solow-Swan growth model designed to assess climate impacts on economic growth under various SSP scenarios and RCP climate trajectories.

## Project Overview

This project implements a forward-looking Solow-Swan growth model that runs at the land grid cell level of climate models. The model integrates:

- **SSP GDP and population scenarios** (Shared Socioeconomic Pathways)
- **RCP climate trajectories** (Representative Concentration Pathways) 
- **Climate damage functions** with multiple response mechanisms
- **Weather variability** considerations in model tuning

## Model Framework

### Core Parameters
Economic parameters are derived from the most recent DICE model with fixed savings rate, except for **Total Factor Productivity (TFP)**, which is tuned at each model grid cell.

### Climate Response Functions

Three types of climate damage mechanisms are considered:

1. **Damage to Production**: Direct loss to output
2. **Damage to Capital Stock**: Affects future years but recovers on depreciation timescale
3. **Damage to Total Factor Productivity**: Directly affects all future years

All damage functions can be calibrated to equivalent magnitude effects (e.g., 20% GDP loss under SSP585).

## Processing Workflow

### 1. No-Climate-Change Counterfactual
- Apply 30-year smoothing through SSP5/RCP simulations
- After 2015: subtract difference between 30-year moving average for target year and 2015
- Preserves weather variability while removing mean climate change signal

### 2. Grid-Level TFP Time Series
- Calculate TFP from historical GDP and SSPs, ignoring weather variability
- Same weather variability applied to both climate and no-climate scenarios
- TFP time series remains independent of weather-response function used

### 3. Weather Variability Integration

Two implementation options:

**Option 1 (Simpler)**: 
- Single TFP time series per SSP
- Same TFP used for all damage function cases
- Weather effects applied uniformly

**Option 2 (More Consistent)**:
- Unique TFP time series for each damage function/SSP combination  
- Climate damage from weather variability incorporated during TFP computation
- Post-2015: subtract 30-year moving average from CMIP6 results, add to 2015-centered mean

## Sensitivity Analysis

### Stochasticity Application
Key research question: Where to optimally apply stochastic processes?

Potential locations:
- Direct application to output
- Depreciation rates
- Alpha parameter
- Total factor productivity
- Savings rate

**Constraint**: Stochastic tuning must preserve mean/median GDP projections unchanged.

### Capital Diffusion
Models resource redistribution mechanisms (e.g., disaster relief, rural subsidies).

Implementation considerations:
- **Within-country diffusion**: Limited to national boundaries
- **Border-neutral diffusion**: Simplified global approach (easier implementation)
- Requires model retuning for each diffusion rate assumption

## Research Questions

1. What is the optimal location for applying stochastic processes in the model?
2. How should capital diffusion be implemented - within-country vs. border-neutral?
3. What additional climate response mechanisms should be considered?
4. How sensitive are results to the choice between weather variability options (1 vs. 2)?

## Implementation Status

### Core Components
- ✅ **Core Functions Module** (`coin_ssp_core.py`): Implements Solow-Swan model to derive total factor productivity time series from GDP and population data, plus other core functions
- ✅ **Test Framework** (`test_tfp.py`): Validation testing with synthetic 20-year scenarios

### Current Capabilities
- Calculates TFP time series assuming steady-state initial conditions
- Handles normalized variables for numerical stability
- Supports standard DICE model parameters (savings rate, capital elasticity, depreciation)

## Getting Started

### Installation
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Usage Example
```python
from coin_ssp_core import calculate_tfp_coin_ssp
import numpy as np

# Example parameters
params = {
    "s": 0.3,       # 30% savings rate
    "alpha": 0.3,   # Capital elasticity
    "delta": 0.1    # 10% depreciation rate
}

# Sample data (GDP and population time series)
gdp = 1000 * (1.03 ** np.arange(20))  # 3% annual growth
pop = 10 * (1.01 ** np.arange(20))    # 1% annual growth

# Calculate TFP and capital stock
tfp, capital = calculate_tfp_coin_ssp(gdp, pop, params)
```

## Model Advantages

- **Comparative Analysis**: Optimized for climate vs. no-climate difference studies
- **Consistent Framework**: Uses established DICE model parameters
- **Flexible Damage Functions**: Multiple climate impact mechanisms
- **Grid-Level Resolution**: Captures spatial heterogeneity in climate impacts
- **Scenario Integration**: Compatible with standard SSP/RCP framework