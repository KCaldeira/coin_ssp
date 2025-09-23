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

JSON configuration files specify:
- **Climate Model**: NetCDF file patterns and model information
- **SSP Scenarios**: Reference SSP for calibration + forward simulation scenarios
- **GDP Targets**: Constant, linear, or quadratic reduction patterns
- **Damage Functions**: Climate response mechanisms (capital/TFP/output)
- **Model Parameters**: Solow-Swan economic parameters

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

- **`coin_ssp_core.py`**: Core economic model and optimization functions
- **`coin_ssp_utils.py`**: Data processing, visualization, and utility functions
- **`main.py`**: Integrated 5-step processing pipeline
- **Configuration**: JSON-based workflow control with unified schema

## Contributing

This project follows elegant, fail-fast coding principles:
- No input validation on function parameters
- Let exceptions bubble up naturally
- Prefer mathematical clarity over defensive checks
- Use numpy vectorization instead of loops