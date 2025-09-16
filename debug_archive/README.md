# Debug Archive Organization

This directory contains debug, test, and helper files that were moved from the main project directory to keep it clean and focused on production code.

## Directory Structure

### `completed_fixes/`
One-time fix scripts that have completed their purpose:
- `fix_gridraw_time_axis.py` - Fixed time coordinates in GridRaw files (axis_0 → time)
- `fix_gridded_density_time_axis.py` - Fixed population files with corrected time axis

### `old_tests/`
Test and verification scripts that have been superseded by integrated functionality:
- `test_temporal_alignment.py` - Temporal alignment testing (now integrated in main pipeline)
- `test_tfp.py` + `tfp_test_results.png` - TFP calculation testing (now working in integrated pipeline)
- `verify_outputs.py` - Basic output verification (superseded by integrated validation)

### `data_prep_scripts/`
One-time data preparation and exploration scripts:
- `create_*.py` - Various dataset creation scripts
- `download_ssp_data.py` - SSP data downloading
- `summarize_ssp_data.py` - Data summarization
- `inspect_ssp_data.py` - Data inspection utilities

### Root debug files (existing)
- `debug_*.py` - Various debugging scripts from development sessions
- `*.nc` files - Debug NetCDF outputs
- `*.pdf` - Debug visualization outputs

## Rationale

These files were moved to maintain a clean main directory focused on production code:
- `coin_ssp_core.py` - Core economic model functions
- `coin_ssp_utils.py` - Utility functions
- `main_integrated.py` - Integrated 5-step processing pipeline
- `main.py` - Country-level workflow
- `calculate_target_gdp_reductions.py` - Standalone target GDP tool
- `model_params_factory.py` - Parameter management factory

The debug archive preserves all development history while keeping the working directory clean for production use.