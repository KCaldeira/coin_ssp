# Claude Code Style Guide for COIN_SSP

## Coding Philosophy
This project prioritizes **elegant, fail-fast code** that surfaces errors quickly rather than hiding them.

## Code Style Preferences

### Error Handling
- **No input validation** on function parameters (except for command-line interfaces)
- **No defensive programming** - let exceptions bubble up naturally
- **Fail fast** - prefer code that crashes immediately on invalid inputs rather than continuing with bad data
- **No try-catch blocks** unless absolutely necessary for program logic (not error suppression)

### Code Elegance
- **Minimize conditional statements** - prefer functional approaches, mathematical expressions, and numpy vectorization
- **Favor mathematical clarity** over defensive checks
- **Use numpy operations** instead of loops and conditionals where possible
- **Prefer concise, readable expressions** over verbose defensive code

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
This is a climate economics modeling project implementing the Solow-Swan growth model for climate impact assessment. Code should reflect the mathematical elegance of economic models while maintaining computational efficiency for country-level analysis.

## Data Processing Philosophy
- **Interpolation over Extrapolation**: Use linear interpolation between known data points rather than extrapolating beyond available data
- **Annual Resolution**: Process all time series at annual resolution for consistency between climate and economic data
- **Country Completeness**: Include countries only when they have complete time series data across all required variables
- **Quality Assurance**: Preserve exact values at original data points when interpolating (e.g., 5-year economic data points)