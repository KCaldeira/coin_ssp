#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from coin_ssp_core import calculate_tfp_coin_ssp, ModelParams

def test_tfp_calculation():
    """Test the TFP calculation with synthetic data over 20 years."""
    
    # Create 20-year time series
    years = 20
    
    # GDP starts at $1000, grows at 3% per year
    gdp = 1000 * (1.03 ** np.arange(years))
    
    # Population starts at 10, grows at 1% per year
    pop = 10 * (1.01 ** np.arange(years))
    
    # Updated parameters
    params = ModelParams(
        s=0.3,       # savings rate (30%)
        alpha=0.3,   # elasticity of output w.r.t. capital
        delta=0.1    # depreciation rate (10% per year)
    )
    
    # Calculate TFP and capital stock
    a, k = calculate_tfp_coin_ssp(pop, gdp, params)
    
    # Print results
    print("Year\tGDP\t\tPop\t\tTFP\t\tCapital")
    print("-" * 60)
    for i in range(years):
        print(f"{i}\t{gdp[i]:.0f}\t\t{pop[i]:.0f}\t\t{a[i]:.3f}\t\t{k[i]:.3f}")
    
    # Plot results
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    years_array = np.arange(years)
    
    ax1.plot(years_array, gdp)
    ax1.set_title('GDP Over Time')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('GDP ($)')
    
    ax2.plot(years_array, pop)
    ax2.set_title('Population Over Time')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Population')
    
    ax3.plot(years_array, a)
    ax3.set_title('Total Factor Productivity (Normalized)')
    ax3.set_xlabel('Year')
    ax3.set_ylabel('TFP (A/A₀)')
    
    ax4.plot(years_array, k)
    ax4.set_title('Capital Stock (Normalized)')
    ax4.set_xlabel('Year')
    ax4.set_ylabel('Capital (K/K₀)')
    
    plt.tight_layout()
    plt.savefig('tfp_test_results.png', dpi=150, bbox_inches='tight')
    print(f"\nPlots saved as 'tfp_test_results.png'")
    
    return a, k, gdp, pop

if __name__ == "__main__":
    test_tfp_calculation()