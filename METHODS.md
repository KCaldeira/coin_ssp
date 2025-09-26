# Methods for Forward Simulations

## 1. Overview

We developed a variant of a Solow-Swann growth model of an economy, with the capability of responding to temperature and precipitation changes. We call this model COIN-SSP.

We tuned this model so that it could approximate both historical GDP growth and GDP growth as represented in the Shared Socioeconomic Pathways. We then specify various known climate response functions and simulate both the historical and future economic growth both with and without climate change, taking these known climate response functions into account.

Various econometric methods have been applied to GDP and weather variability in the real world, and used to project the economic response. In further work, we will apply some of these methods to the COIN-SSP results for the historical period, and test their predictive skill at estimating the climate response in the target period (typically, years 2080 to 2100).

## 2. The COIN-SSP Model

The core of the COIN-SSP model is a variant Solow-Swann growth model with a Cobb-Douglas production function, with flexible climate response specification that can be applied to output, capital stock, and/or the growth rate in total factor productivity.

The core model, in the absence of climate or weather effects, can be described by:

```
Y(t) = A(t) K(t)^α L(t)^(1-α)     (1)
```

and

```
dK(t)/dt = sY(t) - δK(t)     (2)
```

For capital elasticity in the production function (α) and the depreciation rate of capital (δ), we use values from Barrage and Nordhaus (2024), namely α=0.3 and δ=0.1 yr⁻¹. We use a savings rate of s = α = 0.3.

## 3. Historical Calibration of Reference Case

The actual GDP record contains the influence of weather variability and climate change. We do not want to introduce this unknown climate signal into our model calibration. Therefore, we generate a stylized proxy for economic growth in each model grid cell with no influence of climate or weather.

We calibrate the model by assuming that in our reference case that capital stock (K) grows at a constant exponential rate, and that Y(t) is equal to the historical GDP values both in the first year that data is available (t_init) and in year 2015 (t_2015):

```
K_hist(t) = K_hist(t_init) e^(k_K(t-t_init))     (3)
```

Taking the derivative of equation (3) and substituting into equation (2), and solving for k_K and K_hist(t_init), we have:

```
k_K = Log(GDP_2015/GDP_init)/(2015-t_init)     (4)
```

and

```
K_hist(t_init) = s/(δ+k_K) GDP_init     (5)
```

Because we do not want to have actual historical weather influencing our calculation, we further assume that Y(t) increases exponentially from the initial value of GDP_init to the year 2015 value. Following the logic above, this means:

```
Y_hist(t) = Y_hist(t_init) e^(k_K(t-t_init))     (6)
```

From equation (1), we then have as total factor productivity in the historical reference case:

```
A_hist(t) = Y_hist(t) / (K_hist(t)^α L_hist(t)^(1-α))     (7)
```

## 4. Calibration of Reference Case Total Factor Productivity for SSP Scenarios

If we assume at the beginning of the SSP scenario:

```
K_SSP(t_2015) = K_hist(t_2015)     (8)
```

and consider that Y_SSP(t) is provided by the SSP scenario, we can evolve K_SSP(t) through time using equation (2).

For the SSP scenarios, considering equation (1), we then have:

```
A_SSP(t) = Y_SSP(t) / (K_SSP(t)^α L_SSP(t)^(1-α))     (9)
```

## 5. Combined Historical and SSP Reference Cases

We then have for our reference cases:

```
A_ref(t) = A_hist(t) for t < 2015     (10a)
```

and

```
A_ref(t) = A_SSP(t) for t ≥ 2015     (10b)
```

Trivially,

```
K_ref(t_init) = K_hist(t_init)     (11)
```

We then use these reference values to drive our model taking into consideration weather variability and climate change, as described below.

## 6. Climate Response Functions

We consider climate response functions that involve functions of temperature and precipitation. (In practice, in the simulations described here, all of the precipitation-related variables have been set to zero.)

We consider cases in which climate can damage output, capital stock and/or the growth rate in total factor productivity.

For any experimental case, exp, where exp might be a climate-change case, or a weather-only case, we have, for each climate model grid cell, climate response factors that modify the economic components directly.

The climate response factors are calculated as:

```
y_climate = 1.0 + g(T) * (y_tas1 * T + y_tas2 * T²) - g(T_ref) * (y_tas1 * T_ref + y_tas2 * T_ref²)
          + (y_pr1 * P + y_pr2 * P²) - (y_pr1 * P_ref + y_pr2 * P_ref²)     (12)
```

```
k_climate = 1.0 + g(T) * (k_tas1 * T + k_tas2 * T²) - g(T_ref) * (k_tas1 * T_ref + k_tas2 * T_ref²)
          + (k_pr1 * P + k_pr2 * P²) - (k_pr1 * P_ref + k_pr2 * P_ref²)     (13)
```

```
tfp_climate = 1.0 + g(T) * (tfp_tas1 * T + tfp_tas2 * T²) - g(T_ref) * (tfp_tas1 * T_ref + tfp_tas2 * T_ref²)
            + (tfp_pr1 * P + tfp_pr2 * P²) - (tfp_pr1 * P_ref + tfp_pr2 * P_ref²)     (14)
```

where g(T) is a scaling function:

```
g(T(t)) = g0 + g1*T(t) + g2*T(t)²     (17)
```

This user-defined climate response scaling factor function, g(T), has units of fraction of output per degree Celsius. It indicates as a function of temperature the desired slope of an ordinary least squares fit of Y_weather as a function of T_weather; this slope is the correlation coefficient times the ratio of the standard deviations. The parameters g0, g1 and g2 may be chosen to examine various cases of interest. Positive values of g(T) would indicate a positive correlation between Y_weather and T_weather (i.e., climate benefit) and negative values would indicate a negative correlation (i.e., climate losses).

The climate response parameters in the implementation are:
- **Output responses**: y_tas1, y_tas2 (linear and quadratic temperature), y_pr1, y_pr2 (linear and quadratic precipitation)
- **Capital responses**: k_tas1, k_tas2 (linear and quadratic temperature), k_pr1, k_pr2 (linear and quadratic precipitation)
- **TFP responses**: tfp_tas1, tfp_tas2 (linear and quadratic temperature), tfp_pr1, tfp_pr2 (linear and quadratic precipitation)

## 7. Model Calibration

### 7.1 Conceptual Framework

Conceptually, to examine various combinations of climate response pathways, we introduce parameters indicating the relative values of the coefficients for the output, capital stock, and total-factor productivity growth climate response functions: r_Y, r_K and r_A.

To calibrate our model, for each combination of r_Y, r_K and r_A considered, at each grid cell, we determine the value of f0, such that when the climate response parameters are scaled appropriately, the model produces the desired economic impact targets.

This procedure assures that the relationships between output variability and temperature variability are similar across cases considering different climate response pathways.

### 7.2 Implementation: Damage Target Calibration

In practice, for damage targets, this is implemented through an optimization process that finds a scale factor α for each grid cell such that the ratio of climate-affected GDP to weather-only GDP in the target period equals (1 + target_reduction).

The optimization process:
1. For each response function scaling configuration, determines which climate parameters are non-zero
2. Uses scipy.optimize.minimize to find the optimal scale factor
3. Applies the scale factor to create scaled model parameters
4. Runs forward model simulations with both climate and weather-only forcing
5. Computes the objective function as the squared difference between the achieved and target GDP ratios

### 7.3 Implementation: Variability Target Calibration

For variability targets, the calibration uses a sophisticated four-step process:

**Step 1: Optimization for Uniform 10% GDP Loss**
- Run optimization to find scaling factors that produce uniform 10% GDP loss in target period
- Establishes baseline strength of climate-economy relationship needed for target impact

**Step 2: Forward Model Simulations with Scaled Parameters**
- Take parameters from Step 1, scaled by found factors
- Run forward model simulations for each grid cell using scaled parameters
- Generate economic projections over full time period (historical + future)

**Step 3: Weather Variability Regression Analysis**
- For each grid cell: compute regression `log(y_weather) ~ tas_weather` over historical period
- `y_weather` = weather component of GDP (detrended, LOESS-filtered economic signal)
- `tas_weather` = weather component of temperature (detrended, LOESS-filtered climate signal)
- Regression slope = fractional change in GDP per degree C of weather variability

**Step 4: Parameter Normalization by Regression Slope**
- Divide all climate response parameters from Step 1 by regression slope from Step 3
- Normalizes parameters to represent correct strength per degree of variability
- Final parameters capture both target impact magnitude AND observed weather sensitivity

As described below, for each model grid cell, we have data specifying the temperature in the historical cases (T_hist) and for each of the Shared Socioeconomic Pathway cases (T_SSP).

For each specification of climate response parameters (y_tas1, y_tas2, k_tas1, k_tas2, tfp_tas1, tfp_tas2, etc.) and T_ref, we then perform a simulation of the historical case using the forward model equations, yielding a time series of Y_hist for each model grid point. This is the data used to train the various econometric methods.

## 8. Cases Considered

### 8.1 Consideration of Different Relationships Between Output Variability and Temperature Change

We choose three target patterns for the slope of the linear regression of Y_weather against T_weather.

**Constant sensitivity case:** we consider g0 = -0.01 °C⁻¹, g1 = 0 °C⁻², and g2 = 0 °C⁻³, indicating a constant slope of 1% per degree Celsius.

**Linear sensitivity case:** we consider a case where there is a slope of zero at 10°C and 2% at 20°C, which yields g0 = 0.02 °C⁻¹, g1 = -0.002 °C⁻², and g2 = 0 °C⁻³.

**Quadratic sensitivity case:** we consider a case where there is a slope of zero at 10°C, 2% at 20°C, and a slope of 0.001 °C⁻¹ which yields g0 = 0 °C⁻¹, g1 = 0.001 °C⁻², and g2 = -0.0001 °C⁻³.

We consider three patterns of target losses at each grid cell, each designed to predict a global average GDP loss of about 10% in the year 2080 to 2100 time interval under SSP2-4.5.

**Uniform targets:** each grid cell loses 10% of its output

**Linear targets:** GDP losses are chosen such that, under SSP2-4.5, GDP losses scale linearly with T_ref and are 25% for grid cells with T_ref = 30°C, with the global mean GDP loss at 10%.

**Quadratic targets:** GDP losses are chosen such that, under SSP2-4.5, GDP losses scale quadratically with T_ref and are 75% for grid cells with T_ref = 30°C, no net-gains or losses for grid cells with T_ref = 13.5°C, and a global mean GDP loss at 10%.

### 8.2 Climate Response Function Cases

For each SSP scenario and climate model considered, we define response function scaling configurations that specify which climate response parameters are non-zero. Each scaling configuration represents a different pathway through which climate affects the economy.

The response function scalings are defined in the model configuration and typically include:

- **Output linear temperature**: y_tas1 = 1.0 (all others = 0)
- **Output quadratic temperature**: y_tas2 = 1.0 (all others = 0)
- **Capital linear temperature**: k_tas1 = 1.0 (all others = 0)
- **Capital quadratic temperature**: k_tas2 = 1.0 (all others = 0)
- **TFP linear temperature**: tfp_tas1 = 1.0 (all others = 0)
- **TFP quadratic temperature**: tfp_tas2 = 1.0 (all others = 0)

In this way, we consider independently linear and quadratic influences of temperature on output, capital stock, and the growth rate in total factor productivity.

For each grid cell and each SSP/climate-model combination, we calculate the scale factor for each response function configuration that produces the target economic impact, and we then use those scaled parameters in our simulations.

Additional scaling configurations may combine multiple response pathways, where the values from individual cases are combined and then scaled by a common factor to produce the target impact.

The number of response function cases combined with the target climate-response patterns provides multiple scenarios for each SSP/climate-model combination.

## 9. Data Sources

The COIN-SSP model requires several gridded datasets that are specified through configuration files and loaded from NetCDF format files.

### 9.1 Gridded Historical Population
Historical population density data is provided at grid cell resolution, typically covering the period from 1861 to 2014. The data is stored in NetCDF files with the variable name "pop_density" (or as specified in the configuration). Population data is used to calculate the labor input L(t) in the production function and is normalized to the initial year value for each grid cell.

### 9.2 Gridded Historical GDP
Historical GDP density data is provided at matching grid cell resolution for the same time period as population. The data is stored in NetCDF files with the variable name "gdp_density" (or as specified in the configuration). GDP data serves as the economic output Y(t) that the model calibrates to reproduce in its reference case simulations.

### 9.3 Gridded SSP Population
Future population projections under different Shared Socioeconomic Pathway (SSP) scenarios, typically covering 2015-2100. These projections are provided for multiple SSP scenarios (e.g., SSP2-4.5, SSP5-8.5) and are used to drive the forward model simulations for future economic projections.

### 9.4 Gridded SSP GDP
Future GDP projections under different SSP scenarios, matching the temporal and spatial resolution of the SSP population data. These provide the target economic trajectories that the model reproduces in its reference case before applying climate response functions.

### 9.5 Gridded Climate Data
Climate data includes both temperature and precipitation:

**Temperature (tas)**: Surface air temperature data in Kelvin or Celsius, provided for both historical and future periods under different SSP scenarios. Temperature anomalies relative to a reference period (typically 1861-1910) are used to drive climate response functions.

**Precipitation (pr)**: Precipitation data in appropriate units (kg m⁻² s⁻¹ or mm/day), provided for matching periods. While precipitation parameters are included in the model structure, they are typically set to zero in current simulations as noted in Section 6.

Climate data is obtained from climate model output (e.g., CanESM5) and provides the forcing for the climate response functions in the economic model.

### 9.6 Data Processing Requirements
All datasets must:
- Use consistent grid resolution and spatial coverage
- Follow standardized NetCDF conventions with time, latitude, and longitude dimensions
- Include appropriate metadata and coordinate information
- Be accessible through the file naming convention specified in the model configuration (e.g., CLIMATE_CanESM5_historical.nc, GDP_CanESM5_ssp245.nc)

## References

Barrage L, Nordhaus W. 2024. Policies, projections, and the social cost of carbon: results from the DICE-2023 model. PNAS 121:(13):e2312030121