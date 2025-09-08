#!/usr/bin/env python3

import xarray as xr
import numpy as np

def load_gridded_data():
    """
    Load all four NetCDF files and return as a unified dataset.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'tas': temperature data (axis_0, lat, lon) 
        - 'pr': precipitation data (axis_0, lat, lon)
        - 'gdp': GDP data (axis_0, lat, lon) 
        - 'pop': population data (time, lat, lon)
        - 'lat': latitude coordinates
        - 'lon': longitude coordinates
        - 'tas_years': temperature time axis (0-85, assuming 2015-2100)
        - 'pr_years': precipitation time axis (0-85, assuming 2015-2100) 
        - 'gdp_years': GDP time axis (0-17, assuming 5-year intervals)
        - 'pop_years': population time axis (years since reference)
    """
    
    # Load temperature data
    tas_ds = xr.open_dataset('data/input/gridRaw_tas_CanESM5_ssp585.nc', decode_times=False)
    tas = tas_ds.tas.values - 273.15  # Convert from Kelvin to Celsius
    
    # Load precipitation data  
    pr_ds = xr.open_dataset('data/input/gridRaw_pr_CanESM5_ssp585.nc', decode_times=False)
    pr = pr_ds.pr.values  # (86, 64, 128)
    
    # Load GDP data
    gdp_ds = xr.open_dataset('data/input/gridded_gdp_regrid_CanESM5.nc', decode_times=False)
    gdp = gdp_ds.gdp_20202100ssp5.values  # (18, 64, 128)
    
    # Load population data
    pop_ds = xr.open_dataset('data/input/gridded_pop_regrid_CanESM5.nc', decode_times=False)
    pop = pop_ds.pop_20062100ssp5.values  # (95, 64, 128)
    
    # Get coordinate arrays
    lat = tas_ds.lat.values
    lon = tas_ds.lon.values
    
    # Create year arrays (assuming standard time conventions)
    tas_years = np.arange(2015, 2015 + len(tas_ds.axis_0))  # 2015-2100
    pr_years = np.arange(2015, 2015 + len(pr_ds.axis_0))    # 2015-2100
    gdp_years = np.arange(2020, 2020 + len(gdp_ds.axis_0) * 5, 5)  # 2020-2100 in 5-year steps
    pop_years = np.arange(2006, 2006 + len(pop_ds.time))     # 2006-2100
    
    return {
        'tas': tas,
        'pr': pr, 
        'gdp': gdp,
        'pop': pop,
        'lat': lat,
        'lon': lon,
        'tas_years': tas_years,
        'pr_years': pr_years,
        'gdp_years': gdp_years,
        'pop_years': pop_years
    }

def calculate_area_weights(lat):
    """
    Calculate area weights proportional to cosine of latitude.
    
    Parameters
    ---------- 
    lat : array
        Latitude coordinates in degrees
        
    Returns
    -------
    array
        Area weights with same shape as lat, normalized to sum to 1
    """
    weights = np.cos(np.radians(lat))
    return weights / weights.sum()

def calculate_time_means(data, years, start_year, end_year):
    """
    Calculate temporal mean over specified year range.
    
    Parameters
    ----------
    data : array
        Data array with time as first dimension
    years : array  
        Year values corresponding to time dimension
    start_year : int
        Start year (inclusive)
    end_year : int
        End year (inclusive)
        
    Returns
    -------
    array
        Time-averaged data with time dimension removed
    """
    mask = (years >= start_year) & (years <= end_year)
    return data[mask].mean(axis=0)

def calculate_global_mean(data, lat):
    """
    Calculate area-weighted global mean.
    
    Parameters
    ----------
    data : array
        2D spatial data (lat, lon) 
    lat : array
        Latitude coordinates
        
    Returns
    -------
    float
        Area-weighted global mean
    """
    weights = calculate_area_weights(lat)
    # Expand weights to match data shape: (lat,) -> (lat, lon)  
    weights_2d = np.broadcast_to(weights[:, np.newaxis], data.shape)
    return np.average(data, weights=weights_2d)