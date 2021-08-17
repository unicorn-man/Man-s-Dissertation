# Import packages
import os
import urllib
import requests
import math
import scipy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import MultipleLocator
from scipy import stats
from fitter import Fitter


data_full = pd.read_csv("E://UoE//final//data//timeseries//CAMELS_GB_hydromet_timeseries_28018_19701001-20150930.csv")

data = data_full.loc[:,['date','discharge_vol']]


data.index = pd.to_datetime(data.date,format='%Y-%m-%d')



# add a year column to discharge data
data["year"] = data.date.apply(lambda x: pd.to_datetime(x).strftime('%Y'))

# Calculate annual max by resampling
data_annual_max = data.resample('AS').max()

data_annual_max.index = pd.DatetimeIndex(data_annual_max.index).year




# Create a function from the workflow below to calculate probability and return period

# Add an argument for annual vs daily...

def calculate_return(df, colname):

    df_without_NaN =df.dropna(axis=0)

    # Sort data smallest to largest
    sorted_data = df_without_NaN.sort_values(by=colname)
    
    # Count total obervations
    n = sorted_data.shape[0]
    
    # Add a numbered column 1 -> n to use in return calculation for rank
    sorted_data.insert(0, 'rank', range(1, 1 + n))
    
    # Calculate probability
    sorted_data["probability"] = (n - sorted_data["rank"] + 1) / (n + 1)
    
    # Calculate return - data are daily to then divide by 365?
    sorted_data["return-years"] = (1 / sorted_data["probability"])

    return(sorted_data)


data_prob = calculate_return(data, "discharge_vol")

# Because these data are daily,
# divide return period in days by 365 to get a return period in years
data_prob["return-years"] = data_prob["return-years"] / 365
data_prob["probability"] = data_prob["probability"] * 365
print(data_prob.tail())





data_annual_max_prob = calculate_return(data_annual_max, "discharge_vol")
print(data_annual_max_prob.tail())




# Look for an appropriate distribution of annual maximum data

f = Fitter(data_annual_max_prob["discharge_vol"], distributions=['t', 'logistic','lognorm','exponweib','cauchy','gamma','pareto'])
f.fit()

f.summary()

print(f.get_best(method='sumsquare_error'))




# The flow in a certain return period is calculated according to the optimal distribution obtained above

print(stats.kstest(data_annual_max_prob["discharge_vol"], 'gamma', args=(data_annual_max_prob["discharge_vol"].mean(),data_annual_max_prob["discharge_vol"].std())))

param = stats.gamma.fit(data_annual_max_prob["discharge_vol"])

discharge = stats.gamma.ppf(1-1/15,param[0],param[1],param[1])

print(discharge)



