#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 19:37:35 2023

@author: yunbai
"""

import pandas as pd
import matplotlib.pyplot as plt
import pandas as np
import matplotlib.ticker as ticker

# read original data
dataset = pd.read_excel('Tourism forecasting competition II dataset.xlsx',sheet_name='data for forecasting')
dataset = dataset.set_index(dataset.iloc[:,0])
dataset.index.names = ['Date']
dataset = dataset.drop([dataset.columns[0]],axis=1)
    
# creat a null df for filling the predictions
preDf = pd.DataFrame()
preDf['Date'] = ['2023M03','2023M04','2023M05','2023M06','2023M07',
                 '2023M08','2023M09','2023M10','2023M11','2023M12','2024M01',
                 '2024M02','2024M03','2024M04','2024M05','2024M06','2024M07']
preDf = preDf.set_index('Date')

# read forecasted data and intervals
idx = ['2023M08','2023M09','2023M10','2023M11','2023M12','2024M01',
       '2024M02','2024M03','2024M04','2024M05','2024M06','2024M07']
forecast = pd.read_csv('Results_avg_with_xixi.csv')
quantile = pd.read_csv('Quantile_avg_with_xixi.csv')
forecast.index,quantile.index = idx,idx

dataset['Canada'].plot()
forecast['Canada'].plot()


# Sample data (replace this with your actual data)
time_series_data = []  # List of (country,time_series, predictions, prediction_intervals) tuples
countryList = list(dataset.columns)
for country in countryList:
    time_series_data.append((country,dataset[country],
                             forecast[country],quantile[[country+'_0.1',country+'_0.9']]))

# Define the colors for original data, predictions, and intervals
original_data_color = 'black'
predictions_color = 'red'
interval_color = 'gray'

# Create a 5x4 grid of subplots
num_rows = 5
num_cols = 4
fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 12))

# Plot each time series in a separate subplot
for idx, (country, time_series, predictions, prediction_intervals) in enumerate(time_series_data):
    row = idx // num_cols
    col = idx % num_cols
    ax = axs[row, col]

    # Plot original data
    ax.plot(time_series.index, time_series, color=original_data_color, label='Original Data')

    # Plot predictions
    ax.plot(predictions.index, predictions, color=predictions_color, label='Point Forecasts')

    # Plot prediction intervals
    ax.fill_between(prediction_intervals.index, prediction_intervals[country+'_0.1'], prediction_intervals[country+'_0.9'],
                    color=interval_color, alpha=0.3, label='Prediction Intervals')

    # Add a vertical line at the end of the original data
    last_data_index = time_series.index[-1]
    ax.axvline(x=last_data_index, color='gray')
    
    # Set subplot title (you can customize this as needed)
    ax.set_title(country)

    # Add legend to the subplot
    ax.legend()

    # Set the maximum number of x-axis ticks to 5
    ax.xaxis.set_major_locator(ticker.MaxNLocator(4))

# Adjust layout and spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()