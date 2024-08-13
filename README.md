# Time-Series-Forecasting
## Overview
This Python script is designed for time series forecasting of monthly sales data from a database. It connects to an Oracle database, retrieves sales data for a specified SKU, preprocesses the data, and applies various forecasting models including Holt-Winters Exponential Smoothing, ARIMA, and Linear Regression. The script calculates the Mean Absolute Percentage Error (MAPE) for model evaluation and provides predictions for future sales.

## Features
**Database Connection:** Connects to an Oracle database using credentials and settings provided in a config.ini file.
**Data Extraction:** Fetches and aggregates sales data from the database for the specified SKU.
**Data Preprocessing:** Prepares data for model training and testing by splitting it into training and testing sets.
## Forecasting Models:
1. Holt-Winters Exponential Smoothing
2. ARIMA
3. Linear Regression
## Model Evaluation: 
Computes MAPE to evaluate the accuracy of each model.
Future Predictions: Provides forecasts for upcoming months.

## Prerequisites
Python 3.x
Oracle Client installed on the machine (specified in config.ini)
Required Python libraries - requirements.txt

## Setup
1. Install dependencies
2. Configure Database Connection
3. Run the script

## Notes
Ensure that the Oracle Client is correctly configured on your machine.
