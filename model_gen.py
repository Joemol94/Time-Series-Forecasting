import pandas as pd
import numpy as np
import datetime as dt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima.arima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression

# Define a list of SKU identifiers
sku_list = ['SKU001', 'SKU002', 'SKU003', ...]  # Add your SKUs here

def filter_sku(df,sku_name):
    df['date'] = pd.to_datetime(df["TRUNC(INVI_CR_DT,'MM')"])
    df['month_year'] = df['date'].dt.to_period('M')
    df = df[df['INVI_ITEM_CODE']==sku_name]
    return df
def read_data(sku):
    # Read data for the given SKU
    df = pd.read_excel(f'{sku}/{sku}_monthwise_19_23_mod.xlsx')
    df['month_year'] = pd.to_datetime(df['month_year'])
    df = df.rename(columns={'QTY': 'INVI_QTY'}).copy()
    return df

def mape_metric(test_df, predictions):
    # Calculate mean absolute percentage error (MAPE)
    mape = np.mean(np.abs((test_df['INVI_QTY'] - predictions) / test_df['INVI_QTY'])) * 100
    return mape

# Create an empty DataFrame to store results
results_df = pd.DataFrame(columns=['SKU', 'Model', 'MAPE'])

# Define your forecasting models as functions (holt_winter_model, arima_model, moving_average_arima_model, linear_regression_model)

for sku in sku_list:
    df = read_data(sku)
    n_train = len(df) - 6
    n_pred = 3
    
    train_df = df.iloc[:n_train]
    test_df = df.iloc[n_train:]
    future_months_df = pd.DataFrame({'month_year': pd.date_range(start=test_df['month_year'].iloc[-1] + pd.DateOffset(months=1), periods=n_pred, freq='MS')})
    extended_df = pd.concat([train_df, test_df])
    
    # Apply your forecasting models and store results for each SKU and model
    for model_name, model_func in [
        ('ES', holt_winter_model),
        ('ARIMA', arima_model),
        ('Moving_Average_ARIMA', moving_average_arima_model),
        ('Linear_Regression', linear_regression_model)
    ]:
        predictions = model_func(train_df, test_df, future_months_df)
        mape = mape_metric(test_df, predictions)
        results_df = results_df.append({'SKU': sku, 'Model': model_name, 'MAPE': mape}, ignore_index=True)

# Save the results to a CSV file
results_df.to_csv('all_skus_results.csv', index=False)
