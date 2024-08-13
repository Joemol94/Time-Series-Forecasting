import pandas as pd
import numpy as np
from datetime import datetime,timedelta
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import plotly.express as px
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima.arima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')
import itertools
import cx_Oracle

cx_Oracle.init_oracle_client(lib_dir=r"C:\instantclient_21_8")
dsn_tns = cx_Oracle.makedsn('192.168.1.37', '1521', service_name='ORCL') 
conn = cx_Oracle.connect(user='orion10', password='orion10', dsn=dsn_tns)
c = conn.cursor()
print('connected')

def get_data_from_DB():
    end_date = (datetime.now() - timedelta(days=datetime.now().day)).strftime('%Y-%m-%d')
    print('End date: ',end_date)
    query_2 = f"SELECT TRUNC(INVI_CR_DT, 'MM'),INVI_FLEX_09,INVI_ITEM_CODE,INVI_ITEM_DESC, \
            SUM(INVI_QTY) AS qty \
            FROM ORION10.OT_INVOICE_ITEM \
            WHERE TRUNC(INVI_CR_DT,'MM') BETWEEN to_date('2019-01-01', 'YYYY-MM-DD') AND to_date('{end_date}', 'YYYY-MM-DD') \
            GROUP BY TRUNC(INVI_CR_DT, 'MM'),INVI_FLEX_09,INVI_ITEM_CODE,INVI_ITEM_DESC "

    c.execute(query_2)
    print('query executed')
    df_1 = pd.DataFrame(c.fetchall())
    df_1.columns = [x[0] for x in c.description]
    uniq_sku_list = df_1['INVI_ITEM_CODE'].unique()
    return df_1,uniq_sku_list

def filter_sku_n_agg_monthly(df,sku_name):
    df['date'] = pd.to_datetime(df["TRUNC(INVI_CR_DT,'MM')"])
    df['month_year'] = df['date'].dt.to_period('M')
    df = df[df['INVI_ITEM_CODE']==sku_name]
    monthly_data = df.groupby('month_year')['QTY'].sum().reset_index()
    if len(monthly_data)!=57:
        min_month = df['month_year'].min()
        max_month = df['month_year'].max()
        all_months = pd.period_range(min_month, max_month, freq='M')
        # Merge the all_months range with the monthly_data to fill in missing months with zero quantities sold
        monthly_data = monthly_data.merge(pd.DataFrame({'month_year': all_months}), how='right', on='month_year').fillna(0)
        monthly_data = monthly_data.sort_values('month_year')
    return monthly_data

def data_prep(monthly_data):
    df = monthly_data.copy()
    df_1 = df.rename(columns={'QTY': 'INVI_QTY'}).copy()
    n_train = len(df_1) - 6
    n_test = 6
    n_pred = 3
    train_df = df_1.iloc[:n_train]
    test_df = df_1.iloc[n_train:]
    last_month = df['month_year'].iloc[-1].to_timestamp()
    future_months = pd.date_range(start=last_month + pd.DateOffset(months=1) , periods=n_pred, freq='MS')
    future_months_df = pd.DataFrame({'month_year': future_months})
    extended_df = pd.concat([train_df, test_df])    
    future_data = pd.DataFrame({'INVI_QTY': [None] * len(future_months), 'month_year': future_months})
    extended_df = pd.concat([extended_df, future_data]).reset_index(drop=True)
    extended_df['INVI_QTY'].fillna(0, inplace=True)
    result_df = test_df.copy()
    return train_df,test_df,extended_df,future_months_df,n_pred,result_df,future_months

def mape_metric(test_df,predictions):
    mape = np.mean(np.abs((test_df['INVI_QTY'] - predictions) / test_df['INVI_QTY'])) * 100
    return mape

def holt_winter_model(train_df,test_df,n_pred):
    best_mse = float('inf')
    best_smoothing_level = None
    smoothing_level_values = [0.1, 0.2, 0.3, 0.4, 0.5,0.6,0.7,0.8]
    smoothing_slope_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    for smoothing_level in smoothing_level_values:
        for smoothing_slope in smoothing_slope_values:
            model = ExponentialSmoothing(train_df['QTY'], trend='add', seasonal=None, seasonal_periods=None)
            model_fit = model.fit(smoothing_level=smoothing_level,smoothing_slope = smoothing_slope)
            predictions = model_fit.forecast(len(test_df)+n_pred)
            mse = ((test_df['QTY'] - predictions[:len(test_df)]) ** 2).mean()
            if mse < best_mse:
                best_mse = mse
                best_smoothing_level = smoothing_level
                best_smoothing_slope= smoothing_slope
    print(f"Best smoothing_level: {best_smoothing_level}, Best MSE: {best_mse}")
    print(f"Best smoothing_slope: {best_smoothing_slope}")
    model = ExponentialSmoothing(train_df['QTY'], trend='add', seasonal=None, seasonal_periods=None)
    model_fit = model.fit(smoothing_level=best_smoothing_level, smoothing_slope=best_smoothing_slope)
    predictions = model_fit.forecast(len(test_df)+n_pred)
    return predictions

def check_stationarity(time_series, rolling_window=12):
    adf_test = adfuller(time_series, autolag='AIC')
    print('Results of Augmented Dickey-Fuller Test:')
    adf_results = pd.Series(adf_test[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in adf_test[4].items():
        adf_results['Critical Value (%s)' % key] = value
    print(adf_results)
    if adf_results['p-value'] < 0.05 and adf_results['Test Statistic'] < adf_results['Critical Value (5%)']:
        print("The series is stationary.")
    else:
        print("The series is not stationary.")

def apply_differencing(time_series, d=1):
    differenced_series = time_series.diff(periods=d).dropna()
    return differenced_series

def arima_model(train_df,test_df,n_pred):
    model = auto_arima(train_df['QTY'], start_p=0, start_q=0, max_p=5, max_q=5, d=2,suppress_warnings=True, seasonal=False,
                       stepwise=True, trace=True,error_action='ignore',  )
    model.fit(train_df['QTY'])
    preds = model.predict(n_periods=len(test_df)+n_pred)
    return preds

def moving_average_arima_model(train_df, test_df,n_pred,orders, window_size=3):
    train_df['sma'] = train_df['QTY'].rolling(window=window_size, min_periods=1).mean()    
    model_ma = ARIMA(train_df['sma'], order=orders).fit()
    preds = model_ma.predict(start=len(train_df), end=len(train_df) + len(test_df)+n_pred - 1)
    return preds
def pearson_correlation(x, y):
    x = np.array(x)
    y = np.array(y)
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    numerator = np.sum((x - mean_x) * (y - mean_y))
    denominator = np.sqrt(np.sum((x - mean_x)**2) * np.sum((y - mean_y)**2))
    r = numerator / denominator
    return r

def linear_regression_model(train_df, test_df, future_months_df):
    model_lr = LinearRegression()    
    X_train = train_df['Date'].astype('int64').values.reshape(-1, 1)
    y_train = train_df['QTY']
    model_lr.fit(X_train, y_train)
    X_test = test_df['Date'].astype('int64').values.reshape(-1, 1)
    y_test_actual = test_df['QTY']
    preds_test = model_lr.predict(X_test)
    future_months_df['Date'] = future_months_df['Date'].dt.to_period('M')      
    X_future = future_months_df.astype('int64').values.reshape(-1, 1)
    future_preds = model_lr.predict(X_future)
    return preds_test,future_preds


def sarima_model(train_df, n_pred):
  p_range = range(0, 5) 
  d_range = range(0, 2) 
  q_range = range(0, 5) 
  min_aic = float('inf')
  best_model = None
  a,b,c = 0,0,0
  # Loop through all possible parameter combinations
  for p in p_range:
    for d in d_range:
      for q in q_range:
        model = SARIMAX(train_df['QTY'], order=(p, d, q), seasonal_order=(1, 1, 1, 12))
        model.initialize_approximate_diffuse()
        results = model.fit()
        aic = results.aic
        if aic < min_aic:
          min_aic = aic
          best_model = model
          a,b,c = p,d,q

  print(f"Best P: {p}, Best Q: {q}, Best D: {d}")
  print(f"Best P: {a}, Best Q: {c}, Best D: {b}")
  best_model = best_model.fit(initialization='approximate_diffuse')
  pred_sarima = best_model.forecast(len(test_df) + n_pred)
  return pred_sarima


db_dump,sku_list = get_data_from_DB()
print('dump collected')
for sku_name in sku_list:
    monthly_data = filter_sku_n_agg_monthly(db_dump,sku_name)
    print('filtered and aggregated')
    train_df,test_df,extended_df,future_months_df,n_pred,results_df,future_months = data_prep(monthly_data)

#Exponential smoothing
predictions_ES = holt_winter_model(train_df,test_df,n_pred)
print(predictions_ES)
training_pred_ES = predictions_ES[:len(test_df)]
future_pred_ES = predictions_ES[len(test_df):]
mape_metric(test_df,training_pred_ES)

check_stationarity(train_df['QTY'], rolling_window=12)

differenced_series = apply_differencing(train_df['QTY'], d=1)
check_stationarity(differenced_series, rolling_window=12)

#ARIMA
predictions_arima = arima_model(train_df,test_df,n_pred)
print(predictions_arima)
training_pred_arima = predictions_arima[:len(test_df)]
future_pred_arima = predictions_arima[len(test_df):]
mape_metric(test_df,training_pred_arima)

best_aic = float('inf')
best_order = None

p_values = range(0, 5)
d_values = range(0, 2)
q_values = range(0, 5)
pdq_combinations = list(itertools.product(p_values, d_values, q_values))

for order_num in pdq_combinations:
    try:
        train_df['sma'] = train_df['QTY'].rolling(window=6, min_periods=1).mean()
        model_fit = ARIMA(train_df['sma'], order=order_num).fit()
        aic = model_fit.aic
        if aic < best_aic:
            best_aic = aic
            best_order = order_num
    except:
        continue

print(f"Best ARIMA Order: {best_order}, Best AIC: {best_aic}")

predictions_ma = moving_average_arima_model(train_df,test_df,n_pred,orders = (1,1,0), window_size=6)
print(predictions_ma)
training_pred_ma = predictions_ma[:len(test_df)]
future_pred_ma = predictions_ma[len(test_df):]

mape_metric(test_df,training_pred_ma)


df_corr = df.copy()
df_corr['Date_numeric'] = range(1, len(df) + 1)
correlation_coefficient = pearson_correlation(df_corr['Date_numeric'], df_corr['QTY'])
print(f"Pearson's correlation coefficient: {correlation_coefficient}")

#Linear Regression
training_pred_lr,future_pred_lr = linear_regression_model(train_df,test_df,future_months_df)
predictions_lr = np.concatenate((training_pred_lr, future_pred_lr))
predictions_lr
mape_metric(test_df,training_pred_lr)

#SARIMA
predictions_sarima = sarima_model(train_df, n_pred)
training_pred_sarima = predictions_sarima[:len(test_df)]
future_pred_sarima = predictions_sarima[len(test_df):]
print(predictions_sarima)
mape_metric(test_df,training_pred_sarima)
