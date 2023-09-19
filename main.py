import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

from Stock import Stock
from StockDataFetcher import YahooFinanceFetcher

commonStocks = ['MSFT', 'NVDA', 'AAPL', 'NFLX', 'TSLA', 'AMZN', 'FB', 'GOOG', 'ADBE']
comparisonIndex = 'SPY'

data_fetcher = YahooFinanceFetcher()

# Create comparison index and single stock objects
SPY_daily = Stock('SPY', data_fetcher, '360d', '1d')
AAPL_daily = Stock('AAPL', data_fetcher, '360d', '1d')

# Prepare data for linear regression
def prepareLinearRegressionData(stock_df, comparison_index_df):
    df = stock_df.copy()
    df['relStr'] = df['pct_move_from_lastBar'] - comparison_index_df['pct_move_from_lastBar']
    for i in range(1, 8):
        df['relStr_lag_' + str(i)] = df['relStr'].shift(i)

    df = df.dropna()
    df = sm.add_constant(df)

    return df

def linearRegressionModel(df):
    X = df[['const', 'relStr_lag_1', 'relStr_lag_2', 'relStr_lag_3', 'relStr_lag_4', 'relStr_lag_5', 'relStr_lag_6', 'relStr_lag_7', 'Volume']]
    Y = df['Close']
    model = sm.OLS(Y, X)
    result = model.fit()
    prediction = result.predict(X)
    return Y, prediction, result

def plotGraph(Y, prediction, first_deviation_range, second_deviation_range, third_deviation_range):
    plt.plot(Y, label='Actual Price')
    plt.plot(prediction, label='Predicted Price')
    plt.fill_between(Y.index, first_deviation_range[0], first_deviation_range[1], alpha=0.3, label='1st Deviation')
    plt.fill_between(Y.index, second_deviation_range[0], second_deviation_range[1], alpha=0.3, label='2nd Deviation')
    plt.fill_between(Y.index, third_deviation_range[0], third_deviation_range[1], alpha=0.3, label='3rd Deviation')
    plt.legend()
    plt.show()

def predict_and_plot_stock(ticker):
    stock_df = daily.makeStock_df(ticker)
    daily.pct_move_from_lastBar(stock_df)
    stock_df = prepareLinearRegressionData(stock_df)
    Y, prediction, result = linearRegressionModel(stock_df)
    residuals = Y - prediction
    std_dev = residuals.std()

    # Preparing features for one period forward prediction
    future_X = stock_df.iloc[-1:][['const', 'relStr_lag_1', 'relStr_lag_2', 'relStr_lag_3', 'relStr_lag_4', 'relStr_lag_5', 'relStr_lag_6', 'relStr_lag_7', 'Volume']]
    future_prediction = result.predict(future_X).iloc[0]  # Get the single value from the Series
    future_first_deviation_range = (future_prediction - std_dev, future_prediction + std_dev)
    future_second_deviation_range = (future_prediction - 2*std_dev, future_prediction + 2*std_dev)
    future_third_deviation_range = (future_prediction - 3*std_dev, future_prediction + 3*std_dev)


    plt.plot(Y, label='Historical Closing Price')
    plt.axhline(y=future_prediction, color='red', linestyle='-', label='Predicted Price') # Removed [0]
    plt.axhline(y=future_first_deviation_range[0], color='green', linestyle='-', label='1st Deviation Low')
    plt.axhline(y=future_first_deviation_range[1], color='green', linestyle='-', label='1st Deviation High')
    plt.axhline(y=future_second_deviation_range[0], color='blue', linestyle='-', label='2nd Deviation Low')
    plt.axhline(y=future_second_deviation_range[1], color='blue', linestyle='-', label='2nd Deviation High')
    plt.axhline(y=future_third_deviation_range[0], color='purple', linestyle='-', label='3rd Deviation Low')
    plt.axhline(y=future_third_deviation_range[1], color='purple', linestyle='-', label='3rd Deviation High')
    plt.legend()
    plt.show()

# stock_to_predict = input("Enter the stock ticker to be predicted: ")
# predict_and_plot_stock(stock_to_predict)

csv = prepareLinearRegressionData(AAPL_daily.dataframe, SPY_daily.dataframe)
csv.to_csv('AAPL.csv')