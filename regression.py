import pandas as pd
import numpy as np
import statsmodels.api as sm
from scanner import Scanner

comparisonStocks = ['MSFT', 'NVDA', 'AAPL', 'NFLX', 'TSLA', 'AMZN', 'FB', 'GOOG', 'ADBE']
base_stock = 'SPY'

intraday = Scanner(comparisonStocks, base_stock, period="1d", interval="1m")
daily = Scanner(comparisonStocks, base_stock, period="360d", interval="1d")

#create SPY dataframe
SPY = daily.makeStock_df("SPY")
daily.pct_move_from_lastBar(SPY)
SPY.to_pickle('C:\\Python\\simple_relativeStrength\\dataframes\\SPY.pkl')

#create dataframes for all comparisonStocks
#intraday.fill_and_pickle()

#create daily stock frames for all stocks in list, store
daily.fill_and_pickle()
NVDA = daily.read_pkl('NVDA')

#from a daily frame for a single stock, create 5 new columns for -n days relative strength
def prepFrame(df):
    df['relStr'] = df['pct_move_from_lastBar'] - SPY['pct_move_from_lastBar']

    #create lagging relative strength columns
    for i in range(1, 8):
        df['relStr_lag_'+str(i)] = df['relStr'].shift(i)

    #create column for whether or not stock is Up on that day and drop NA cells
    df['Direction'] = [1 if i > 0 else 0 for i in df['pct_move_from_lastBar']]
    df = df.dropna()
    df = sm.add_constant(df)
    return df

#setup prediction model
def predictModel(df):
    X = df[['const', 'relStr_lag_1', 'relStr_lag_2', 'relStr_lag_3', 'relStr_lag_4', 'relStr_lag_5', 'relStr_lag_6', 'relStr_lag_7', 'Volume']]
    Y = df['Direction']
    model = sm.Logit(Y, X)
    result = model.fit()
    prediction = result.predict(X)
    return Y, prediction

#create confusion matrix to present the results
def confusion_matrix(act,pred):
    predtrans = ['Up' if i> 0.5 else "Down" for i in pred]
    actuals = ['Up' if i > 0 else "Down" for i in act]
    confusion_matrix = pd.crosstab(pd.Series(actuals),pd.Series(predtrans),rownames=['Actual'],colnames=['Predicted'])
    return confusion_matrix

#temp statements to output the results
NVDA = prepFrame(NVDA)
matrix = confusion_matrix(predictModel(NVDA)[0], predictModel(NVDA)[1])
print(len(NVDA))
print(matrix)
print(matrix[1])



