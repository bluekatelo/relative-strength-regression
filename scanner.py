import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import statsmodels.api as sm
import os

comparisonStocks = ['MSFT', 'NVDA', 'AAPL', 'NFLX', 'TSLA', 'AMZN', 'FB', 'GOOG', 'ADBE']
base_stock = 'SPY'

class backtesting:
    def __init__(self) -> None:
        self.inTrade = False
        self.money = 1000000
        self.portfolio = {}
        self.default_qty = 100
        pass

    #returns direction of stock (up/down) based on SMA
    def direction(self, df):
        df['Direction'] = 0
        for i in range(0, len(df['Close'])):
            if df['5_SMA'][i] > df['8_SMA'][i]:
                df['Direction'][i] = 1

    #really eh overcomplicated trade function, should be replaced with separate buy and sell functions
    def trade(self, df):
        for i in range(1, len(df['Close'])):
            if df['relStr'][i] > 0.4 and SPY['Direction'][i] == 1 and self.inTrade == False:
                self.inTrade = True
                self.portfolio['Shares'] = self.default_qty
                self.money = self.money - (df['Close'][i] * self.default_qty)
            elif df['5_SMA'][i] < df['8_SMA'][i] and self.inTrade == True:
                self.inTrade = False
                self.portfolio['Shares'] = {}
                self.money = self.money + (df['Close'][i] * self.default_qty)
        return self.money


class Scanner:
    def __init__(self, comparisonStocks, base_index, period, interval) -> None:
        self.comparisonStocks = comparisonStocks
        self.base_index = base_index
        self.period = period
        self.interval = interval
        pass

    #create pandas dataframe from a given stock ticker string
    def makeStock_df(self, ticker):
        ticker = yf.Ticker(ticker)
        return ticker.history(period=self.period, interval=self.interval)

    # assign to new column = subtract by 1st open price, divide by first open price
    def pct_move_from_open(self, df):
        df['pct_move_from_open'] = ((df['Close'] - df.iloc[0, 0]) / df.iloc[0, 0]) * 100

    def pct_move_from_lastBar(self, df):
        df['pct_move_from_lastBar'] = (df['Close'] - df['Open']) / df['Open'] 

    #add moving average column to dataframe
    def sma(self, df, outputColumn, inputColumn, span):
        df[outputColumn] = df[inputColumn].rolling(span).mean()

    def relativeStrength(self, df):
        df['relStr'] = df['5_SMA'] - SPY['5_SMA']

    def read_pkl(self, df):
        path = "models/%s.pkl" % df
        return pd.read_pickle(path)

    #fill out stock 
    def fill_and_pickle(self):
        for i in self.comparisonStocks:
            df = self.makeStock_df(i)
            if self.interval == '1d':
                self.pct_move_from_lastBar(df)
                self.sma(df, '8_SMA', 'pct_move_from_lastBar', 8)
                self.sma(df, '5_SMA', 'pct_move_from_lastBar', 5)
            else:
                self.pct_move_from_open(df)
                self.sma(df, '8_SMA', 'pct_move_from_open', 8)
                self.sma(df, '5_SMA', 'pct_move_from_open', 5)
                backtest.direction(df)
                self.relativeStrength(df)

            # set path to models folder
            path = "models/" + i + ".pkl"
            df.to_pickle(path)

    #return a sorted list of stocks with strongest relative strength in last minute
    def strengthSort(self):
        str_df = pd.DataFrame(columns=['Ticker', 'Relative Strength'])
        for i in self.comparisonStocks:
            ticker = i
            df = self.read_pkl(i)
            relStr = df['relStr'][-1]
            str_df.loc[len(str_df.index)] = [ticker, relStr]
        
        str_df = str_df.sort_values(by=['Relative Strength'], key=abs, ascending=False, ignore_index=True)
        return str_df