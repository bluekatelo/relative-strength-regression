from abc import ABC, abstractmethod
import yfinance as yf

class IDataFetcher(ABC):
    @abstractmethod
    def fetch_stock_data(self, ticker, period, interval):
        pass

class YahooFinanceFetcher(IDataFetcher):
    def fetch_stock_data(self, ticker, period, interval):
        ticker_obj = yf.Ticker(ticker)
        return ticker_obj.history(period=period, interval=interval)