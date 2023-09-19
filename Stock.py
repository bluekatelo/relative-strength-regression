class Stock:
    def __init__(self, ticker, data_fetcher, period, interval):
        self.ticker = ticker
        self.data_fetcher = data_fetcher
        self.dataframe = self.data_fetcher.fetch_stock_data(ticker, period, interval)

        # take care of indicators that are definitely needed within the constructor
        self.add_pct_move_from_lastBar()

    def add_sma(self, outputColumn, inputColumn, span):
        self.dataframe[outputColumn] = self.dataframe[inputColumn].rolling(span).mean()

    def add_pct_move_from_open(self):
        self.dataframe['pct_move_from_open'] = ((self.dataframe['Close'] - self.dataframe.iloc[0, 0]) / self.dataframe.iloc[0, 0]) * 100
    
    def add_pct_move_from_lastBar(self):
        self.dataframe['pct_move_from_lastBar'] = (self.dataframe['Close'] - self.dataframe['Open']) / self.dataframe['Open']

    def add_relative_strengh(self, comparison_index):
        self.dataframe['relStr'] = self.dataframe['5_SMA'] - comparison_index.dataframe['5_SMA']