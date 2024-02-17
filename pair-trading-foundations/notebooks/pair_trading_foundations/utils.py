import yfinance as yf
yf.pdr_override()
from pandas_datareader import data as pdr
import pandas as pd
import numpy as np

class GetSP500Data:
    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date=end_date
        return None
    
    def get_all_sp_tickers(self):
        
        data = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        df = data[0].reset_index()
        #Extracting required features
        df = df[['Symbol','GICS Sector','GICS Sub-Industry']]

        self.sp500_stocks = df

        return self
        
    def get_single_stock_history(self, ticker):
        tb = pdr.get_data_yahoo(ticker, start=self.start_date, end=self.end_date).reset_index()
        tb['Ticker'] = ticker
        return tb
    
    def get_consolidated_data(self):
        
        combined_table=pd.DataFrame()
        for ticker in self.sp500_stocks['Symbol']:
            temp = self.get_single_stock_history(ticker=ticker)
            combined_table = pd.concat([combined_table,temp], ignore_index=True)
        
        combined_table = pd.merge(
            combined_table, 
            self.sp500_stocks,
            how='left',
            left_on='Ticker',
            right_on='Symbol'
            ).drop(['Symbol'], axis=1)
        
        self.combined_table = combined_table
        return combined_table
    