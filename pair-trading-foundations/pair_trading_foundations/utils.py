import yfinance as yf
yf.pdr_override()
from pandas_datareader import data as pdr
import pandas as pd
import numpy as np
import plotly.express as px

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
    
def examine_output_data(tb, ticker1, ticker2, date):

    sub = tb[(tb.Ticker_P1==ticker1)&(tb.Ticker_P2==ticker2)].reset_index(drop=True)
    idx = np.where(sub.Date==date)[0][0]
    sub = tb[(tb.Ticker_P1==ticker1)&(tb.Ticker_P2==ticker2)].reset_index(drop=True)[idx:(idx+121)]

    # Get the trading execution
    trade_exec = sub.trade_executions.values[0]

    fig = px.line(sub, x="Date", y=['Close_P1','Close_P2', 'abs_spread'])

    for entry_date in trade_exec.entry_dates.values: 
        fig.add_vline(x=entry_date, line_width=3, line_dash="dash", line_color="green")
    for exit_date in trade_exec.exit_dates.values: 
        fig.add_vline(x=exit_date, line_width=3, line_dash="dash", line_color="red")

    fig.show()
    trade_exec

    return None