import numpy as np
from numpy.linalg import norm
import pandas as pd
import itertools
import random
from matplotlib import pyplot as plt
from time import time

class ExecutePairTrading:
    def __init__(self, abs_spread_mean, abs_spread_std, entry_signal=1.5, stop_loss=0.2):
        self.abs_spread_mean = abs_spread_mean
        self.abs_spread_std = abs_spread_std
        self.entry_signal=entry_signal
        self.stop_loss=0.2
        self.final_pl=0
        self.final_pl_pct=0

    def long_stock1_flag(self, stock1, stock2, idx):
        """
        This function is a ultility function to determine which stock to long/short given an entry signal.
        It:
            1. Takes the prices of two stocks, and the position where the entry signal appears 
            2. Calculate the percentage deltas between the current price and the price 7 days ago (or the earliest record) for each stock

        Then we will tell the ago to short the one with higher percentage delta and long the other.

        The function returns a boolean on whether we should long the stock 1.
        """
        stock1_current = stock1[idx]
        stock1_ref = stock1[max(0, idx-7)]

        stock2_current = stock2[idx]
        stock2_ref = stock2[max(0, idx-7)]

        pct_delta_1 = (stock1_current/stock1_ref) - 1
        pct_delta_2 = (stock2_current/stock2_ref) - 1

        if pct_delta_1 >= pct_delta_2:
            return False
        else:
            return True
        
    def execute(self, vec1, vec2, dates, base_fund=100, split=0.5, verbose=False):

        abs_spread = abs(np.array(vec1) - np.array(vec2))
        entry_thresh = self.abs_spread_mean + self.entry_signal*self.abs_spread_std
        exit_thresh = self.abs_spread_mean + 0.1*self.abs_spread_std

        # get the positions where the entry/exit signals appears
        entry_signals = np.array([1 if abs_spread[i] >= entry_thresh else 0 for i in range(0, len(abs_spread))])
        exit_signals = np.array([1 if abs_spread[i] <= exit_thresh else 0 for i in range(0, len(abs_spread))])

        # when an entry is detected, execute the next day
        entry_positions = np.where(entry_signals == 1)[0]+1
        exit_positions = np.where(exit_signals == 1)[0]+1

        # Sometimes the entry signal is detected is on the last day of the testing window, which makes the entry position out of bound
        length_spread = len(abs_spread)
        entry_positions = entry_positions[entry_positions<length_spread]
        exit_positions = exit_positions[exit_positions<length_spread]

        signal_pairs = []
        for entry_pos in entry_positions:
            # Find the first exit position that is greater than the entry position
            next_exit_pos = exit_positions[exit_positions > entry_pos]
            if next_exit_pos.size > 0:
                exit_pos = next_exit_pos[0]
            else:
                # Default exit position if no exit signal is found after the entry signal
                exit_pos = length_spread - 1
            signal_pairs.append((entry_pos, exit_pos))

        self.final_pl = 0
        self.num_entries = 0
        if len(signal_pairs) >0:
            # Create a dataframe to store the results
            temp_tb = pd.DataFrame(signal_pairs)
            temp_tb.columns = ['entry_idx', 'exit_idx']
            temp_tb = temp_tb.groupby('exit_idx').min().reset_index()

            # record the number of entries detected
            self.num_entries = temp_tb.shape[0]

            temp_tb['stock1_price_entry'] = vec1[temp_tb['entry_idx']] 
            temp_tb['stock1_price_exit'] = vec1[temp_tb['exit_idx']] 
            temp_tb['stock2_price_entry'] = vec2[temp_tb['entry_idx']] 
            temp_tb['stock2_price_exit'] = vec2[temp_tb['exit_idx']] 
            temp_tb['long_stock_1'] = [self.long_stock1_flag(vec1, vec2, x) for x in temp_tb.entry_idx]

            pnls = []
            trade_descriptions = []
            for row in range(temp_tb.shape[0]):
                long_pnl=0
                short_pnl=0
                if temp_tb.long_stock_1[row]:
                    # calculate pnl when we long stock 1 and short stock 2
                    long_pnl = base_fund * split * ((temp_tb.stock1_price_exit.values[row] - temp_tb.stock1_price_entry.values[row])/temp_tb.stock1_price_entry.values[row])
                    short_pnl = base_fund * (1-split) * ((temp_tb.stock2_price_entry.values[row] - temp_tb.stock2_price_exit.values[row])/temp_tb.stock2_price_entry.values[row])
                else:
                    # calculate pnl when we long stock 2 and short stock 1
                    long_pnl = base_fund * (1-split) * ((temp_tb.stock2_price_exit.values[row] - temp_tb.stock2_price_entry.values[row])/temp_tb.stock2_price_entry.values[row])
                    short_pnl = base_fund * (split) * ((temp_tb.stock1_price_entry.values[row] - temp_tb.stock1_price_exit.values[row])/temp_tb.stock1_price_entry.values[row])
                pnls.append(long_pnl+short_pnl)
            temp_tb['pnl'] = pnls
            temp_tb['entry_dates'] = temp_tb.entry_idx.apply(lambda x: dates[x])
            temp_tb['exit_dates'] = temp_tb.exit_idx.apply(lambda x: dates[x])
            self.final_pl = temp_tb.pnl.sum()
            self.trade_execution_table = temp_tb
        self.final_pl_pct = self.final_pl/base_fund

        return self
    
def cos_sim(rs,df):
            rows = df.loc[rs.index]
            vec1_sub1 = rows['Close_P1']
            vec2_sub1 = rows['Close_P2']
            return np.dot(vec1_sub1, vec2_sub1) / (norm(vec1_sub1) * norm(vec2_sub1))

# Correlation coef
def corr_coef(rs,df):
    rows = df.loc[rs.index]
    vec1_sub1 = rows['Close_P1']
    vec2_sub1 = rows['Close_P2']
    return np.corrcoef(vec1_sub1, vec2_sub1)[0, 1]

def generate_training_data(data, training_len=500, test_len=120, calculate_label=True ,verbose=False, execution_class=ExecutePairTrading, combinations=None):
    """
    data: A pandas dataframe from the GetSP500Data module in ultils
    """
    ts1 = time()
    # set seed
    random.seed(23)

    # Get a smaller version of the data for look-ups
    data_agg = data[['Ticker','GICS Sector', 'GICS Sub-Industry']].drop_duplicates().reset_index(drop=True)

    if combinations is None:
        # Get all the combinations of tickers
        tickers = list(data_agg.Ticker.values)
        combinations = list(itertools.combinations(tickers, 2))
    print(f"{len(combinations)} stock pairs detected")

    # Initiate data tables to store the generated results
    feature_columns = [
        'Date', 'Ticker_P1', 'Close_P1', 'Ticker_P2', 'Close_P2', 
        'High_P1', 'High_P2', 'Low_P1', 'Low_P2', 'Volume_P1', 'Volume_P2', 'abs_spread',
        'same_sector_flag', 'same_sub_industry_flag',
       'abs_spread_mean', 'abs_spread_std', 'abs_spread_mean_l28',
       'abs_spread_std_l28', 'spread_normed', 'abs_spread_normed_max',
       'abs_spread_normed_90th', 'abs_spread_normed_75th',
       'abs_spread_normed_median', 'abs_spread_normed_l7_avg',
       'abs_spread_normed_l14_avg', 'cos_sim', 'corr_coef'
    ]
    
    label_columns = [
        'Date', 'Ticker_P1', 'Ticker_P2', 'pnls', 'num_entries'
    ]
    metadata_tb_columns = [
         'Date', 'Ticker_P1', 'Ticker_P2', 'pnls', 'trade_executions'
    ]

    features_tb = pd.DataFrame(columns=feature_columns)
    labels_tb = pd.DataFrame(columns=label_columns)
    pnl_metadata_tb = pd.DataFrame(columns=metadata_tb_columns)
    
    i = 1
    # Get a list of unique dates for later use
    all_dates = data['Date'].unique()    
    ts_pre_loop =  time()
    print(f"Took {ts_pre_loop - ts1} to initilize. Entering ticker pair loop")

    for ticker1, ticker2 in combinations:
        ts2 = time()
        if verbose:
            print(f"{i}th pair ------------------")

        if (i>0) & (i%100==0):
            # break
            print(f"Getting the {i}th pair")
            ts1000 = time()
            print(f'Used {ts1000-ts_pre_loop} for the 100 pairs')
            ts_pre_loop = time()
        i+=1

        # Flag indicating whether the two tickers are from the same sector
        same_sector_flag = data_agg[data_agg.Ticker==ticker1]['GICS Sector'].values[0] == data_agg[data_agg.Ticker==ticker2]['GICS Sector'].values[0]
        same_sub_industry_flag = data_agg[data_agg.Ticker==ticker1]['GICS Sub-Industry'].values[0] == data_agg[data_agg.Ticker==ticker2]['GICS Sub-Industry'].values[0]
        
        # The the full history of the data
        vec1_full = data[['Ticker','Date', 'High', 'Low', 'Volume','Adj Close']][data.Ticker==ticker1].reset_index(drop=True)
        vec2_full = data[['Ticker','Date', 'High', 'Low', 'Volume','Adj Close']][data.Ticker==ticker2].reset_index(drop=True)
        vec1_full.columns = ['Ticker','Date', 'High', 'Low', 'Volume','Close']
        vec2_full.columns = ['Ticker','Date', 'High', 'Low', 'Volume','Close']
        
        # Check if a ticker has incomplete data
        if len(vec1_full) != len(vec2_full):
            # print(f"Incomplete data detected when generating for pair {ticker1} and {ticker2}. Skipping")
            continue

        # Create a temp table for the features
        df = pd.merge(vec1_full,vec2_full,on='Date',how='left',suffixes=['_P1','_P2'])

        start_ts = time()
        # Calculate the features
        df['same_sector_flag'] = same_sector_flag
        df['same_sub_industry_flag'] = same_sub_industry_flag
        df['abs_spread'] = (df['Close_P1'] - df['Close_P2']).abs()
        df['abs_spread_mean'] = df.rolling(training_len).abs_spread.mean()
        df['abs_spread_std'] = df.rolling(training_len).abs_spread.std()
        df['abs_spread_mean_l28'] = df.rolling(28).abs_spread.mean()
        df['abs_spread_std_l28'] = df.rolling(28).abs_spread.std()
        df['spread_normed'] = (df['abs_spread']-df['abs_spread_mean'])/df['abs_spread_std']
        df['abs_spread_normed_max'] = df.spread_normed.abs().rolling(training_len).max()
        df['abs_spread_normed_90th'] = df.spread_normed.abs().rolling(training_len).quantile(0.9)
        df['abs_spread_normed_75th'] = df.spread_normed.abs().rolling(training_len).quantile(0.75)
        df['abs_spread_normed_median'] = df.spread_normed.abs().rolling(training_len).median()
        df['abs_spread_normed_l7_avg'] = df.spread_normed.abs().rolling(7).mean()
        df['abs_spread_normed_l14_avg'] = df.spread_normed.abs().rolling(14).mean()
        df['cos_sim'] = df['Close_P1'].rolling(training_len).apply(cos_sim, args=(df,))
        df['corr_coef'] = df['Close_P1'].rolling(training_len).apply(corr_coef, args=(df,))

        end_ts = time()
        if verbose:
            print(f"{end_ts - start_ts} to calculate features")

        ## Appending the tables
        features_tb = pd.concat(
            [
                features_tb,
                df[
                    feature_columns
                ]
            ]
        )

        # Calculate the labels
        if calculate_label:
            start_ts = time()
            pnls = []
            num_entries = []
            trading_tables = []
            for idx in range(df.shape[0]):
                if (idx < training_len) | (idx > df.shape[0]-test_len-1):
                    pnls.append(np.nan)
                    trading_tables.append(np.nan)
                    num_entries.append(np.nan)
                else:
                    current_row = df.loc[idx]
                    result=execution_class(
                                    current_row.abs_spread_mean,
                                    current_row.abs_spread_std,
                                    entry_signal=1.5
                                ).execute(
                                    vec1=df.loc[(idx+1):(idx+test_len)]['Close_P1'].values,
                                    vec2=df.loc[(idx+1):(idx+test_len)]['Close_P2'].values,
                                    dates=df.loc[(idx+1):(idx+test_len)]['Date'].values
                                )
                    pnls.append(result.final_pl_pct)
                    num_entries.append(result.num_entries)
                    if result.final_pl_pct != 0:
                        trading_tables.append(result.trade_execution_table)
                    else:
                        trading_tables.append(None)

            end_ts = time()
            if verbose:
                print(f"{end_ts - start_ts} to calculate labels")

            df['pnls'] = pnls
            df['trade_executions'] = trading_tables
            df['num_entries'] = num_entries

            labels_tb = pd.concat(
                [
                    labels_tb,
                    df[
                        label_columns
                    ]
                ]
            )

            pnl_metadata_tb = pd.concat(
                [
                    pnl_metadata_tb,
                    df[
                        metadata_tb_columns
                    ]
                ]
            )
        
    ts_final = time()
    print(f"Took {ts_final - ts1} to finish")
    return features_tb, labels_tb, pnl_metadata_tb

