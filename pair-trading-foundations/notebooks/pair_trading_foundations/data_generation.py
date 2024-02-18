import numpy as np
from numpy.linalg import norm
import pandas as pd
import itertools
import random
from matplotlib import pyplot as plt

class ExecutePairTrading:
    def __init__(self, abs_spread_mean, abs_spread_std, entry_signal=1.5, stop_loss=0.2):
        self.abs_spread_mean = abs_spread_mean
        self.abs_spread_std = abs_spread_std
        self.entry_signal=entry_signal
        self.stop_loss=0.2
        
        self.stock1_position=None
        self.stock2_position=None 
        self.stock1_long=None
        self.stock1_pl=0
        self.stock2_pl=0
        self.final_pl=0
        self.final_pl_pct=0

    def execute(self, vec1, vec2, base_fund=100, split=0.5, verbose=False):
        
        abs_spread = abs(np.array(vec1) - np.array(vec2))

        entry_signal=False
        for i in range(len(abs_spread)):
            # if currently not holding any position
            if self.stock1_position is None:
                
                ## if an entry signal was not generated from the previous trading date
                if not entry_signal:
                    # Check if entry signal appears in the current date
                    if abs_spread[i] >= self.abs_spread_mean + self.entry_signal*self.abs_spread_std:
                        entry_signal=True
                    
                ## If an entry signal was generated from the previous trading date, execute the strategy
                else:
                    self.stock1_position=vec1[i]
                    self.stock2_position=vec2[i]
                    
                    if vec1[i] > vec2[i]:
                        self.stock1_long=False
                        if verbose:
                            print(f"Entered long position at {vec1[i]} for stock1")
                            print(f"Entered short position at {vec2[i]} for stock2")
                    else:
                        self.stock1_long=True
                        if verbose:
                            print(f"Entered short position at {vec1[i]} for stock1")
                            print(f"Entered long position at {vec2[i]} for stock2")

            # if holding a position, check if need to exit and update pnl
            else:
                # calculate pnl
                if self.stock1_long:
                    # calculate pnl when we long stock 1 and short stock 2
                    self.stock1_pl = base_fund * split * ((vec1[i] - self.stock1_position)/self.stock1_position)
                    self.stock2_pl = base_fund * (1-split) * ((self.stock2_position - vec2[i])/self.stock2_position)
                else:
                    # calculate pnl when we long stock 2 and short stock 1
                    self.stock1_pl = base_fund *split* (self.stock1_position - vec1[i])/self.stock1_position
                    self.stock2_pl = base_fund *(1-split)* (vec2[i] - self.stock2_position)/self.stock2_position    

                # when the absolute spread is less than 0.1 of the std, exit the positions
                ## to-do in the future - this calculation is in-accurate as the exit should be executed the next
                ## trading day, when the pnl would be differentt.
                if abs_spread[i] <= self.abs_spread_mean + 0.1 * self.abs_spread_std:
                    # reset the params
                    entry_signal=False
                    self.stock1_position=None
                    self.stock2_position=None
                    self.stock1_long=None

                    if verbose:
                        print(f"Closed stock1 position at {vec1[i]}. Close stock2 position at {vec2[i]}")
                    # record the pnl from this trade
                    self.final_pl += (self.stock1_pl + self.stock2_pl)
                    self.final_pl_pct = self.final_pl/base_fund

                    if verbose:
                        print(f'Total PnL from this trade is {(self.stock1_pl + self.stock2_pl)}')
                        print(f'cumulative PnL percentage is {self.final_pl/base_fund}')
                    self.stock1_pl=0
                    self.stock2_pl=0
                     

        return self


def generate_training_data(data, training_len=500, test_len=120, sample_size_per_pair=100, verbose=False, execution_class=ExecutePairTrading):
    """
    data: A pandas dataframe from the GetSP500Data module in ultils
    """
    random.seed(23)
    # Get all the combinations of tickers
    tickers = list(set(data.Ticker))
    combinations = list(itertools.combinations(tickers, 2))
    print(f"{len(combinations)} stock pairs detected")

    # Initiate data tables to store the generated results
    recorded_info_tb = pd.DataFrame(columns=[
            'ticker1', 
            'ticker2',
            'target_date',
            'abs_spread_mean',
            'abs_spread_std',
            'abs_spread_mean_l28',
            'abs_spread_std_l28',
    ])

    features_tb = pd.DataFrame(columns=[
        'ticker1', 
        'ticker2',
        'target_date',
        'same_sector_flag',
        'same_sub_industry_flag',
        'cos_sim',
        'corr_coef',
        'abs_spread_normed_max',
        'abs_spread_normed_90th',
        'abs_spread_normed_75th',
        'abs_spread_normed_median',
        'abs_spread_normed_l7_avg',
        'abs_spread_normed_l14_avg'
    ])

    labels_tb = pd.DataFrame(columns=[
        'ticker1', 
        'ticker2',
        'target_date',
        'total_pnl',
        'total_pnl_l28_mean_std'
    ])
    
    i = 1
    for ticker1, ticker2 in combinations:
        if i%1000 == 0:
            print(f"Getting the {i}th pair")
        i+=1
        # Get a list of unique dates for later use
        all_dates = data['Date'].unique()    
        # Flag indicating whether the two tickers are from the same sector
        same_sector_flag = data[data.Ticker==ticker1]['GICS Sector'].values[0] == data[data.Ticker==ticker2]['GICS Sector'].values[0]
        same_sub_industry_flag = data[data.Ticker==ticker1]['GICS Sub-Industry'].values[0] == data[data.Ticker==ticker2]['GICS Sub-Industry'].values[0]

        # The the full history of the data
        vec1_full = data['Close'][data.Ticker==ticker1].values
        vec2_full = data['Close'][data.Ticker==ticker2].values

        # Check if a ticker has incomplete data
        if len(vec1_full) != len(vec2_full):
            # print(f"Incomplete data detected when generating for pair {ticker1} and {ticker2}. Skipping")
            continue


        # Number of days in the data
        num_days_total = len(vec1_full)

        # Keep x days for training and y days for label calculation
        possible_indices_to_sample = list(range(training_len, num_days_total-test_len-1))
        sampled_indices = random.choices(possible_indices_to_sample, k=sample_size_per_pair)

        for idx in sampled_indices:
            if verbose:
                print(f"Sampled date is {all_dates[idx]}")
                print(f"Dataset1 starts from {all_dates[idx-500]} to {all_dates[idx]}(exclusive)")
                print(f"Dataset2 starts from {all_dates[idx]} (inclusive) to {all_dates[idx+120]}")

            # Previous N trading days
            vec1_sub1 = vec1_full[(idx - training_len):idx]
            vec2_sub1 = vec2_full[(idx - training_len):idx]

            # Next N trading days
            vec1_sub2 = vec1_full[idx:(idx+test_len)]
            vec2_sub2 = vec2_full[idx:(idx+test_len)]
            
            # Cosine sim
            cos_sim = np.dot(vec1_sub1, vec2_sub1) / (norm(vec1_sub1) * norm(vec2_sub1))
            # Correlation coef
            corr_coef = np.corrcoef(vec1_sub1, vec2_sub1)[0, 1]

            # Absolute value of the difference of the two stocks
            abs_spread = abs(vec1_sub1 - vec2_sub1)
            abs_spread_mean = np.mean(abs_spread)
            abs_spread_std = np.std(abs_spread)

            # Sometimes, historical data might be too strict to get a signal for trade in
            abs_spread_mean_l28 = np.mean(abs_spread[-28:])
            abs_spread_std_l28 = np.std(abs_spread[-28:])

            # normalize
            spread_normed = (abs_spread - abs_spread_mean)/abs_spread_std

            # distribution of abs std
            ## This is to capture when we should expect the reverting to mean to happen for this stock
            abs_spread_normed_max = max(abs(spread_normed))
            abs_spread_normed_90th = np.percentile(abs(spread_normed),90)
            abs_spread_normed_75th = np.percentile(abs(spread_normed),75)
            abs_spread_normed_median = np.percentile(abs(spread_normed),50)

            # latest 7 day/14 day avg normalized spread
            ## These could help predict whether a trading signal will appear
            abs_spread_normed_l7_avg = abs(np.mean(spread_normed[-7:]))
            abs_spread_normed_l14_avg = abs(np.mean(spread_normed[-14:]))

            ## These are information we record for each iteration for debugging an analysis
            recorded_info = [
                ticker1, 
                ticker2,
                all_dates[idx],
                abs_spread_mean,
                abs_spread_std,
                abs_spread_mean_l28,
                abs_spread_std_l28
            ]

            ## There are features that goes into the model (not ticker1, ticker2, the target date, which are used to join the tables)
            features = [
                ticker1, 
                ticker2,
                all_dates[idx],
                same_sector_flag,
                same_sub_industry_flag,
                cos_sim,
                corr_coef,
                abs_spread_normed_max,
                abs_spread_normed_90th,
                abs_spread_normed_75th,
                abs_spread_normed_median,
                abs_spread_normed_l7_avg,
                abs_spread_normed_l14_avg
            ]

            # Append the generated data as new rows to the tables
            recorded_info_tb.loc[len(recorded_info_tb)] = recorded_info
            features_tb.loc[len(features_tb)] = features

            # Calculate the labels
            trade_l28_signal = execution_class(
                    abs_spread_mean_l28,
                    abs_spread_std_l28,
                    entry_signal=1.5
                ).execute(
                    vec1=vec1_sub2,
                    vec2=vec2_sub2
                )
            
            trade_all_signal = execution_class(
                    abs_spread_mean,
                    abs_spread_std,
                    entry_signal=1.5
                ).execute(
                    vec1=vec1_sub2,
                    vec2=vec2_sub2
                )

            label_list = [
                ticker1,
                ticker2,
                all_dates[idx],
                trade_l28_signal.final_pl_pct,
                trade_all_signal.final_pl_pct
            ]
                      
            labels_tb.loc[len(labels_tb)] = label_list
                        
    return recorded_info_tb, features_tb, labels_tb

