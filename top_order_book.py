# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 22:16:15 2021

@author: Hansen
"""
from typing import Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class TradeVisualize:
    """
    Class that visualize the ask/bid changes with trade in it.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initiate with a DataFrame as formatted by Ben. Note the data read will be reordered automatically.

        Parameters
        ----------
        data : pd.DataFrame
            The df directly read from the csv file cleaned by Ben.
        """

        
        # Sort the data by sequence at first, then by price
        self.data = data.sort_values(by=['Sequence', 'Price'], ascending=[True, False]).reset_index(drop=True)
        # Change into real prices
        # self.data['Price'] = self.data['Price'].multiply(1/100)
    
    def get_trades(self, start_time: Union[int, None] = None, end_time: Union[int, None] = None,
                      include_0_vol: bool = False) -> pd.Series:
        """
        Locate the index when the trade happens.

        Parameters
        ----------
        start_time : Union[int, None], optional
            The starting time of the search in HHMMSS format. None means from the beginning of the file. The default is
            None. Inclusive.
        end_time : Union[int, None], optional
            The ending time of the search in HHMMSS format. None means from the beginning of the file. The default is
            None. Inclusive.
        include_0_vol : bool, optional
            Whether to inlcude 0 vol data. The default is False.

        Returns: pd.Series of index indicating where trades happen.
        -------
        """

        # Setting up
        data = self.data
        lowest_vol = 0 if include_0_vol else 1
        if start_time is None:
            start_time = data.iloc[0]['Time']  # Starting time in the data sheet
        if end_time is None:
            end_time = data.iloc[-1]['Time']  # Endind time in the data sheet

        # 1. Filter by time
        data_time = data[(data['Time'] >= start_time) & (data['Time'] <= end_time)]

        # 2. Filter by trades
        trades_bool = (data_time['ASK/BID'].isna()) & (data_time['Volume'] >= lowest_vol)
        trades_idx = trades_bool[trades_bool].index

        # Returns the index series when a trade happens
        return trades_idx

    def get_prev_ab_idx(self, trades_idx: pd.Series) -> pd.DataFrame:
        """
        Get the prevailing ask and bid's index right before a trade.

        Parameters
        ----------
        trades_idx : pd.Series
            The indices at which the trade occurs.

        Returns
        -------
        prev_ab_df : pd.DataFrame
            The index of prevailing BID and ASK, and also their sequence number. They should share the same sequence.
        """

        prev_ab_idx_dict = dict()
        # 1. Find the two indicies immediately above a trade that is not a trade
        for num, idx in enumerate(trades_idx):
            # Handle the continuous trade case, and move the pointer to the most recent non-trade pair.
            pointer = idx
            while True:
                if pointer not in trades_idx:  # If not in there then it is a valid choise
                    break
                pointer -= 1  # Else, keep looking

            # Put data into the dictionary
            prev_ab_idx_dict[idx] = {'BID idx': pointer, 'ASK idx': pointer-1,
                                     'Sequence BID': self.data.iloc[pointer]['Sequence'],
                                     'Sequence ASK': self.data.iloc[pointer-1]['Sequence']}

        # 2. Change dict into pd.DataFrame
        prev_ab_df = pd.DataFrame.from_dict(data=prev_ab_idx_dict, orient="index")

        return prev_ab_df

    def plot_abt(self, prev_ab_df: pd.DataFrame, ax: plt.axes = None) -> plt.axes:
        """
        Quick plot the prevailing bid ask for a trade, and the trade price.

        Parameters
        ----------
        prev_ab_df : pd.DataFrame
            The data frame having the prevailing bid and ask index in the sheet.
        ax : plt.axes, optional
            Plotting axes. The default is None.

        Returns
        -------
        ax : plt.axes
            Plotting axes.
        """

        trade_idx = prev_ab_df.index
        bid_idx = prev_ab_df['BID idx']
        ask_idx = prev_ab_df['ASK idx']

        # 1. Get the prices data.
        trade_prices = self.data.iloc[trade_idx][['Time', 'Price', 'Volume']]
        bid_prices = self.data.iloc[bid_idx][['Time', 'Price', 'Volume']]
        ask_prices = self.data.iloc[ask_idx][['Time', 'Price', 'Volume']]

        # 2. Plot them
        ax = ax or plt.gca()
        ax.scatter(trade_prices['Time'], trade_prices['Price'], label='trade', marker='.', linewidth=1,
                   s=trade_prices['Volume']**1.5, alpha=0.8)
        ax.scatter(trade_prices['Time'], bid_prices['Price'], label='bid', marker='.', linewidth=1,
                   s=bid_prices['Volume']**1.5, alpha=0.02)
        ax.scatter(trade_prices['Time'], ask_prices['Price'], label='ask', marker='.', linewidth=1,
                   s=ask_prices['Volume']**1.5, alpha=0.02)
        ax.set_ylabel('Price')
        ax.set_xlabel('Time')
        ax.legend()
        ax.grid()

        return ax

