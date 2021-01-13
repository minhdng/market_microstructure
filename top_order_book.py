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
    
    def __init__(self, data: pd.DataFrame, lump_trades: bool = False):
        """
        Initiate with a DataFrame as formatted by Ben. Note the data read will be reordered automatically.

        Parameters
        ----------
        data : pd.DataFrame
            The df directly read from the csv file cleaned by Ben.
        lump_trades : bool, optional
            Whether to lump continuous trades together. The default is False.
        """

        
        # Sort the data by sequence at first, then by price
        self.data = data.sort_values(by=['Sequence', 'Price'], ascending=[True, False]).reset_index(drop=True)
        # Change into real prices
        # self.data['Price'] = self.data['Price'].multiply(1/100)
        if lump_trades:
            self.data = self.lump_conti_trades()
    
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

    def lump_conti_trades(self) -> pd.DataFrame:
        """
        Lump continuous trades together in their volume.
        """

        # 1. Get trades
        trades_bool = (self.data['ASK/BID'].isna()) & (self.data['Volume'] >= 1)
        trades_idx = trades_bool[trades_bool].index
        
        # 2. Lump the continuous trades' volumes
        # 2.1 Trades that are first in a sequence (including those having no sequel)
        first_trades_idx = []
        for num, idx in enumerate(trades_idx):
            # Handle the continuous trade case, and move the pointer to the most recent non-trade pair.
            pointer = idx
            while True:
                if pointer not in trades_idx:  # If not in there then it is a valid choice
                    break
                pointer -= 1  # Else, keep looking
            first_trades_idx.append(pointer + 1)
        
        first_trades_idx = np.unique(np.array(first_trades_idx))  # The unique ones are what we need
        
        # 2.2 Add the sequel's volumes to the trade that firstly occur
        trades_idx_npa = np.array(trades_idx)
        lumped_volumes = np.zeros_like(first_trades_idx)
        p_fti = 0  # Pointer for first_trades_idx
        p_ti = 0  # Pointer for trade_idx
        while p_ti < len(trades_idx):
            # If the ti pointer catches up with the fti ponter + 1, then increase fti pointer
            if trades_idx[p_ti] == first_trades_idx[p_fti+1]:
                p_fti += 1
            # Lump the volume to the first trade in a sequel.
            lumped_volumes[p_fti] += self.data.iloc[trades_idx[p_ti]]['Volume']
            p_ti += 1
            
        # 3. Format the new DataFrame with lumped volume
        # 3.1 Change the volume
        lumped_df = self.data.copy()
        lumped_df.loc[first_trades_idx, 'Volume'] = lumped_volumes
        # 3.2 Get rid of sequel trades and only keep the leading one
        sequel_trades_idx = np.setdiff1d(trades_idx_npa, first_trades_idx)
        lumped_df = lumped_df.drop(index=sequel_trades_idx, inplace=False)

        return lumped_df

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
                if pointer not in trades_idx:  # If not in there then it is a valid choice
                    break
                pointer -= 1  # Else, keep looking

            # Put data into the dictionary
            prev_ab_idx_dict[idx] = {'BID idx': pointer, 'ASK idx': pointer-1,
                                     'Sequence BID': self.data.iloc[pointer]['Sequence'],
                                     'Sequence ASK': self.data.iloc[pointer-1]['Sequence']}

        # 2. Change dict into pd.DataFrame
        prev_ab_df = pd.DataFrame.from_dict(data=prev_ab_idx_dict, orient="index")

        return prev_ab_df
    
    def get_immediate_info(self, ab_df: pd.DataFrame, mid_type: str = 'simple') -> pd.DataFrame:
        """
        Get the immediate ask and bid info for a trade. 

        Depends on if ad_df is for the prevailing quote or the immediate afterwards info.

        Parameters
        ----------
        ab_df : pd.DataFrame
            The ask and bid dataframe.
        mid_type : str. Optional.
            The type of mid price. Choices are 'simple' and 'weighted'. The default is 'simple'.

        Returns
        -------
        immed_info_df : pd.DataFrame

        """

        # Get all the pieces from data, need to reset index
        trade_idx = ab_df.index
        trade_idx_series = pd.Series(trade_idx)  # Turn into a df for assemble
        dates = self.data.iloc[trade_idx]['Date'].reset_index(drop=True)
        entry_dates = self.data.iloc[trade_idx]['EntryDate'].reset_index(drop=True)
        times = self.data.iloc[trade_idx]['Time'].reset_index(drop=True)
        trade_volumes = self.data.iloc[trade_idx]['Volume'].reset_index(drop=True)
        ask_volumes = self.data.iloc[ab_df['ASK idx']]['Volume'].reset_index(drop=True)
        bid_volumes = self.data.iloc[ab_df['BID idx']]['Volume'].reset_index(drop=True)
        trade_prices = self.data.iloc[trade_idx]['Price'].reset_index(drop=True)
        ask_prices =  self.data.iloc[ab_df['ASK idx']]['Price'].reset_index(drop=True)
        bid_prices = self.data.iloc[ab_df['BID idx']]['Price'].reset_index(drop=True)
        mid_prices = self._get_mid_prices(ask_volumes, bid_volumes, ask_prices, bid_prices, mid_type)
        buy_or_sell = self._check_buy_or_sell(trade_prices, ask_prices, bid_prices)

        # Assemble into a DataFrame
        frame = {'TradeIdx': trade_idx_series,
                 'Date': dates,
                 'EntryDate': entry_dates,
                 'Time': times,
                 'TradeVolume': trade_volumes,
                 'AskVolume': ask_volumes,
                 'BidVolume': bid_volumes,
                 'TradePrice': trade_prices,
                 'AskPrice': ask_prices,
                 'BidPrice': bid_prices,
                 'MidPrice': mid_prices,
                 'Buy/Sell': buy_or_sell}
        immed_info_df = pd.DataFrame(frame)
        
        return immed_info_df
    
    @staticmethod
    def _get_mid_prices(ask_volumes: pd.Series, bid_volumes: pd.Series, ask_prices: pd.Series,
                        bid_prices: pd.Series, mid_type: str) -> pd.Series:
        """
        Get mid prices for bid and ask.

        Parameters
        ----------
        ask_volumes : pd.Series
            Immediate prevailing ask volumes for a trade.
        bid_volumes : pd.Series
            Immediate prevailing bid volumes for a trade.
        ask_prices : pd.Series
            Immediate prevailing ask prices for a trade.
        bid_prices : pd.Series
            Immediate prevailing bid prices for a trade.
        mid_type : str
            Simple or weighted.

        Raises
        ------
        ValueError
            No such mid_type.

        Returns
        -------
        mid_prices : pd.Series
            Calculated mid prices.

        """

        if mid_type == 'simple':
            mid_prices = (ask_prices + bid_prices) / 2

        elif mid_type == 'weighted':
            mid_prices = (ask_volumes * ask_prices + bid_volumes * bid_prices) / (ask_volumes + bid_volumes)

        else:
            raise ValueError('mid_type = "simple" or "weighted"')

        return mid_prices

    @staticmethod
    def _check_buy_or_sell(trade_prices: pd.Series, ask_prices: pd.Series, bid_prices: pd.Series) -> pd.Series:
        """
        Find if a trade is a buy or sell order by matching with immediate prevailing price.
        
        1 means buy, -1 means sell, 0 means not clear.

        Parameters
        ----------
        trade_prices : pd.Series
            Price at which trade occured.
        ask_prices : pd.DataFrame
            The ask price immediately prevailing a trade.
        bid_prices : pd.DataFrame
            The bid price immediately prevailing a trade.

        Returns
        -------
        buy_or_sell : pd.Series
            The result recorded in 1 or -1 for buy or sell respectively.

        """

        buy_or_sell_data = np.zeros_like(trade_prices)
        
        for idx, trade_price in trade_prices.iteritems():
            if trade_price > bid_prices.iloc[idx]:  # Greater than bid, so a buy.
                buy_or_sell_data[idx] = 1
            if trade_price < ask_prices.iloc[idx]:  # Smaller than ask, so a sell.
                buy_or_sell_data[idx] = -1
        
        # Change into series.
        buy_or_sell = pd.Series(buy_or_sell_data)
        
        return buy_or_sell

    def plot_by_trade(self, immed_info_df: pd.DataFrame, ax: plt.axes = None) -> plt.axes:
        """
        Quick plot the prevailing bid ask prices and volumes for a trade, and the trade price.
        
        Plot in candlesticks, with length indicating the volume.

        Parameters
        ----------
        immed_info_df : pd.DataFrame
            The data frame having the prevailing bid and ask info needed.
        ax : plt.axes, optional
            Plotting axes. The default is None.

        Returns
        -------
        ax : plt.axes
            Plotting axes.
        """

        trade_prices = immed_info_df['TradePrice']
        ask_prices = immed_info_df['AskPrice']
        bid_prices = immed_info_df['BidPrice']
        ask_volumes = immed_info_df['AskVolume']
        bid_volumes = immed_info_df['BidVolume']
        # trade_volumes = immed_info_df['TradePrice']

        # 2. Plot them
        ax = ax or plt.gca()
        
        # Plotting for the ask
        for idx, ask_price in ask_prices.iteritems():
            ax.vlines(x=idx, ymin=ask_price, ymax=ask_price + ask_volumes.iloc[idx], color='black')
            
        # Plotting for the bid
        for idx, bid_price in bid_prices.iteritems():
            ax.vlines(x=idx, ymin=bid_price - bid_volumes.iloc[idx], ymax=bid_price, color='brown')
            
        # Plotting for the trade
        ax.plot(trade_prices.index, trade_prices, marker='.', linestyle='dotted')
        
        ax.set_ylabel('Price')
        ax.set_xlabel('Trade Sequence')

        ax.grid()

        return ax

