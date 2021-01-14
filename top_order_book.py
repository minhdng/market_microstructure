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
        
        Note: The result DataFrame will be re-indexed. Another concern is sometimes for continuous trades, the prices
        change. So lumping their volume together may not be ideal.

        Returns
        -------
        lumped_df: pd.DataFrame
            The lumped dataframe with continuous trades volume collected together.

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

        return lumped_df.sort_values(by=['Sequence', 'Price'], ascending=[True, False]).reset_index(drop=True)

    def get_prev_ab_idx(self, trades_idx: pd.Series, handle_conti_trade: str = 'bidask') -> pd.DataFrame:
        """
        Get the prevailing ask and bid's index right before a trade.

        Parameters
        ----------
        trades_idx : pd.Series
            The indices at which the trade occurs.
        handle_conti_trades: str, optional
            How continuous trades are handled. Options are ['bidask', 'trade']. 'bidask' means look at the most
            recent bid and ask prices before a trade occurs. 'trade' means use the trade prevailing the current trade
            as the bid/ask price and in this case they are equal.

        Returns
        -------
        prev_ab_df : pd.DataFrame
            The index of prevailing BID and ASK, and also their sequence number. They should share the same sequence.
        """

        prev_ab_idx_dict = dict()
        if handle_conti_trade == 'bidask':
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
                
        elif handle_conti_trade == 'trade':
            # 1. Find the two indicies immediately above a trade that is not a trade
            for num, idx in enumerate(trades_idx):
                # Decide if the prev row is a trade or not.
                if idx-1 in trades_idx:  # If it is a trade, then put the trade data as the prevailing data
                    prev_ab_idx_dict[idx] = {'BID idx': idx-1, 'ASK idx': idx-1,
                                             'Sequence BID': self.data.iloc[idx-1]['Sequence'],
                                             'Sequence ASK': self.data.iloc[idx-1]['Sequence']}
                
                else:  # If the prev is not a trade, look up 2 indices for bid/ask as the prevailing data
                    prev_ab_idx_dict[idx] = {'BID idx': idx-1, 'ASK idx': idx-2,
                                             'Sequence BID': self.data.iloc[idx-1]['Sequence'],
                                             'Sequence ASK': self.data.iloc[idx-2]['Sequence']}
        
        else:
            raise ValueError("Available handle_conti_trade values are ['bidask', 'trade']")

        # 2. Change dict into pd.DataFrame
        prev_ab_df = pd.DataFrame.from_dict(data=prev_ab_idx_dict, orient="index")

        return prev_ab_df
    
    def get_immediate_info(self, ab_df: pd.DataFrame) -> pd.DataFrame:
        """
        Get the immediate ask and bid info for a trade. 

        Depends on if ad_df is for the prevailing quote or the immediate afterwards info.

        Parameters
        ----------
        ab_df : pd.DataFrame
            The ask and bid dataframe.

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
        mid_prices_s = self._get_mid_prices(ask_volumes, bid_volumes, ask_prices, bid_prices, mid_type='simple')
        mid_prices_w = self._get_mid_prices(ask_volumes, bid_volumes, ask_prices, bid_prices, mid_type='weighted')
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
                 'MidPrice': mid_prices_s,
                 'MidPriceW': mid_prices_w,
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

    def plot_by_trade(self, immed_info_df: pd.DataFrame, ax: plt.axes = None, plot_mid_price: bool = True) -> plt.axes:
        """
        Quick plot the prevailing bid ask prices and volumes for a trade, and the trade price.
        
        Plot in candlesticks, with length indicating the volume.

        Parameters
        ----------
        immed_info_df : pd.DataFrame
            The data frame having the prevailing bid and ask info needed.
        ax : plt.axes, optional
            Plotting axes. The default is None.
        plot_mid_price : bool, optional
            Whether plotting the weighted mid_price. The default is True.

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
        trade_volumes = immed_info_df['TradeVolume']
        avg_prices = self._get_mid_prices(ask_volumes, bid_volumes, ask_prices, bid_prices, mid_type='weighted')
        
        def marker_size_remap(trade_volumes: pd.Series) -> pd.Series:
            """
            Remap trade volumes for marker size.
            """
            
            marker_sizes = trade_volumes.map(lambda x: x ** 0.7 * 2 + 1)
            
            return marker_sizes

        # 2. Plot them
        ax = ax or plt.gca()
        
        # Plotting for the ask
        for idx, ask_price in ask_prices.iteritems():
            ax.vlines(x=idx, ymin=ask_price, ymax=ask_price + ask_volumes.iloc[idx], color='black', alpha=0.7)
            
        # Plotting for the bid
        for idx, bid_price in bid_prices.iteritems():
            ax.vlines(x=idx, ymin=bid_price - bid_volumes.iloc[idx], ymax=bid_price, color='brown', alpha=0.7)
        
        # Plotting for the trade
        # Line
        ax.plot(trade_prices.index, trade_prices, linestyle='dotted', linewidth=0.5, color='darkslategrey')
        # Marker
        marker_sizes = marker_size_remap(trade_volumes)
        ax.scatter(trade_prices.index, trade_prices, s=marker_sizes, alpha=0.9, color='forestgreen')
        
        # Plotting the weighted avg
        if plot_mid_price:
            ax.plot(avg_prices.index, avg_prices, alpha=0.9, linewidth=0.4, marker='s', markersize=2,
                    linestyle='--', color='darkcyan')
        
        ax.set_ylabel('Price')
        ax.set_xlabel('Trade Sequence')

        ax.grid()

        return ax
    
    def get_aggre_info(self, daily_immed_info: pd.DataFrame, duration: Union[None, int] = None) -> pd.DataFrame:
        """
        Get aggregated info for trades by seconds.
        
        The daily immediate info need to have its time continuous, concatenated from immediate info DataFrames, and
        they have to come from one day.
        col: 'Date', 'EntryDate', 'BegTime', 'EndTime', 'MidPriceBeg', 'MidPriceEnd', 'MidPriceWBeg', 'MidPriceWEnd',
        'OrderFlowImba'.

        Parameters
        ----------
        daily_immed_info : pd.DataFrame
            Concatenated daily info for each trade, should be an output of the get_immediate_info method.
        duration : Union[None, int], optional
            The duration of time in seconds. None means no aggregating, use the immediate previous data. 
            The default is None.

        Returns
        -------
        aggre_info : pd.DataFrame
            Aggregated information by seconds.

        """
        
        # 0. Setting params and intervals up
        if duration is None:  # R1 result follows from another framework.
            aggre_info = self._get_aggre_info_r1(daily_immed_info)
            
            return aggre_info
        else:
            dt = duration
        # Beginning time and ending time from the dataframe (has to be from a single day)
        df_beg_time = daily_immed_info.iloc[0]['Time']
        df_end_time = daily_immed_info.iloc[-1]['Time']
        # Create grids and intervals for aggregation
        time_checkpoints = np.arange(df_beg_time, df_end_time, dt)
        beg_times = time_checkpoints[:-1]  # Beginning time, exclude the last grid.
        end_times = time_checkpoints[1:] - 1  # Ending time for each duration, exclude the 0th grid, move 1 ahead.
        
        # 1. Calculating the aggregated volumes for each trade
        agg_impacts = np.zeros_like(beg_times)
        time_grid_pt = 0  # Time grid pointer
        cache_volume = []  # Volume within each time interval
        cache_buysell = []  # Volume within each time interval
        for row_id, row in daily_immed_info.iterrows():

            if row['Time'] >= beg_times[time_grid_pt] and row['Time'] <= end_times[time_grid_pt]:
                cache_volume.append(row['TradeVolume'])
                cache_buysell.append(row['Buy/Sell'])

            else:
                # Dot product to calculate aggregated impact
                agg_impact = np.dot(cache_volume, cache_buysell)
                # Document this impact
                agg_impacts[time_grid_pt] = agg_impact
                time_grid_pt += 1
                cache_volume = []
                cache_buysell = []
            
            if time_grid_pt >= len(beg_times):
                break

        # 2. Calculate the index in daily_immed_info corresponding to beg_times and end_times.
        time_grid_pt = 0  # Time grid pointer
        beg_idx = np.empty(len(beg_times))
        beg_idx[:] = np.NaN
        end_idx = np.copy(beg_idx)
        first_in_interval = True  # Flag variable
        for row_id, row in daily_immed_info.iterrows():
            # First instance in the interval
            if first_in_interval:
                beg_idx[time_grid_pt] = row_id

            if row['Time'] >= beg_times[time_grid_pt] and row['Time'] <= end_times[time_grid_pt]:
                first_in_interval = False
            else:
                # Last instancein the interval
                end_idx[time_grid_pt] =  row_id
                time_grid_pt += 1  # Pointer increment for the next subinterval.
                first_in_interval = True
            
            if time_grid_pt >= len(beg_times):
                break
        
        # 3. Find values for all other columns via index. Then put together in a df.
        frame = {'Date': daily_immed_info.iloc[beg_idx]['Date'].to_numpy(),
                 'EntryDate': daily_immed_info.iloc[beg_idx]['EntryDate'].to_numpy(),
                 'BegTime': beg_times,
                 'EndTime': end_times,
                 'MidPriceBeg': daily_immed_info.iloc[beg_idx]['MidPrice'].to_numpy(),
                 'MidPriceEnd': daily_immed_info.iloc[end_idx]['MidPrice'].to_numpy(),
                 'MidPriceWBeg': daily_immed_info.iloc[beg_idx]['MidPriceW'].to_numpy(),
                 'MidPriceWEnd': daily_immed_info.iloc[end_idx]['MidPriceW'].to_numpy(),
                 'OrderFlowImba': agg_impacts}
        aggre_info = pd.DataFrame(frame)
        
        return aggre_info
        
        
    def _get_aggre_info_r1(self, daily_immed_info: pd.DataFrame) -> pd.DataFrame:
        """
        Generate data for R1 calculation.

        col: 'Date', 'EntryDate', 'BegTime', 'EndTime', 'MidPriceBeg', 'MidPriceEnd', 'MidPriceWBeg', 'MidPriceWEnd',
        'OrderFlowImba'.
        
        Parameters
        ----------
        daily_immed_info : pd.DataFrame
            Concatenated daily info for each trade, should be an output of the get_immediate_info method.

        Returns
        -------
        aggre_info : pd.DataFrame
            Aggregated information by seconds.

        """
        
        agg_impacts = daily_immed_info['TradeVolume'] * daily_immed_info['Buy/Sell']

        frame = {'Date': daily_immed_info['Date'].to_numpy(),
                 'EntryDate': daily_immed_info['EntryDate'].to_numpy(),
                 'BegTime': daily_immed_info['Time'].to_numpy(),
                 'EndTime': daily_immed_info['Time'].to_numpy(),
                 'MidPriceBeg': daily_immed_info['MidPrice'].to_numpy(),
                 'MidPriceEnd': daily_immed_info['MidPrice'].to_numpy(),
                 'MidPriceWBeg': daily_immed_info['MidPriceW'].to_numpy(),
                 'MidPriceWEnd': daily_immed_info['MidPriceW'].to_numpy(),
                 'OrderFlowImba': agg_impacts}
        aggre_info = pd.DataFrame(frame)
        
        return aggre_info
    
    def get_aggre_info_tick(self, daily_immed_info: pd.DataFrame, duration: Union[None, int] = 1) -> pd.DataFrame:
        """
        Get aggregated info for trades by market time (ticks).
        
        The daily immediate info need to have its time continuous, concatenated from immediate info DataFrames, and
        they have to come from one day.
        col: 'Date', 'EntryDate', 'BegTime', 'EndTime', 'MidPriceBeg', 'MidPriceEnd', 'MidPriceWBeg', 'MidPriceWEnd',
        'OrderFlowImba'.

        Parameters
        ----------
        daily_immed_info : pd.DataFrame
            Concatenated daily info for each trade, should be an output of the get_immediate_info method.
        duration : Union[None, int], optional
            The duration of time in ticks (market time). None means no aggregating, use the immediate previous data.
            This number is the same as when you want to calculate R(duration).
            The default is None.

        Returns
        -------
        aggre_info : pd.DataFrame
            Aggregated information by ticks.

        """
        
        # 0. Setting params and intervals up
        if duration is None:  # R1 result follows from another framework.
            aggre_info = self._get_aggre_info_r1(daily_immed_info)
            
            return aggre_info
        else:
            dt = duration
            
        # 1. Loop through the rows.
        agg_impacts = np.zeros(len(daily_immed_info) - dt)
        dates = agg_impacts.copy()
        entry_dates = agg_impacts.copy()
        beg_times = agg_impacts.copy()  # Beginning time for those aggregated ticks
        end_times = agg_impacts.copy()  # Ending time for those aggregated ticks
        mid_prices_beg = agg_impacts.copy()  # Simple medium
        mid_prices_end = agg_impacts.copy()  # Simple medium
        mid_prices_w_beg = agg_impacts.copy()  # Weighted medium
        mid_prices_w_end = agg_impacts.copy()  # Weighted medium
        cache_volume = []  # Volume within each time interval
        cache_buysell = []  # Volume within each time interval
        for row_id, row in daily_immed_info.iterrows():
            for i in range(dt + 1):
                cache_volume.append(daily_immed_info.iloc[i + row_id]['TradeVolume'])
                cache_buysell.append(daily_immed_info.iloc[i + row_id]['Buy/Sell'])
                
            # Dot product to calculate aggregated impact
            agg_impact = np.dot(cache_volume, cache_buysell)
            # Document this impact
            agg_impacts[row_id] = agg_impact
            cache_volume = []
            cache_buysell = []
            
            # Calculate other quantities
            dates[row_id] = int(row['Date'])
            entry_dates[row_id] = int(row['EntryDate'])
            beg_times[row_id] = int(row['Time'])
            end_times[row_id] = int(daily_immed_info.iloc[row_id + dt]['Time'])
            mid_prices_beg[row_id] = row['MidPrice']
            mid_prices_end[row_id] = daily_immed_info.iloc[row_id + dt]['MidPrice']
            mid_prices_w_beg[row_id] = row['MidPriceW']
            mid_prices_w_end[row_id] = daily_immed_info.iloc[row_id + dt]['MidPriceW']
            
            if row_id + 1 >= len(agg_impacts):
                 break
 
        # 3. Find values for all other columns via index. Then put together in a df.
        frame = {'Date': dates,
                 'EntryDate': entry_dates,
                 'BegTime': beg_times,
                 'EndTime': end_times,
                 'MidPriceBeg': mid_prices_beg,
                 'MidPriceEnd': mid_prices_end,
                 'MidPriceWBeg': mid_prices_w_beg,
                 'MidPriceWEnd': mid_prices_w_end,
                 'OrderFlowImba': agg_impacts}
        aggre_info = pd.DataFrame(frame)
        
        return aggre_info