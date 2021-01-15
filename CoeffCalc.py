# -*- coding: utf-8 -*-
"""
Created on Thurs Jan 14 22:12:35 2021
@author: Stark
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os,sys
from scipy.optimize import curve_fit
import data_parse 
import top_order_book as tob

# Define functions for optimization
def scale_function(x, alpha, beta):
    """Scaling function
    """
    return x / np.power(1 + np.power(abs(x), alpha), beta/alpha)

def func(qT, chi, kappa, alpha, beta):
    """Function used for optimization
    """
    q = qT[0]
    T = qT[1]
    x = q / np.power(T, kappa)
    return np.power(T, chi) * scale_function( x, alpha, beta ) 


def make_daily_trades_info(directory: str) -> pd.DataFrame:
    """
    Concatenate all .csv files into a single trades_info DataFrame

    Parameters
    ----------
    directory : str
        The file directory that stores the daily data csv. Example
        r"D://MFM2021//Data//CMD_BBO//20201020//"

    Returns
    -------
    daily_trades_info : pd.DataFrame
        Daily trades info.

    """
    dps = data_parse.DataParse()
    daily_data = dps.read_folder(directory)
    filt_daily_data = dps.filter_trade_hour(data=daily_data)
    TV = tob.TradeVisualize(filt_daily_data, lump_trades=False)
    my_data = TV.data
    trade_idx = TV.get_trades()
    prev_ab_df = TV.get_prev_ab_idx(trade_idx, handle_conti_trade='bidask')
    trades_info = TV.get_immediate_info(prev_ab_df)
    
    return trades_info

def _calc_size_frame(df,by='OrderFlowImba'):
    #grouped_DV_5 = aggre_info_tick_r5.groupby(by='OrderFlowImba')
    return df.groupby(by=by).size()

def _calc_fn_frame(n,srs_n):
    fn = srs_n.to_frame().reset_index()
    Tn = pd.Series(n, index=fn.index)
    fn['T'] = Tn
    return fn

def _calc_qi(aggre_info, daily_vol):
    return aggre_info['OrderFlowImba'] / daily_vol
    
def main(path_in, path_out,durations=[5, 10, 20], suffix = ['_popt','_pcov']):
    daily_trades_info = make_daily_trades_info(directory=path_in)
    TV = tob.TradeVisualize()
    aggre_info_dict = {duration: TV.get_aggre_info_tick_nonconsec(daily_trades_info, duration)
                  for duration in durations}
    srs_dict = dict(map(lambda x: (x[0], _calc_size_frame(x[1])), aggre_info_dict.items()))
    fn_dict = dict(map(lambda x: (x[0], _calc_fn_frame(x[0],x[1])), srs_dict.items()))
    for i , df in fn_dict.items():
        if i == durations[0]:
            frame = df
        else:
            frame.append(df,ignore_index = True)
    daily_vol = np.sum(daily_trades_info['TradeVolume'])
    Qi_dict = dict(map(lambda x: (x[0], _calc_qi(x[1],daily_vol)), aggre_info_dict.items()))
    R1 = (daily_trades_info['MidPrice'].shift(1) - daily_trades_info['MidPrice'])  * daily_trades_info['Buy/Sell']
    R1[0] = 0
    popt, pcov = curve_fit(func, np.transpose(frame.iloc[:, 1:].to_numpy()), frame.iloc[:, 0].to_numpy(), bounds=(0, np.inf))
    pd.DataFrame(popt,index=["X","K","A","B"]).T.to_csv(path_out+suffix[0]+".csv",index = False)
    pd.DataFrame(pcov).to_csv(path_out+suffix[1]+".csv")

if __name__ == '__main__':
    main(sys.argv[1],sys.argv[2])
