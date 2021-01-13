# -*- coding: utf-8 -*-
"""
Examples showing how to use top_order_book.py

@author: Hansen
"""
from typing import Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import top_order_book as tob

#%% Test, read data
data = pd.read_csv(r'D:\MFM2021\Data\CMD_BBO\20201019\xcme-bbo-es-fut-20201019-r-00095_8.csv')
mini_data = data
TV = tob.TradeVisualize(mini_data, lump_trades=False)
my_data = TV.data
#%%

# Index where trade happens
trade_idx = TV.get_trades()

# Immediate prevailing bid and ask index
prev_ab_df = TV.get_prev_ab_idx(trade_idx, handle_conti_trade='bidask')

#%% Get the immediate quotes, volume etc right before the trade for calculating R(1), page 213 (11.5)

# weighted mid prices
r1_info = TV.get_immediate_info(prev_ab_df, mid_type='weighted')

# simple mid prices
r1_info_simple = TV.get_immediate_info(prev_ab_df, mid_type='simple')
summary = r1_info.describe()

#%% Get aggregated impact
# For calculating R1, don't aggregate
aggre_info_r1 = TV.get_aggre_info(r1_info, duration=None)
# Aggregating 5 seconds
aggre_info_5 = TV.get_aggre_info(r1_info, duration=5)
#%% Plot each trade for visualization
# Note: Don't plot too much, generally around 200-400 data points works best for visualization
plt.style.use('seaborn')
fig, ax = plt.subplots(dpi=300)
ax = TV.plot_by_trade(r1_info, ax, plot_mid_price = True)
ax.grid()
plt.tight_layout()
plt.show()


