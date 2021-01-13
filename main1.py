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
mini_data = data.iloc[:10000]
TV = tob.TradeVisualize(mini_data, lump_trades=False)
my_data = TV.data
#%%

# Index where trade happens
trade_idx = TV.get_trades()
# Immediate prevailing bid and ask index
prev_ab_df = TV.get_prev_ab_idx(trade_idx, handle_conti_trade='trade')

#%% Get the immediate quotes, volume etc right before the trade for calculating R(1), page 213 (11.5)

r1_info = TV.get_immediate_info(prev_ab_df, mid_type='weighted')
summary = r1_info.describe()

#%% Plot
plt.style.use('seaborn')
fig, ax = plt.subplots(dpi=300)
ax = TV.plot_by_trade(r1_info, ax)
plt.tight_layout()
plt.show()


