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
mini_data = data.iloc[:100000]
TV = tob.TradeVisualize(mini_data)
my_data = TV.data
#%%

# Index where trade happens
trade_idx = TV.get_trades()
# Immediate prevailing bid and ask index
prev_ab_df = TV.get_prev_ab_idx(trade_idx)

#%% Get the immediate quotes, volume etc right before the trade for calculating R(1), page 213 (11.5)

r1_info = TV.get_immediate_info(prev_ab_df)
#%% testing plot
plt.style.use('default')
fig, ax = plt.subplots(dpi=300)
ax = TV.plot_abt(prev_ab_df, ax)
plt.tight_layout()
plt.show()
