# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 22:16:51 2021

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
#%% 
trades_idx = TV.get_trades()
prev_ab_df = TV.get_prev_ab_idx(trades_idx)
#%% testing plot
plt.style.use('default')
fig, ax = plt.subplots(dpi=300)
ax = TV.plot_abt(prev_ab_df, ax)
plt.tight_layout()
plt.show()