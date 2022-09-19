# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 10:35:23 2022

@author: uqymao1
"""

#%%
import os
import numpy as np
from datetime import datetime

# download and parse input data
data_orig = np.load(r'data/data.npy')

with open(r'data/dates.txt') as f:
    dates = f.read().split('\n')
    dates = [datetime.strptime(d, '%Y-%m-%d') for d in dates if len(d) > 0]
    
#%%
from bfast.monitor.utils import crop_data_dates
start_hist = datetime(2002, 1, 1)
start_monitor = datetime(2010, 1, 1)
end_monitor = datetime(2018, 1, 1)
data, dates = crop_data_dates(data_orig, dates, start_hist, end_monitor)
print("First date: {}".format(dates[0]))
print("Last date: {}".format(dates[-1]))
print("Shape of data array: {}".format(data.shape))

#%% 
import bfast
import datetime
from bfast import BFAST
from bfast import BFASTMonitor

def year_fraction(date):
    start = datetime.date(date.year, 1, 1).toordinal()
    year_length = datetime.date(date.year+1, 1, 1).toordinal() - start
    return date.year + float(date.toordinal() - start) / year_length
dates_fractional = np.array([year_fraction(i) for i in dates])


# model = BFASTMonitor(
#             start_monitor,
#             freq=365,
#             k=3,
#             hfrac=0.25,
#             trend=False,
#             level=0.05,
#             backend='python',
#             device_id=0,
#         )

model = BFAST(
    frequency=365,
    h=0.15,
    season_type="dummy",
    max_iter=10,
    max_breaks=None,
    level=0.05,
    verbose=1
        )


model.fit(data, dates_fractional)