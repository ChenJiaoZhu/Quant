# Coding: UTF-8
#################### MA ####################
import pandas as pd

# Simple Moving Average
def SMA(data, ndays=5):
    SMA = pd.Series(data['Close'].rolling(ndays).mean(), name=str(ndays)+'_SMA')
    data = data.join(SMA)
    return data

# Exponentially-weighted Moving Average
def EWMA(data, ndays=10):
    EWMA = pd.Series(data['Close'].ewm(span = ndays, min_periods = ndays-1).mean(),
                    name = str(ndays)+'_EWMA')
    data = data.join(EWMA)
    return data

#################### Test ####################
# from pandas_datareader import data as web
# import matplotlib.pyplot as plt
#
# # Retrieve the Nifty data from Yahoo finance:
# data = web.DataReader('^NSEI',data_source='yahoo',start='1/1/2013', end='1/1/2016')
# data = pd.DataFrame(data)
# close = data['Close']
#
# # Compute the 50-day SMA for NIFTY
# n = 50
# SMA_NIFTY = SMA(data,n)
# SMA_NIFTY = SMA_NIFTY.dropna()
# SMA = SMA_NIFTY[str(n)+'_SMA']
#
# # Compute the 200-day EWMA for NIFTY
# ew = 200
# EWMA_NIFTY = EWMA(data,ew)
# EWMA_NIFTY = EWMA_NIFTY.dropna()
# EWMA = EWMA_NIFTY[str(ew)+'_EWMA']
#
# # Plotting the NIFTY Price Series chart and Moving Averages below
# plt.figure(figsize=(9,5))
# plt.plot(data['Close'],lw=1, label='NSE Prices')
# plt.plot(SMA,'g',lw=1, label='50-day SMA (green)')
# plt.plot(EWMA,'r', lw=1, label='200-day EWMA (red)')
# plt.legend(loc=2,prop={'size':11})
# plt.grid(True)
# plt.setp(plt.gca().get_xticklabels(), rotation=30)
# plt.show()