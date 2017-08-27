# Coding: UTF-8
#################### MACD ####################
import pandas as pd

# Moving Average Convergence/Divergence
def MACD(data):
    SEWMA = data['Close'].ewm(span = 12, min_periods = 11).mean()
    LEWMA = data['Close'].ewm(span = 26, min_periods = 25).mean()
    MACD = pd.Series(SEWMA-LEWMA, name='MACD')
    data = data.join(MACD)
    return data

def EMA_MACD(data, ndays=9):
    SEWMA = data['Close'].ewm(span = 12, min_periods = 11).mean()
    LEWMA = data['Close'].ewm(span = 26, min_periods = 25).mean()
    MACD = SEWMA - LEWMA
    EMA_MACD = pd.Series(MACD.ewm(span=ndays, min_periods=ndays-1).mean(),
                     name=str(ndays) + '_EMA_MACD')
    data = data.join(EMA_MACD)
    MACDH = pd.Series(MACD-EMA_MACD, name=str(ndays) + '_MACDH')
    data = data.join(MACDH)
    return data

#################### Test ####################
# from pandas_datareader import data as web
# import matplotlib.pyplot as plt
#
# # Retrieve the Nifty data from Yahoo finance:
# data = web.DataReader('^NSEI',data_source='yahoo',start='1/1/2013', end='1/1/2016')
# data = pd.DataFrame(data)
#
# # Compute the MACD for NIFTY
# MACD_NIFTY = MACD(data)
# MACD = MACD_NIFTY['MACD']
#
# # Compute the 9-day EMA_MACD for NIFTY
# ew = 9
# EMA_MACD_NIFTY = EMA_MACD(MACD_NIFTY,ew)
# EMA_MACD = EMA_MACD_NIFTY[str(ew)+'_EMA_MACD']
# MACDH = EMA_MACD_NIFTY[str(ew)+'_MACDH']
#
# # Plotting the NIFTY Price Series chart and Moving Average Convergence/Divergence below
# plt.figure(figsize=(9,5))
# plt.plot(data['Close']/80,lw=1, label='NSE Prices')
# plt.plot(MACD,'g',lw=1, label='MACD (green)')
# plt.plot(EMA_MACD,'r', lw=1, label='9-day EMA_MACD (red)')
# plt.plot(MACDH,'k', lw=1, label='9-day MACDH (black)')
# plt.legend(loc=2,prop={'size':11})
# plt.grid(True)
# plt.setp(plt.gca().get_xticklabels(), rotation=30)
# plt.show()