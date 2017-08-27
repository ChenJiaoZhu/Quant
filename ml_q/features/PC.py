# Coding: UTF-8
#################### PC ####################
import pandas as pd

# Price Channels
def PC(data, ndays=20):
    PL = pd.Series(data['Low'].rolling(ndays).min(), name=str(ndays)+'_PL')
    data = data.join(PL)
    PH = pd.Series(data['High'].rolling(ndays).max(), name=str(ndays)+'_PH')
    data = data.join(PH)
    PC = pd.Series((PH+PL)/2, name=str(ndays)+'_PC')
    data = data.join(PC)
# OD
    O = data['Open'].diff(1)
    OD = pd.Series(O/(PH - PL), name=str(ndays) + '_OD')
    data = data.join(OD)
    return data

#################### Test ####################
# from pandas_datareader import data as web
#
# # Retrieve the NIFTY data from Yahoo finance:
# data = web.DataReader('^NSEI', data_source='yahoo', start='6/1/2015', end='1/1/2016')
# data = pd.DataFrame(data)
#
# # Compute the 20-period Price Channels for NIFTY
# n = 20
# NIFTY_PC = PC(data, n)

