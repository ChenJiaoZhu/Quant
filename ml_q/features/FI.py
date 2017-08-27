# Coding: UTF-8
#################### FI ####################
import pandas as pd

# Force Index
def FI(data, ndays=13):
    FI = data['Close'].diff(1) * data['Volume']
    FI_MA = pd.Series(FI.rolling(ndays).mean(), name=str(ndays)+'_FI')
    data = data.join(FI_MA)
    return data

#################### Test ####################
# from pandas_datareader import data as web
#
# # Retrieve the Apple data from Yahoo finance:
# data = web.DataReader('AAPL',data_source='yahoo',start='1/1/2010', end='1/1/2016')
# data = pd.DataFrame(data)
#
# # Compute the Force Index for Apple
# n = 13
# AAPL_ForceIndex = FI(data,n)
# print AAPL_ForceIndex
