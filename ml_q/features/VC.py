# Coding: UTF-8
#################### VC ####################
import pandas as pd

# Volume Channels
def VC(data, ndays=20):
    VL = pd.Series(data['Volume'].rolling(ndays).min(), name=str(ndays)+'_VL')
    data = data.join(VL)
    VH = pd.Series(data['Volume'].rolling(ndays).max(), name=str(ndays)+'_VH')
    data = data.join(VH)
    VC = pd.Series((VH+VL)/2, name=str(ndays)+'_VC')
    data = data.join(VC)
# VD
    V = data['Volume'].diff(1)
    VD = pd.Series(V/(VH - VL), name=str(ndays) + '_VD')
    data = data.join(VD)
    return data

#################### Test ####################
# from pandas_datareader import data as web
#
# # Retrieve the NIFTY data from Yahoo finance:
# data = web.DataReader('^NSEI', data_source='yahoo', start='6/1/2015', end='1/1/2016')
# data = pd.DataFrame(data)
#
# # Compute the 20-period Volume Channels for NIFTY
# n = 20
# NIFTY_VC = VC(data, n)

