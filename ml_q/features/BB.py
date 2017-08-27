# coding: UTF-8
#################### BB ####################
import pandas as pd

# Bollinger Bands
def BB(data, ndays=20, type='Close'):
    MA = data[type].rolling(ndays).mean()
    SD = data[type].rolling(ndays).std()

    b1 = MA + (2 * SD)
    B1 = pd.Series(b1, name = str(ndays) + '_U_BB_'+ type)
    data = data.join(B1)

    b2 = MA - (2 * SD)
    B2 = pd.Series(b2, name = str(ndays) + '_L_BB_'+ type)
    data = data.join(B2)

    return data

#################### Test ####################
# from pandas_datareader import data as web
#
# # Retrieve the Nifty data from Yahoo finance:
# data = web.DataReader('^NSEI',data_source='yahoo',start='1/1/2010', end='1/1/2016')
# data = pd.DataFrame(data)
#
# # Compute the Bollinger Bands for NIFTY using the 50-day Moving average
# n = 50
# NIFTY_BBANDS = BB(data, n)
# print NIFTY_BBANDS.tail()

