# Coding: UTF-8
#################### MMT ####################
import pandas as pd

# Time Lag
def TL(data, lags=21):
    rtn = data["Close"].pct_change() * 100.0
    for i in range(1, lags):
        lag = pd.Series(rtn.shift(i-1), name='Lag' + str(i))
        data = data.join(lag)
    return data

# Momentum of Return
def MMT(data, ndays=5):
    rtn = data["Close"].pct_change() * 100.0
    MEAN_RTN = pd.Series(rtn.rolling(ndays).mean(), name=str(ndays)+'_MEAN_RTN')
    data = data.join(MEAN_RTN)
    VAR_RTN = pd.Series(rtn.rolling(ndays).var(), name=str(ndays) + '_VAR_RTN')
    data = data.join(VAR_RTN)
    SKEW_RTN = pd.Series(rtn.rolling(ndays).skew(), name=str(ndays) + '_SKEW_RTN')
    data = data.join(SKEW_RTN)
    KURT_RTN = pd.Series(rtn.rolling(ndays).kurt(), name=str(ndays) + '_KURT_RTN')
    data = data.join(KURT_RTN)
    return data