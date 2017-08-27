# Coding: UTF-8
#################### PVC ####################
import pandas as pd

# Price Volume Change
def PVC(data):
    OD = data['Open'].diff(1)
    OS = data['Open'].shift(1)
    POC = pd.Series(OD / OS, name='POC')
    data = data.join(POC)
    HD = data['High'].shift(1).diff(1)
    HS = data['High'].shift(2)
    PHC = pd.Series(HD / HS, name='PHC')
    data = data.join(PHC)
    LD = data['Low'].shift(1).diff(1)
    LS = data['Low'].shift(2)
    PLC = pd.Series(LD / LS, name='PLC')
    data = data.join(PLC)

    VD = data['Volume'].diff(1)
    VS = data['Volume'].shift(1)
    VC = pd.Series(VD / VS, name='VC')
    data = data.join(VC)
    return data

#################### Test ####################
# from pandas_datareader import data as web
#
# # Retrieve the NIFTY data from Yahoo finance:
# data = web.DataReader('^NSEI', data_source='yahoo', start='6/1/2015', end='1/1/2016')
# data = pd.DataFrame(data)
#
# # Compute the 5-period Rate of Change for NIFTY
# NIFTY_PVC = PVC(data)

