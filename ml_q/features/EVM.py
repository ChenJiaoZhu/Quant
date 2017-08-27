# Coding: UTF-8
#################### EVM ####################
import pandas as pd

# Ease of Movement
def EVM(data, ndays=14):
    dm = ((data['High'] + data['Low']) / 2) - (
        (data['High'].shift(1) + data['Low'].shift(1)) / 2)
    br = (data['Volume'] / 100000000) / ((data['High'] - data['Low']))
    EVM = dm / br
    EVM_MA = pd.Series(EVM.rolling(ndays).mean(), name=str(ndays)+'_EVM')
    data = data.join(EVM_MA)
    return data

#################### Test ####################
# from pandas_datareader import data as web
# import matplotlib.pyplot as plt
#
# # Retrieve the AAPL data from Yahoo finance:
# data = web.DataReader('AAPL', data_source='yahoo', start='1/1/2015', end='1/1/2016')
# data = pd.DataFrame(data)
#
# # Compute the 14-day Ease of Movement for AAPL
# n = 14
# AAPL_EVM = EVM(data, n)
# EVM = AAPL_EVM[str(n) + '_EVM']
#
# # Plotting the Price Series chart and the Ease Of Movement below
# fig = plt.figure(figsize=(7, 5))
# ax = fig.add_subplot(2, 1, 1)
# ax.set_xticklabels([])
# plt.plot(data['Close'], lw=1)
# plt.title('AAPL Price Chart')
# plt.ylabel('Close Price')
# plt.grid(True)
# bx = fig.add_subplot(2, 1, 2)
# plt.plot(EVM, 'k', lw=0.75, linestyle='-', label='EVM(14)')
# plt.legend(loc=2, prop={'size': 9})
# plt.ylabel('EVM values')
# plt.grid(True)
# plt.setp(plt.gca().get_xticklabels(), rotation=30)
# plt.show()