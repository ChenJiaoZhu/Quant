# Coding: UTF-8
#################### ROC ####################
import pandas as pd

# Rate of Change
def ROC(data, ndays=12):
    N = data['Close'].diff(ndays)
    D = data['Close'].shift(ndays)
    ROC = pd.Series(N / D, name=str(ndays)+'_ROC')
    data = data.join(ROC)
    return data

#################### Test ####################
# from pandas_datareader import data as web
# import matplotlib.pyplot as plt
#
# # Retrieve the NIFTY data from Yahoo finance:
# data = web.DataReader('^NSEI', data_source='yahoo', start='6/1/2015', end='1/1/2016')
# data = pd.DataFrame(data)
#
# # Compute the 5-period Rate of Change for NIFTY
# n = 5
# NIFTY_ROC = ROC(data, n)
# ROC = NIFTY_ROC[str(n)+'_ROC']
#
# # Plotting the Price Series chart and the Rate of Change below
# fig = plt.figure(figsize=(7, 5))
# ax = fig.add_subplot(2, 1, 1)
# ax.set_xticklabels([])
# plt.plot(data['Close'], lw=1)
# plt.title('NSE Price Chart')
# plt.ylabel('Close Price')
# plt.grid(True)
# bx = fig.add_subplot(2, 1, 2)
# plt.plot(ROC, 'k', lw=0.75, linestyle='-', label='ROC')
# plt.legend(loc=2, prop={'size': 9})
# plt.ylabel('ROC values')
# plt.grid(True)
# plt.setp(plt.gca().get_xticklabels(), rotation=30)
# plt.show()
