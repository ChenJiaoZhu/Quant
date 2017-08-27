# Coding: UTF-8
#################### RSV ####################
import pandas as pd

# Raw Stochastic Value
def RSV(data, ndays=9):
    N = data['Close']
    PL = data['Low'].rolling(ndays).min()
    PH = data['High'].rolling(ndays).max()
    RSV = pd.Series((N-PL)/(PH-PL), name=str(ndays)+'_RSV')
    data = data.join(RSV)
# KDJ
    K_, D_ = 0.5, 0.5
    K, D = [], []
    for rsv in RSV:
        if rsv!=rsv:
            K.append(0.5)
            D.append(0.5)
        else:
            K_ = (2*K_)/3 + rsv/3
            K.append(K_)
            D_ = (2*D_)/3 + K_/3
            D.append(D_)
    K = pd.Series(K, index=RSV.index, name=str(ndays)+'_K')
    data = data.join(K)
    D = pd.Series(D, index=RSV.index, name=str(ndays) + '_D')
    data = data.join(D)
    J = pd.Series(3*K-2*D, name=str(ndays) + '_J')
    data = data.join(J)
    return data

#################### Test ####################
# from pandas_datareader import data as web
# import matplotlib.pyplot as plt
#
# # Retrieve the NIFTY data from Yahoo finance:
# data = web.DataReader('^NSEI', data_source='yahoo', start='6/1/2015', end='1/1/2016')
# data = pd.DataFrame(data)
#
# # Compute the 5-period Raw Stochastic Value for NIFTY
# n = 5
# NIFTY_RSV = RSV(data, n)
# RSV = NIFTY_RSV[str(n)+'_RSV']
#
# # Plotting the Price Series chart and the Raw Stochastic Value below
# fig = plt.figure(figsize=(7, 5))
# ax = fig.add_subplot(2, 1, 1)
# ax.set_xticklabels([])
# plt.plot(data['Close'], lw=1)
# plt.title('NSE Price Chart')
# plt.ylabel('Close Price')
# plt.grid(True)
# bx = fig.add_subplot(2, 1, 2)
# plt.plot(RSV, 'k', lw=0.75, linestyle='-', label='RSV')
# plt.legend(loc=2, prop={'size': 9})
# plt.ylabel('RSV values')
# plt.grid(True)
# plt.setp(plt.gca().get_xticklabels(), rotation=30)
# plt.show()
