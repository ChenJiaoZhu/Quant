# Coding: UTF-8


def benchmark_return(index='000300', backtest_date='2016-08-01', end_date='2017-08-01'):

    import tushare as ts

    hs = ts.get_k_data(index, index=True)
    hs = hs[hs['date']>=backtest_date]
    hs = hs[hs['date']<end_date]
    returns = hs['close'].pct_change()
    returns.loc[returns.index[0], 'returns'] = 0.0
    returns['equity_curve'] = (1.0 + returns['returns']).cumprod()

    return returns


def create_sharpe_ratio(returns, freq, rf=0.03):

    if freq == 'day':
        sharpe_ratio = (returns.mean()*245 - rf) / (returns.std()*(245**0.5))