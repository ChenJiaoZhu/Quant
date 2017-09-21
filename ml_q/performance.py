# Coding: UTF-8

import pandas as pd


def benchmark_return(index='000300', backtest_date='2016-08-01', end_date='2017-08-01'):
    """
    Gets the return series of given benchmark.
    """
    import tushare as ts

    hs = ts.get_k_data(index, index=True)
    hs = hs[hs['date']>=backtest_date]
    hs = hs[hs['date']<end_date]
    hs.set_index('date', inplace=True)
    returns = hs['close'].pct_change()
    returns[0] = 0.0
    returns = (1.0 + returns).cumprod()

    return returns, index


def create_sharpe_ratio(returns, freq='day', rf=0.03):
    """
    Calculates the sharpe ratio of given return series.
    """
    if freq == 'day':
        sharpe_ratio = (returns.mean()*245 - rf) / (returns.std()*(245**0.5))

        return sharpe_ratio


def create_drawdowns(pnl):
    """
    Calculates the draw-down of given equity curve and
    its max draw-down as well as corresponding duration.
    """
    drawdown = pnl.copy().rename('drawdown')
    period = []
    for i in pnl.index:
        line = {}
        before = pnl[pnl.index<=i]
        drawdown[i] = 1 - (pnl[i] / before.max())
        last_days = before[before==before.max()].index
        last_day = ''
        for n in last_days:
            last_day += n + ', '
        line['datetime'] = i
        line['past_max'] = last_day[:-2]
        period.append(line)

    period = pd.DataFrame(period)
    period.set_index('datetime', inplace=True)

    max_dd = drawdown.max()

    dd_duration = []
    last_ = drawdown[drawdown == drawdown.max()].index
    for l in last_:
        dd_duration.append([period['past_max'][l], l])

    return drawdown, max_dd, dd_duration
