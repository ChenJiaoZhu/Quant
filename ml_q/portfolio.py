# coding: UTF-8

import pandas as pd
from event import OrderEvent, SignalEvent


class Portfolio(object):
    """
    The Portfolio class handles the positions and market
    value of all instruments at a resolution of a "bar",
    i.e. secondly, minutely, 5-min, 30-min, 60 min or EOD.

    The positions DataFrame stores a time-index of the
    quantity of positions held.

    The holdings DataFrame stores the cash and total market
    holdings value of each symbol for a particular
    time-index, as well as the percentage change in
    portfolio total across bars.

    Parameters:

    bars : The DataHandler object with current market data.
    events : The Event Queue object.
    start_date : The start date (bar) of the portfolio.
    backtest_date : The start datetime of back-test.
    initial_capital : The starting capital in RMB.
    """
    def __init__(self, bars, events, start_date, backtest_date, initial_capital=100000.0):

        self.bars = bars
        self.events = events
        self.symbol_list = self.bars.symbol_list
        self.start_date = start_date
        self.backtest_date = backtest_date
        self.initial_capital = initial_capital

        self.all_positions = self._construct_all_positions()
        self.current_positions = dict((s, 0) for s in self.symbol_list)

        self.all_holdings = self._construct_all_holdings()
        self.current_holdings = self._construct_current_holdings()

    def _construct_all_positions(self):
        """
        Constructs the positions list using the backtest_date object
        to determine when the time index will begin.
        """
        d = dict((s, 0) for s in self.symbol_list)
        d['datetime'] = self.backtest_date
        return [d]

    def _construct_all_holdings(self):
        """
        Constructs the holdings list using the backtest_date object
        to determine when the time index will begin.
        """
        d = dict((s, 0.0) for s in self.symbol_list)
        d['datetime'] = self.backtest_date
        d['cash'] = self.initial_capital
        d['commission'] = 0.0
        d['total'] = self.initial_capital
        return [d]

    def _construct_current_holdings(self):
        """
        This constructs the dictionary which will hold the instantaneous
        value of the portfolio across all symbols.
        """
        d = dict((s, 0.0) for s in self.symbol_list)
        d['cash'] = self.initial_capital
        d['commission'] = 0.0
        d['total'] = self.initial_capital
        return d

    def update_signal(self, event):
        """
        Acts on a SignalEvent to generate new orders based on the portfolio logic.
        """
        if event.type == 'SIGNAL':
            order_event = self.generate_naive_order(event)
            self.events.put(order_event)

    def generate_naive_order(self, signal):
        """
        Simply files an Order object as a constant quantity sizing of the signal
        object, without risk management or position sizing considerations.
        """
        order = None

        symbol = signal.symbol
        datetime = signal.datetime
        direction = signal.signal_type
        strength = signal.strength
        price = signal.price

        lmt_quantity = 100
        cur_quantity = self.current_positions[symbol]
        order_type = 'LMT'

        if direction == 'LONG':
            order = OrderEvent(symbol, datetime, order_type, lmt_quantity, price, 'BUY')
        if direction == 'EXIT':
            if cur_quantity < 100:
                print 'Current quantity: %s is smaller than 100.' % cur_quantity
                raise KeyError
            else:
                order = OrderEvent(symbol, datetime, order_type, cur_quantity, price, 'SELL')

        return order

    def update_fill(self, event):
        """
        Updates the current positions and holdings of portfolio from a FillEvent.
        """
        if event.type == 'FILL':
            self.update_positions_from_fill(event)
            self.update_holdings_from_fill(event)

    def update_positions_from_fill(self, fill):
        """
        Takes a Fill object and updates the position matrix to reflect the new position.
        """
        fill_dir = 0
        if fill.direction == 'BUY':
            fill_dir = 1
        if fill.direction == 'SELL':
            fill_dir = -1

        self.current_positions[fill.symbol] += fill_dir*fill.quantity
        self.current_positions['datetime'] = fill.timeindex

    def update_holdings_from_fill(self, fill):
        """
        Takes a Fill object and updates the holdings matrix to reflect the holdings value.
        """
        fill_dir = 0
        if fill.direction == 'BUY':
            fill_dir = -1
        if fill.direction == 'SELL':
            fill_dir = 1

        cost = fill_dir * fill.price * fill.quantity

        self.current_holdings[fill.symbol] += (cost - fill.commission)
        self.current_holdings['commission'] += fill.commission
        self.current_holdings['cash'] += (cost - fill.commission)

        self.current_holdings['datetime'] = fill.timeindex

    def update_timeindex(self):
        """
        Adds a complete performance record of one day to the performance matrix.
        This contains calculating the total market value of this day.
        """
        self.all_positions.append(self.current_positions.copy())

        self.current_holdings['total'] = self.current_holdings['cash']
        for s in self.symbol_list:
            # Approximates the real value
            market_value = self.current_positions[s] * \
                self.bars.get_latest_bar_value(s, 'True_close')
            self.current_holdings['total'] += market_value

        # Append the current holdings
        self.all_holdings.append(self.current_holdings.copy())

    def sell_all_holdings(self, date):
        """
        Sells all the holdings by generating signal events to get the final return.
        """
        for s in self.symbol_list:
            if self.current_positions[s] > 0:
                price = self.bars.get_latest_bar_value(s, 'True_close')
                sell_all_event = SignalEvent(1, s, date, 'EXIT', 1.0, price)
                self.events.put(sell_all_event)

    def create_equity_curve_dataframe(self):
        """
        Creates a pandas DataFrame from the all_holdings list of dictionaries.
        """
        curve = pd.DataFrame(self.all_holdings)
        curve.set_index('datetime', inplace=True)
        curve['returns'] = curve['total'].pct_change()
        curve['equity_curve'] = (1.0 + curve['returns']).cumprod()
        self.equity_curve = curve
