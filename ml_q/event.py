# coding: UTF-8


class Event(object):
    """
    Base class for all the subsequent events.
    """
    pass


class MarketEvent(Event):
    """
    Handles the event of receiving a new market update with
    corresponding bars.

    Parameters:

    datetime : The timestamp at which the signal was generated.
    """

    def __init__(self, datetime):
        self.type = 'MARKET'
        self.datetime = datetime


class SignalEvent(Event):
    """
    Handles the event of sending a Signal from a Strategy object.

    Parameters:

    strategy_id : The unique ID of the strategy sending the signal.
    symbol : The ticker symbol, e.g. '000001'.
    datetime : The timestamp at which the signal was generated.
    signal_type : 'LONG' or 'EXIT'.
    strength : An adjustment factor "suggestion" used to scale
               quantity at the portfolio level. Useful for
               pairs strategies.
    price : The price to buy or sell.
    """

    def __init__(self, strategy_id, symbol, datetime,
                 signal_type, strength, price):

        self.type = 'SIGNAL'
        self.strategy_id = strategy_id
        self.symbol = symbol
        self.datetime = datetime
        self.signal_type = signal_type
        self.strength = strength
        self.price = price


class OrderEvent(Event):
    """
    Handles the event of sending an Order to an execution system.

    Parameters:

    symbol : The ticker symbol, e.g. '000001'.
    datetime : The timestamp at which the order was generated.
    order_type : 'MKT' or 'LMT' for Market or Limit.
    quantity : Non-negative integer for quantity.
    price : The price to buy or sell.
    direction : 'BUY' or 'SELL' for long or exit.
    """

    def __init__(self, symbol, datetime, order_type,
                 quantity, price, direction):

        self.type = 'ORDER'
        self.symbol = symbol
        self.datetime = datetime
        self.order_type = order_type
        self.quantity = quantity
        self.price = price
        self.direction = direction

    def print_order(self):
        """
        Outputs the values within the Order.
        """
        print "Order: Symbol=%s, Type=%s, Quantity=%s, Price=%s, Direction=%s" % \
            (self.symbol, self.order_type, self.quantity, self.price, self.direction)


class FillEvent(Event):
    """
    Encapsulates the notion of a Filled Order, just like returned
    from a brokerage. Stores the quantity of a stock actually hold
    with what price, and the commission of this trade.
    If commission is not provided, the Fill object will calculate
    it based on the trade size.

    Parameters:

    timeindex : When the order was filled.
    symbol : The ticker symbol which was filled.
    quantity : The filled quantity.
    direction : Direction of fill, 'BUY' or 'SELL'
    price : The fill price in RMB.
    commission : An optional commission.
    """

    def __init__(self, timeindex, symbol, quantity, direction,
                 price, commission=None):

        self.type = 'FILL'
        self.timeindex = timeindex
        self.symbol = symbol
        self.quantity = quantity
        self.direction = direction
        self.price = price

        # Calculate commission
        if commission is None and direction == 'BUY':
            self.commission = max(self.price * self.quantity * 0.003, 5.0)
        elif commission is None and direction == 'SELL':
            self.commission = max(self.price * self.quantity * 0.004, 5.0)
        else:
            self.commission = commission
