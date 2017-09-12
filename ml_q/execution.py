# coding: UTF-8

from event import FillEvent


class ExecutionHandler(object):
    """
    Class to execute the trading and generate fill event for back test.

    Parameters:

    events : The Event Queue object.
    """
    def __init__(self, event):

        self.events = event

    def execute_order(self, order):
        """
        Execute the order in real trading.
        """
        pass

    def update_order(self, event):
        """
        Acts on an OrderEvent to generate new FillEvent.
        """
        if event.type == 'ORDER':
            fill_event = self.generate_fill_event(event)
            self.events.put(fill_event)

    def generate_fill_event(self, order):
        """
        Acts on an OrderEvent to generate new FillEvent.
        """
        timeindex = order.datetime
        symbol = order.symbol
        quantity = order.quantity
        direction = order.direction
        price = order.price

        fill = FillEvent(timeindex, symbol, quantity, direction, price)

        return fill