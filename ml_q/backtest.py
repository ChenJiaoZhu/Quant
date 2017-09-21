# coding: UTF-8

try:
    import Queue as queue
except ImportError:
    import queue
import time
import pprint


class Backtest(object):
    """
    Encapsulates the settings and components for carrying out
    an event-driven backtest.

    Parameters:

    symbol_list : The list of symbol strings.
    intial_capital : The starting capital for the portfolio.
    heartbeat : Backtest "heartbeat" in seconds
    start_date : The start datetime of the strategy.
    backtest_date : The start datetime of back-test.
    data_handler : (Class) Handles the market data feed.
    execution_handler : (Class) Handles the orders/fills for trades.
    portfolio : (Class) Keeps track of all portfolio information.
    strategy : (Class) Generates signals based on market data.
    threshold : One trade return of each stock which decides
                the price to buy or sell.
    per_return : Expect return of each stock.
    ndays : How many days look back.
    idays : How many days the price decreased.
    """
    def __init__(self, symbol_list, initial_capital, heartbeat,
                 start_date, backtest_date, data_handler,
                 execution_handler, portfolio, strategy,
                 threshold, per_return, ndays, idays):

        self.symbol_list = symbol_list
        self.initial_capital = initial_capital
        self.heartbeat = heartbeat
        self.start_date = start_date
        self.backtest_date = backtest_date

        self.data_handler_cls = data_handler
        self.execution_handler_cls = execution_handler
        self.portfolio_cls = portfolio
        self.strategy_cls = strategy
        self.threshold = threshold
        self.per_return = per_return
        self.ndays = ndays
        self.idays = idays

        self.events = queue.Queue()

        self.signals = 0
        self.orders = 0
        self.fills = 0

        self._generate_trading_instances()

    def _generate_trading_instances(self):
        """
        Generates the trading instance objects from their class types.
        """
        print "Creating DataHandler, Strategy, Portfolio and ExecutionHandler..."
        self.data_handler = self.data_handler_cls(self.events,
                                                  self.start_date,
                                                  self.backtest_date,
                                                  self.symbol_list)
        self.strategy = self.strategy_cls(self.data_handler,
                                          self.events,
                                          self.ndays,
                                          self.idays,
                                          self.threshold,
                                          self.per_return)
        self.portfolio = self.portfolio_cls(self.data_handler,
                                            self.events,
                                            self.start_date,
                                            self.backtest_date,
                                            self.initial_capital)
        self.execution_handler = self.execution_handler_cls(self.events)

        self.data_handler.get_portfolio(self.portfolio)

    def _run_backtest(self):
        """
        Executes the backtest.
        """
        i = 0
        while True:
            i += 1
            # Update the market bars
            if self.data_handler.continue_backtest == True:
                self.data_handler.update_bars(i)
            else:
                break

            # Handle the events
            while True:
                try:
                    event = self.events.get(False)
                except queue.Empty:
                    self.portfolio.update_timeindex(i)
                    break
                else:
                    if event is not None:
                        if event.type == 'MARKET':
                            self.strategy.calculate_signals(event)

                        elif event.type == 'SIGNAL':
                            self.signals += 1
                            self.portfolio.update_signal(event)

                        elif event.type == 'ORDER':
                            self.orders += 1
                            self.execution_handler.execute_order(event)
                            self.execution_handler.update_order(event)

                        elif event.type == 'FILL':
                            self.fills += 1
                            self.portfolio.update_fill(event)

            time.sleep(self.heartbeat)

    def _output_performance(self):
        """
        Outputs the strategy performance from the backtest.
        """
        self.portfolio.create_equity_curve_dataframe()

        print "Creating summary stats..."
        stats = self.portfolio.output_summary_stats()

        print "Creating equity curve..."
        print self.portfolio.equity_curve.iloc[-1, :]
        pprint.pprint(stats)

        print "Signals: %s" % self.signals
        print "Orders: %s" % self.orders
        print "Fills: %s" % self.fills

        self.portfolio.plot_returns()

    def simulate_trading(self):
        """
        Simulates the backtest and outputs portfolio performance.
        """
        self._run_backtest()
        self._output_performance()








################
# coding: UTF-8

try:
    import Queue as queue
except ImportError:
    import queue
import time
import pprint


class Backtest(object):
    """
    Encapsulates the settings and components for carrying out
    an event-driven backtest.

    Parameters:

    symbol_list : The list of symbol strings.
    intial_capital : The starting capital for the portfolio.
    heartbeat : Backtest "heartbeat" in seconds
    start_date : The start datetime of the strategy.
    backtest_date : The start datetime of back-test.
    data_handler : (Class) Handles the market data feed.
    execution_handler : (Class) Handles the orders/fills for trades.
    portfolio : (Class) Keeps track of portfolio current and prior positions.
    strategy : (Class) Generates signals based on market data.
    """
    def __init__(self, symbol_list, initial_capital, heartbeat, start_date,
                 backtest_date, data_handler, execution_handler, portfolio,
                 strategy, threshold, per_return, ndays, idays,
                 X, y, backtest_X, backtest_y_info, models, ensemble):

        self.symbol_list = symbol_list
        self.initial_capital = initial_capital
        self.heartbeat = heartbeat
        self.start_date = start_date
        self.backtest_date = backtest_date

        self.data_handler_cls = data_handler
        self.execution_handler_cls = execution_handler
        self.portfolio_cls = portfolio
        self.strategy_cls = strategy
        self.threshold = threshold
        self.per_return = per_return

        self.events = queue.Queue()

        self.signals = 0
        self.orders = 0
        self.fills = 0

        self._generate_trading_instances(ndays, idays, X, y, backtest_X, backtest_y_info, models, ensemble)

    def _generate_trading_instances(self, ndays, idays, X, y, backtest_X, backtest_y_info, models, ensemble):
        """
        Generates the trading instance objects from their class types.
        """
        print "Creating DataHandler, Strategy, Portfolio and ExecutionHandler..."
        self.data_handler = self.data_handler_cls(self.events, self.start_date,
                                                  self.backtest_date, self.symbol_list, X, y, backtest_X, backtest_y_info)
        self.strategy = self.strategy_cls(self.data_handler, self.events, ndays, idays, models, ensemble,
                                          self.threshold, self.per_return)
        self.portfolio = self.portfolio_cls(self.data_handler, self.events, self.start_date,
                                            self.backtest_date, self.initial_capital)
        self.execution_handler = self.execution_handler_cls(self.events)
        self.data_handler.get_portfolio(self.portfolio)

    def _run_backtest(self):
        """
        Executes the backtest.
        """
        i = 0
        while True:
            i += 1
            # Update the market bars
            if self.data_handler.continue_backtest == True:
                self.data_handler.update_bars(i)
            else:
                break

            # Handle the events
            while True:
                try:
                    event = self.events.get(False)
                except queue.Empty:
                    self.portfolio.update_timeindex(i)
                    break
                else:
                    if event is not None:
                        if event.type == 'MARKET':
                            self.strategy.calculate_signals(event)

                        elif event.type == 'SIGNAL':
                            self.signals += 1
                            self.portfolio.update_signal(event)

                        elif event.type == 'ORDER':
                            self.orders += 1
                            self.execution_handler.execute_order(event)
                            self.execution_handler.update_order(event)

                        elif event.type == 'FILL':
                            self.fills += 1
                            self.portfolio.update_fill(event)

            time.sleep(self.heartbeat)

    def _output_performance(self):
        """
        Outputs the strategy performance from the backtest.
        """
        self.portfolio.create_equity_curve_dataframe()

        print "Creating summary stats..."
        stats = self.portfolio.output_summary_stats()

        print "Creating equity curve..."
        print self.portfolio.equity_curve.iloc[-1, :]
        pprint.pprint(stats)

        print "Signals: %s" % self.signals
        print "Orders: %s" % self.orders
        print "Fills: %s" % self.fills

        self.portfolio.plot_returns()

    def simulate_trading(self):
        """
        Simulates the backtest and outputs portfolio performance.
        """
        self._run_backtest()
        self._output_performance()
