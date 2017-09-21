# coding: UTF-8

import pandas as pd
from Quant.ml_q.regression.feature_selection import get_best_subset
from Quant.ml_q.regression.ensemble_model import training_model
from Quant.ml_q.regression.ensemble_model import get_model, predict_by_cv

from event import SignalEvent
from data import DataHandler, select_stock
from backtest import Backtest
from execution import ExecutionHandler
from portfolio import Portfolio


class Strategy(object):
    """
    Base class for all the strategy.
    """
    pass


class MLModelingStrategy(Strategy):
    """
    A strategy based on machine learning modeling to
    predict the stock close price of the next day.

    Parameters:

    bars : The DataHandler object that provides bar information.
    events : The Event Queue object.
    ndays : How many days look back.
    idays : How many days the price decreased.
    threshold : One trade return of each stock which decides
                the price to buy or sell.
    per_return : Expect return of each stock.
    """
    def __init__(self, bars, events, ndays, idays,
                 threshold=0.02, per_return=0.1):

        self.bars = bars
        self.symbol_list = self.bars.symbol_list
        self.events = events
        self.threshold = threshold
        self.per_return = per_return

        # Trains the prediction models.
        self.models, self.ensemble_model = self._train_model()

        # Sets to True if a symbol is in the market
        self.bought = self._calculate_initial_bought()

        self.decrease = self.bars.whether_decrease(ndays, idays)

    def _train_model(self):

        # Gets the best subset for every model
        X_train_Ridge78, X_train_Lasso65, X_train_RFR78, X_train_RFR48 \
            = get_best_subset(self.bars.X)

        # Creates the parametrised models of each
        # single model and ensemble model
        single, ensemble = get_model()

        # Trains each single model
        models = training_model(X_train_Ridge78, X_train_Lasso65,
                                X_train_RFR78, X_train_RFR48,
                                self.bars.X, self.bars.backtest_X,
                                self.bars.y,
                                self.bars.backtest_y_info["True_reg"])

        # Creates the predicted X-train data sets by k-fold prediction
        # of Each single model to train the ensemble model
        pred_train = []
        pred_train.append(predict_by_cv(X_train_Ridge78, self.bars.y,
                                        estimator=single[0]))
        pred_train.append(predict_by_cv(X_train_Ridge78, self.bars.y,
                                        estimator=single[1]))
        pred_train.append(predict_by_cv(X_train_Lasso65, self.bars.y,
                                        estimator=single[2]))
        pred_train.append(predict_by_cv(X_train_Ridge78, self.bars.y,
                                        estimator=single[3]))
        pred_train.append(predict_by_cv(X_train_RFR78, self.bars.y,
                                        estimator=single[5]))
        pred_train.append(predict_by_cv(X_train_RFR48, self.bars.y,
                                        estimator=single[6]))
        X_train_pred = pd.DataFrame(pred_train).T

        # Trains the ensemble model
        ensemble.fit(X_train_pred, self.bars.y)

        return models, ensemble

    def _calculate_initial_bought(self):
        """
        Adds keys to the bought dictionary for all symbols
        and sets them to 'OUT'.
        """
        bought = {}
        for s in self.symbol_list:
            bought[s] = 'OUT'

        return bought

    def calculate_signals(self, event, deviation=0.0, fee=0.003):
        """
        Generates a new set of signals based on machine
        learning models.
        """
        if event.type == 'MARKET':
            # Gets the best subset for every model
            X_test_Ridge78, X_test_Lasso65, X_test_RFR78, X_test_RFR48 \
                = get_best_subset(self.bars.bar_X)

            # Gets the prediction results of every model
            # and combines them into new X
            pred_list = []
            pred_list.append(self.models['LR'].predict(X_test_Ridge78))
            pred_list.append(self.models['Ridge'].predict(X_test_Ridge78))
            pred_list.append(self.models['Lasso'].predict(X_test_Lasso65))
            pred_list.append(self.models['LSVR'].predict(X_test_Ridge78))
            pred_list.append(self.models['ETR'].predict(X_test_RFR78))
            pred_list.append(self.models['RFR'].predict(X_test_RFR48))
            X_test_pred = pd.DataFrame(pred_list).T

            # Predicts the final results through fitting the
            # new X to parametrised ensemble model
            self.y_pred = self.ensemble_model.predict(X_test_pred)

            # Strategy based on predict results
            for i, pred in enumerate(self.y_pred):
                buy_threshold = ((pred * (1.0 + deviation)) / (1.0 + self.threshold)) / (1.0 + fee)
                sell_threshold = ((pred * (1.0 + deviation)) * (1.0 + self.threshold)) / (1.0 - fee - 0.001)
                bars = self.bars.get_latest_bar_values(i)
                symbol = bars['Code']
                dt = bars.name
                sig_dir = ""
                strength = 1.0
                strategy_id = 1

                if self.bought[symbol] == "LONG":
                    position = self.bars.get_current_position(symbol)
                    current_holding = self.bars.get_current_holding(symbol)
                    last_holding = self.bars.get_last_holding(symbol)
                    if position > 0:
                        mean_price = - (current_holding - last_holding) / position
                        sell_p = (mean_price * (1.0 + self.per_return)) / (1.0 - fee - 0.001)
                        sell_p2 = (mean_price * (1.0 - self.per_return)) / (1.0 - fee - 0.001)
                        if sell_p < bars['True_high']:
                            sig_dir = 'EXIT'
                            signal = SignalEvent(strategy_id, symbol, dt, sig_dir, strength,
                                                 max(sell_p, bars['True_open']))
                            self.events.put(signal)
                            self.bought[symbol] = 'OUT'
                        elif sell_threshold < bars['True_high']:
                            sig_dir = 'EXIT'
                            signal = SignalEvent(strategy_id, symbol, dt, sig_dir, strength,
                                                 max(sell_threshold, bars['True_open']))
                            self.events.put(signal)
                            self.bought[symbol] = 'OUT'
                        elif sell_p2 < bars['True_high'] and sell_p2 > bars['True_low']:
                            sig_dir = 'EXIT'
                            signal = SignalEvent(strategy_id, symbol, dt, sig_dir, strength, sell_p2)
                            self.events.put(signal)
                            self.bought[symbol] = 'OUT'

                if buy_threshold > bars['True_low'] and buy_threshold < bars['True_high']:
                    sig_dir = 'LONG'
                    signal = SignalEvent(strategy_id, symbol, dt, sig_dir, strength,
                                             min(buy_threshold, bars['True_open']))
                    if self.decrease[symbol][dt]:
                        self.events.put(signal)
                        self.bought[symbol] = 'LONG'


if __name__ == "__main__":

    codes = select_stock()
    initial_capital = 10000.0
    heartbeat = 0.0
    start_date = '2007-01-01'
    backtest_date = '2016-08-01'
    threshold = 0.03
    per_return = 0.09
    ndays, idays = 13, 4

    backtest = Backtest(codes,
                        initial_capital,
                        heartbeat,
                        start_date,
                        backtest_date,
                        DataHandler,
                        ExecutionHandler,
                        Portfolio,
                        MLModelingStrategy,
                        threshold,
                        per_return,
                        ndays,
                        idays)

    backtest.simulate_trading()
