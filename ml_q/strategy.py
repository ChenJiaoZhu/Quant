# coding: UTF-8

import pandas as pd
from event import SignalEvent
from Quant.ml_q.regression.feature_selection import get_best_subset
from Quant.ml_q.regression.ensemble_model import training_model
from Quant.ml_q.regression.ensemble_model import get_model, predict_by_cv

from Quant.ml_q.data import DataHandler
from Quant.ml_q.backtest import Backtest
from Quant.ml_q.execution import ExecutionHandler
from Quant.ml_q.portfolio import Portfolio


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

    bars : The DataHandler object that provides bar information
    events : The Event Queue object.
    """
    def __init__(self, bars, events):

        self.bars = bars
        self.symbol_list = self.bars.symbol_list
        self.events = events

        # Trains the prediction models.
        self.models, self.ensemble_model = self._train_model()

        # Sets to True if a symbol is in the market
        self.bought = self._calculate_initial_bought()

    def _train_model(self):
        # Gets the best subset for every model
        X_train_Ridge78, X_train_Lasso65, X_train_RFR78, X_train_RFR48 = get_best_subset(self.bars.X)

        # Creates the parametrised models of each single model and ensemble model
        single, ensemble = get_model()

        # Trains each single model
        models = training_model(X_train_Ridge78, X_train_Lasso65, X_train_RFR78, X_train_RFR48,
                                self.bars.X, self.bars.backtest_X, self.bars.y, self.bars.backtest_y_info["True_reg"])

        # Creates the predicted X-train data sets by k-fold prediction of
        # Each single model to train the ensemble model
        pred_train = []
        pred_train.append(predict_by_cv(X_train_Ridge78, self.bars.y, estimator=single[0]))
        pred_train.append(predict_by_cv(X_train_Ridge78, self.bars.y, estimator=single[1]))
        pred_train.append(predict_by_cv(X_train_Lasso65, self.bars.y, estimator=single[2]))
        pred_train.append(predict_by_cv(X_train_Ridge78, self.bars.y, estimator=single[3]))
        # pred_train.append(predict_by_cv(X.iloc[X.index>'2010-08-01',:], y, estimator=single[4]))
        pred_train.append(predict_by_cv(X_train_RFR78, self.bars.y, estimator=single[5]))
        pred_train.append(predict_by_cv(X_train_RFR48, self.bars.y, estimator=single[6]))
        X_train_pred = pd.DataFrame(pred_train).T

        # Trains the ensemble model
        ensemble.fit(X_train_pred, self.bars.y)

        # Tests the model performance
        # svr_search(X_train_pred, X_test_pred, y, backtest_y_info["True_reg"])
        # pred_list = model_predict(X_test_Ridge78, X_test_Lasso65, X_test_RFR78, X_test_RFR48,
        #                           X, backtest_X, y, backtest_y_info["True_reg"], models)
        # X_test_pred = pd.DataFrame(pred_list).T
        # print 'Ensemble: %s' % ensemble.score(X_test_pred, backtest_y_info["True_reg"])

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

    def calculate_signals(self, event, threshold=0.01, deviation=0.0, fee=0.003):
        """
        Generates a new set of signals based on machine
        learning models.

        Parameters
        event - A MarketEvent object.
        """
        if event.type == 'MARKET':
            # Gets the best subset for every model
            X_test_Ridge78, X_test_Lasso65, X_test_RFR78, X_test_RFR48 = get_best_subset(self.bars.bar_X)

            # Gets the prediction results of every model and combines them into new X
            pred_list = []
            pred_list.append(self.models['LR'].predict(X_test_Ridge78))
            pred_list.append(self.models['Ridge'].predict(X_test_Ridge78))
            pred_list.append(self.models['Lasso'].predict(X_test_Lasso65))
            pred_list.append(self.models['LSVR'].predict(X_test_Ridge78))
            pred_list.append(self.models['ETR'].predict(X_test_RFR78))
            pred_list.append(self.models['RFR'].predict(X_test_RFR48))
            X_test_pred = pd.DataFrame(pred_list).T

            # Predicts the final results through fitting the new X to parametrised ensemble model
            y_pred = self.ensemble_model.predict(X_test_pred)

            for i, pred in enumerate(y_pred):
                buy_threshold = ((pred * (1.0 + deviation)) / (1.0 + threshold)) / (1.0 + fee)
                sell_threshold = ((pred * (1.0 + deviation)) * (1.0 + threshold)) / (1.0 - fee - 0.001)
                bars = self.bars.get_latest_bar_values(i)
                symbol = bars['Code']
                dt = bars.name
                sig_dir = ""
                strength = 1.0
                strategy_id = 1

                if buy_threshold > bars['True_low']:
                    sig_dir = 'LONG'
                    signal = SignalEvent(strategy_id, symbol, dt, sig_dir, strength, buy_threshold)
                    self.events.put(signal)
                    self.bought[symbol] = 'LONG'

                if sell_threshold < bars['True_high'] and self.bought[symbol] == "LONG":
                    sig_dir = 'SHORT'
                    signal = SignalEvent(strategy_id, symbol, dt, sig_dir, strength, sell_threshold)
                    self.events.put(signal)
                    self.bought[symbol] = 'OUT'


if __name__ == "__main__":
    codes = [u'000039', u'000060', u'000061', u'000333', u'000503', u'000623',
             u'000625', u'000651', u'000728', u'000738', u'000768', u'000793',
             u'000800', u'000858', u'000917', u'002008', u'002065', u'002129',
             u'002202', u'002230', u'002415', u'002465', u'002673', u'300002',
             u'300027', u'300058', u'300104', u'300251', u'600037', u'600060',
             u'600074', u'600104', u'600118', u'600150', u'600196', u'600362',
             u'600406', u'600518', u'600547', u'600585', u'600588', u'600637',
             u'600649', u'600703', u'600718', u'600739', u'600804', u'600827',
             u'600875', u'600895', u'601088', u'601601', u'601607', u'601628',
             u'601788', u'601928']
    initial_capital = 1000000.0
    heartbeat = 0.0
    start_date = '2007-01-01'

    backtest = Backtest(codes,
                        initial_capital,
                        heartbeat,
                        start_date,
                        DataHandler,
                        ExecutionHandler,
                        Portfolio,
                        MLModelingStrategy)

    backtest.simulate_trading()