# coding: UTF-8

import pandas as pd
from Quant.ml_q import get_data
from Quant.ml_q.regression.feature_selection import get_best_subset
from Quant.ml_q.regression.ensemble_model import get_model, predict_by_cv
from Quant.ml_q.regression.ensemble_model import training_model, model_predict


def train_model():
    # create the training data and test data
    X, y, backtest_X, backtest_y_info = get_data.Get_Data(type_y='reg')
    X_train_Ridge78, X_train_Lasso65, X_train_RFR78, X_train_RFR48 = get_best_subset(X)
    X_test_Ridge78, X_test_Lasso65, X_test_RFR78, X_test_RFR48 = get_best_subset(backtest_X)

    # create the parametrised models of each single model and ensemble model
    single, ensemble = get_model()

    # train each single model
    models = training_model(X_train_Ridge78, X_train_Lasso65, X_train_RFR78, X_train_RFR48,
                            X, backtest_X, y, backtest_y_info["True_reg"])

    # create the predicted X-train data sets by k-fold prediction of
    # each single model to train the ensemble model
    pred_train = []
    pred_train.append(predict_by_cv(X_train_Ridge78, y, estimator=single[0]))
    pred_train.append(predict_by_cv(X_train_Ridge78, y, estimator=single[1]))
    pred_train.append(predict_by_cv(X_train_Lasso65, y, estimator=single[2]))
    pred_train.append(predict_by_cv(X_train_Ridge78, y, estimator=single[3]))
    # pred_train.append(predict_by_cv(X.iloc[X.index>'2010-08-01',:], y, estimator=single[4]))
    pred_train.append(predict_by_cv(X_train_RFR78, y, estimator=single[5]))
    pred_train.append(predict_by_cv(X_train_RFR48, y, estimator=single[6]))
    X_train_pred = pd.DataFrame(pred_train).T

    # train the ensemble model
    ensemble.fit(X_train_pred, y)

    # test
    pred_list = model_predict(X_test_Ridge78, X_test_Lasso65, X_test_RFR78, X_test_RFR48,
                              X, backtest_X, y, backtest_y_info["True_reg"], models)
    X_test_pred = pd.DataFrame(pred_list).T
    ensemble.score(X_test_pred, backtest_y_info["True_reg"])

    return models, ensemble, backtest_X, backtest_y_info

if __name__ == "__main__":
    models, ensemble, backtest_X, backtest_y_info = train_model()