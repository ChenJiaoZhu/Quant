# coding: UTF-8

import pandas as pd
import numpy as np
from Quant.ml_q import get_data
from Quant.ml_q.regression.feature_selection import get_best_subset
from Quant.ml_q.regression.forecast import training_model, predict_by_cv, model_test
from Quant.ml_q.regression.param_search import easy_search, svr_search, etr_search, rfr_search
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import LinearSVR, SVR

if __name__ == "__main__":

    X, y, backtest_X, backtest_y_info = get_data.Get_Data(type_y='reg')
    X_train, X_test, y_train, y_test = get_data.split_by_weigh(X, y)
    X_train_Ridge78, X_train_Lasso65, X_train_RFR78, X_train_RFR48 = get_best_subset(X_train)
    X_test_Ridge78, X_test_Lasso65, X_test_RFR78, X_test_RFR48 = get_best_subset(X_test)
    models, pred_list = training_model(X_train_Ridge78, X_train_Lasso65, X_train_RFR78, X_train_RFR48,
                                       X_test_Ridge78, X_test_Lasso65, X_test_RFR78, X_test_RFR48,
                                       X_train, X_test, y_train, y_test)
    X_pred = pd.DataFrame(pred_list).T

    models = [("LR", LinearRegression()),
              ("Ridge", Ridge(alpha=0.09)),
              ("Lasso", Lasso(alpha=0.0002)),
              ("LSVR", LinearSVR(C=17.0)),
              ("SVR", SVR(C=100.0, gamma=0.01, epsilon=0.1, kernel='rbf')),
              ("ETR", ExtraTreesRegressor(n_estimators=1000, n_jobs=2,
                                          max_features=0.5, max_depth=18,
                                          min_samples_split=6, min_samples_leaf=4)),
              ("RFR", RandomForestRegressor(n_estimators=1000, n_jobs=2,
                                            max_features=0.5, max_depth=16,
                                            min_samples_split=5, min_samples_leaf=2))]

    pred_train = []
    pred_train.append(predict_by_cv(X_train_Ridge78, y_train, estimator=models[0]))
    pred_train.append(predict_by_cv(X_train_Ridge78, y_train, estimator=models[1]))
    pred_train.append(predict_by_cv(X_train_Lasso65, y_train, estimator=models[2]))
    pred_train.append(predict_by_cv(X_train_Ridge78, y_train, estimator=models[3]))
    pred_train.append(predict_by_cv(X_train, y_train, estimator=models[4]))
    pred_train.append(predict_by_cv(X_train_RFR78, y_train, estimator=models[5]))
    pred_train.append(predict_by_cv(X_train_RFR48, y_train, estimator=models[6]))
    X_train_pred = pd.DataFrame(pred_train).T

    model = model_test(X_train_pred, X_pred, y_train, y_test)

    estimator = []
    for a in [10, 50, 100, 200, 400, 600, 800, 1000, 2000]:
        # estimator.append([{'alpha':a}, Ridge(alpha=a)])
        # estimator.append([{'alpha':a}, Lasso(alpha=a)])
        # estimator.append([{'C':a}, LinearSVR(C=a)])
        estimator.append([{'n_estimators': a, 'n_jobs':2, 'max_features':'auto'},
                          ExtraTreesRegressor(n_estimators=a, n_jobs=2, max_features='auto')])
    # estimator = [('None', LinearRegression())]
    for e in estimator:
        easy_search(X_train_pred, X_pred, y_train, y_test, e)

    svr_search(X_train_pred, X_pred, y_train, y_test)
    etr_search(X_train_pred, X_pred, y_train, y_test)
    rfr_search(X_train_pred, X_pred, y_train, y_test)

