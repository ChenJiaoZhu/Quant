# coding: UTF-8
# forecast the difference between close price of next day and today

import numpy as np
import pandas as pd
import datetime, time
from Quant.ml_q import get_data
from Quant.ml_q.regression.feature_selection import get_best_subset
from Quant.ml_q.regression.param_search import easy_search, svr_search, etr_search, rfr_search
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import LinearSVR, SVR
from sklearn.metrics import r2_score


def training_model(X_train_Ridge78, X_train_Lasso65, X_train_RFR78, X_train_RFR48,
                   X_train, X_test, y_train, y_test):

    models = {}
    t1 = time.time()
    lr = LinearRegression().fit(X_train_Ridge78, y_train)
    models['LR'] = lr
    t2 = time.time()
    print 'LR has fitted. Spent %0.2f s' % (t2 - t1)

    ridge = Ridge(alpha=0.09).fit(X_train_Ridge78, y_train)
    models['Ridge'] = ridge
    t3 = time.time()
    print 'Ridge has fitted. Spent %0.2f s' % (t3 - t2)

    lasso = Lasso(alpha=0.0002).fit(X_train_Lasso65, y_train)
    models['Lasso'] = lasso
    t4 = time.time()
    print 'Lasso has fitted. Spent %0.2f s' % (t4 - t3)

    lsvr = LinearSVR(C=17.0).fit(X_train_Ridge78, y_train)
    models['LSVR'] = lsvr
    t5 = time.time()
    print 'LSVR has fitted. Spent %0.2f s' % (t5 - t4)

    svr = SVR(C=100.0, gamma=0.01, epsilon=0.1, kernel='rbf', max_iter=1000).fit(X_train, y_train)
    models['SVR'] = svr
    t6 = time.time()
    print 'SVR has fitted. Spent %0.2f min' % ((t6 - t5) / 60)

    etr = ExtraTreesRegressor(n_estimators=1000, n_jobs=2, max_features=0.5,
                              max_depth=18, min_samples_split=6,
                              min_samples_leaf=4).fit(X_train_RFR78, y_train)
    models['ETR'] = etr
    t7 = time.time()
    print 'ETR has fitted. Spent %0.2f min' % ((t7 - t6) / 60)

    rfr = RandomForestRegressor(n_estimators=1000, n_jobs=2, max_features=0.5,
                                max_depth=16, min_samples_split=5,
                                min_samples_leaf=2).fit(X_train_RFR48, y_train)
    models['RFR'] = rfr
    t8 = time.time()
    print 'RFR has fitted. Spent %0.2f min' % ((t8 - t7) / 60)

    return models


def model_predict(X_test_Ridge78, X_test_Lasso65, X_test_RFR78, X_test_RFR48,
                   X_train, X_test, y_train, y_test, models):
    pred_list = []
    pred_list.append(models['LR'].predict(X_test_Ridge78))
    print 'Single model test score:\nLR: %s' % r2_score(y_test, pred_list[0])

    pred_list.append(models['Ridge'].predict(X_test_Ridge78))
    print 'Ridge: %s' % r2_score(y_test, pred_list[1])

    pred_list.append(models['Lasso'].predict(X_test_Lasso65))
    print 'Lasso: %s' % r2_score(y_test, pred_list[2])

    pred_list.append(models['LSVR'].predict(X_test_Ridge78))
    print 'LSVR: %s' % r2_score(y_test, pred_list[3])

    pred_list.append(models['SVR'].predict(X_test))
    print 'SVR: %s' % r2_score(y_test, pred_list[4])

    pred_list.append(models['ETR'].predict(X_test_RFR78))
    print 'ETR: %s' % r2_score(y_test, pred_list[5])

    pred_list.append(models['RFR'].predict(X_test_RFR48))
    print 'RFR: %s' % r2_score(y_test, pred_list[6])

    return pred_list


def predict_by_cv(X, y, estimator, cv=5):

    pred = np.array([])
    l = int(X.shape[0] / cv)
    for n in range(cv):
        if n == cv-1:
            X_test = X.iloc[(n * l):, :]
            y_test = y[(n * l):]
            X_train = X.iloc[:(n * l), :]
            y_train = y[:(n * l)]
        else:
            X_test = X.iloc[(n * l):(n + 1) * l, :]
            y_test = y[(n * l):(n + 1) * l]
            X_train = pd.concat([X.iloc[:(n * l), :], X.iloc[(n + 1) * l:, :]])
            y_train = np.append(y[:(n * l)], y[(n + 1) * l:])

        t1 = time.time()
        estimator[1].fit(X_train, y_train)
        prd = estimator[1].predict(X_test)
        pred = np.append(pred, prd)
        t2 = time.time()

        print '%s: prediction of No.%s piece is finished. %0.2f min' \
              % (estimator[0], n+1, (t2 - t1) / 60)

    return pred


def model_test(X_train, X_test, y_train, y_test):

    # Create the (parametrised) models
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

    # Iterate through the models
    model = {}
    for m in models:

        m[1].fit(X_train, y_train)
        model[m[0]] = m[1]
        print "%s:\nR^2 on training set: %0.8f" % (m[0], m[1].score(X_train, y_train))
        print "R^2 on test set: %0.8f" % m[1].score(X_test, y_test)
        print "%s\n" % datetime.datetime.now()
    return model


def model_selection():

    X, y, backtest_X, backtest_y_info = get_data.Get_Data(type_y='reg')
    X_train, X_test, y_train, y_test = get_data.split_by_weigh(X, y)
    X_train_Ridge78, X_train_Lasso65, X_train_RFR78, X_train_RFR48 = get_best_subset(X_train)
    X_test_Ridge78, X_test_Lasso65, X_test_RFR78, X_test_RFR48 = get_best_subset(X_test)

    single, ensemble = get_model()

    models = training_model(X_train_Ridge78, X_train_Lasso65, X_train_RFR78, X_train_RFR48,
                            X_train, X_test, y_train, y_test)
    pred_list = model_predict(X_test_Ridge78, X_test_Lasso65, X_test_RFR78, X_test_RFR48,
                              X_train, X_test, y_train, y_test, models)
    X_test_pred = pd.DataFrame(pred_list).T

    pred_train = []
    pred_train.append(predict_by_cv(X_train_Ridge78, y_train, estimator=single[0]))
    pred_train.append(predict_by_cv(X_train_Ridge78, y_train, estimator=single[1]))
    pred_train.append(predict_by_cv(X_train_Lasso65, y_train, estimator=single[2]))
    pred_train.append(predict_by_cv(X_train_Ridge78, y_train, estimator=single[3]))
    pred_train.append(predict_by_cv(X_train, y_train, estimator=single[4]))
    pred_train.append(predict_by_cv(X_train_RFR78, y_train, estimator=single[5]))
    pred_train.append(predict_by_cv(X_train_RFR48, y_train, estimator=single[6]))
    X_train_pred = pd.DataFrame(pred_train).T

    model = model_test(X_train_pred, X_test_pred, y_train, y_test)

    estimator = []
    for a in [10, 50, 100, 200, 400, 600, 800, 1000, 2000]:
        # estimator.append([{'alpha':a}, Ridge(alpha=a)])
        # estimator.append([{'alpha':a}, Lasso(alpha=a)])
        # estimator.append([{'C':a}, LinearSVR(C=a)])
        estimator.append([{'n_estimators': a, 'n_jobs':2, 'max_features':'auto'},
                          ExtraTreesRegressor(n_estimators=a, n_jobs=2, max_features='auto')])
    # estimator = [('None', LinearRegression())]
    for e in estimator:
        easy_search(X_train_pred, X_test_pred, y_train, y_test, e)

    svr_search(X_train_pred, X_test_pred, y_train, y_test)
    etr_search(X_train_pred, X_test_pred, y_train, y_test)
    rfr_search(X_train_pred, X_test_pred, y_train, y_test)


def get_model():

    single = [("LR", LinearRegression()),
              ("Ridge", Ridge(alpha=0.09)),
              ("Lasso", Lasso(alpha=0.0002)),
              ("LSVR", LinearSVR(C=17.0)),
              ("SVR", SVR(C=100.0, gamma=0.01, epsilon=0.1, kernel='rbf', max_iter=1000)),
              ("ETR", ExtraTreesRegressor(n_estimators=1000, n_jobs=2,
                                          max_features=0.5, max_depth=18,
                                          min_samples_split=6, min_samples_leaf=4)),
              ("RFR", RandomForestRegressor(n_estimators=1000, n_jobs=2,
                                            max_features=0.5, max_depth=16,
                                            min_samples_split=5, min_samples_leaf=2))]

    ensemble = SVR(C=3.0, gamma=0.0005, epsilon=0.1, kernel='rbf')

    return single, ensemble