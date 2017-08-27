# Coding: UTF-8
# forecast the up or down of stock price

from __future__ import print_function

import datetime
import numpy as np
import pandas as pd

from pandas_datareader import data as web
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import LinearSVC, SVC


def create_lagged_series(symbol, start_date, end_date, lags=5):
    """
    This creates a pandas DataFrame that stores the
    percentage returns of the adjusted closing value of
    a stock obtained from Yahoo Finance, along with a
    number of lagged returns from the prior trading days
    (lags defaults to 5 days). Trading volume, as well as
    the Direction from the previous day, are also included.
    """

    # Obtain stock information from Yahoo Finance
    ts = web.DataReader(
        symbol, "yahoo",
        start_date - datetime.timedelta(days=365),
        end_date)

    # Create the new lagged DataFrame
    tslag = pd.DataFrame(index=ts.index)
    tslag["Today"] = ts["Adj Close"]
    tslag["Volume"] = ts["Volume"]

    # Create the shifted lag series of prior trading period close values
    for i in range(0, lags):
        tslag["Lag%s" % str(i + 1)] = ts["Adj Close"].shift(i + 1)

    # Create the returns DataFrame
    tsret = pd.DataFrame(index=tslag.index)
    tsret["Volume"] = tslag["Volume"]
    tsret["Today"] = tslag["Today"].pct_change() * 100.0

    # If any of the values of percentage returns equal zero, set them to
    # a small number (stops issues with QDA model in scikit-learn)
    for i, x in enumerate(tsret["Today"]):
        if (abs(x) < 0.0001):
            tsret["Today"][i] = 0.0001

    # Create the lagged percentage returns columns
    for i in range(0, lags):
        tsret["Lag%s" % str(i + 1)] = \
            tslag["Lag%s" % str(i + 1)].pct_change() * 100.0

    # Create the "Direction" column (+1 or -1) indicating an up/down day
    tsret["Direction"] = np.sign(tsret["Today"])
    tsret = tsret[tsret.index >= start_date]

    return tsret


if __name__ == "__main__":
    # Create a lagged series of the S&P500 US stock market index
    snpret = create_lagged_series(
        "^GSPC", datetime.datetime(2001, 1, 10),
        datetime.datetime(2005, 12, 31), lags=5
    )

    # Use the prior two days of returns as predictor
    # values, with direction as the response
    X = snpret[["Lag1", "Lag2"]]
    y = snpret["Direction"]

    # The test data is split into two parts: Before and after 1st Jan 2005.
    start_test = datetime.datetime(2005, 1, 1)

    # Create training and test sets
    X_train = X[X.index < start_test]
    X_test = X[X.index >= start_test]
    y_train = y[y.index < start_test]
    y_test = y[y.index >= start_test]

    # Create the (parametrised) models
    print("Hit Rates/Confusion Matrices:\n")
    models = [("LR", LogisticRegression()),
              ("RC", RidgeClassifier()),
              ("LDA", LDA()),
              ("QDA", QDA()),
              ("LSVC", LinearSVC(dual=False)),
              ("SVC", SVC(C=10000.0, cache_size=200, class_weight=None,
                  coef0=0.0, degree=3, gamma=0.001, kernel='rbf',
                  max_iter=-1, probability=False, random_state=None,
                  shrinking=True, tol=0.001, verbose=False)),
              ("RFC", RandomForestClassifier(n_estimators=200,
                  criterion='gini',max_depth=None, min_samples_split=2,
                  min_samples_leaf=1, max_features='auto',
                  oob_score=False, n_jobs=2, random_state=None)),
              ("ETC", ExtraTreesClassifier(n_estimators=200,
                  min_samples_leaf=1,max_features='auto'))]

    # Iterate through the models
    for m in models:

        m[1].fit(X_train, y_train)
        pred = m[1].predict(X_test)
        cm = confusion_matrix(y_test, pred)
        pred_prpt = cm.sum(axis=0) / float(cm.sum())
        ture_prpt = cm.sum(axis=1) / float(cm.sum())
        basic_score = (pred_prpt*ture_prpt).sum()

        # Output the hit-rate and the confusion matrix for each model
        print("%s:\nPredict score: %0.3f" % (m[0], m[1].score(X_test, y_test)))
        print("Train score: %0.3f" % (m[1].score(X_train, y_train)))
        print("Basical score: %0.3f\n%s" % (basic_score, cm))
        print(classification_report(y_test, pred))
        print("%s\n" % datetime.datetime.utcnow())