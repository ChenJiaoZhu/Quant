# Coding: UTF-8
# forecast the difference between close price of next day and today

import datetime
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import LinearSVR, SVR

def forecast(X_train, X_test, y_train, y_test):

    # Create the (parametrised) models
    models = [("LR", LinearRegression()),
              ("Ridge", Ridge(alpha=0.09)),
              ("Lasso", Lasso(alpha=0.0002)),
              ("LSVR", LinearSVR(C=17.0)),
              ("SVR", SVR(C=10000.0, gamma=0.00001, kernel='rbf')),
              ("ETR", ExtraTreesRegressor(n_estimators=1000, n_jobs=6,
                                          max_features=0.5, max_depth=15,
                                          min_samples_split=5, min_samples_leaf=3)),
              ("RFR", RandomForestRegressor(n_estimators=1000, n_jobs=6,
                                            max_features=0.5, max_depth=20,
                                            min_samples_split=5, min_samples_leaf=3))]

    # Iterate through the models
    model = {}
    for m in models:

        m[1].fit(X_train, y_train)
        model[m[0]] = m[1]
        print "%s:\nR^2 on training set: %f" % (m[0], m[1].score(X_train, y_train))
        print "R^2 on test set: %f" % m[1].score(X_test, y_test)
        print "%s\n" % datetime.datetime.now()