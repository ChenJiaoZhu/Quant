# Coding: UTF-8
# forecast the difference between close price of next day and today

import datetime
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import LinearSVR, SVR

def forecast(X_train, X_test, y_train, y_test):

    # Create the (parametrised) models
    models = [("LR", LinearRegression()),
              ("Ridge", Ridge(alpha=34)),
              ("Lasso", Lasso(alpha=0.012)),
              ("LSVR", LinearSVR(C=1.0)),
              ("SVR", SVR(C=10000.0, cache_size=200, max_iter=-1,
                          coef0=0.0, degree=3, gamma=0.00001, kernel='rbf',
                          shrinking=True, tol=0.001, verbose=False)),
              ("ETR", ExtraTreesRegressor(n_estimators=1000, n_jobs=6,
                                          max_features=0.5, max_depth=15,
                                          min_samples_split=5, min_samples_leaf=3)),
              ("RFR", RandomForestRegressor(n_estimators=200,
                  criterion='mse',max_depth=None, min_samples_split=2,
                  min_samples_leaf=1, max_features='auto',
                  oob_score=False, n_jobs=2, random_state=None))]

    # Iterate through the models
    model = {}
    for m in models:

        m[1].fit(X_train, y_train)
        model[m[0]] = m[1]
        print "%s:\nR^2 on training set: %f" % (m[0], m[1].score(X_train, y_train))
        print "R^2 on test set: %f" % m[1].score(X_test, y_test)
        print "%s\n" % datetime.datetime.now()