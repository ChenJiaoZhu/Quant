# Coding: UTF-8
# !/usr/bin/python
# model_selection.py

import time
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR, LinearSVR
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor


def lr_cross_validation(X, y):

    cv = 10
    lr = LinearRegression()
    cv_results = cross_validate(lr, X, y, cv=cv)
    print 'LR:\n%0.8f    [X.shape=%s, cv=%s]' % \
          (cv_results['test_score'].mean(), str(X.shape), cv)


def lasso_grid_search(X, y):

    print "R^2 scores calculated on training set:"
    cv = 10
    start = time.time()
    for a in [0.0004, 0.0003, 0.0002, 0.003, 0.05]:
        # Set the parameters by cross-validation
        tuned_parameters = [{'alpha': [a]}]

        # Perform the grid search on the tuned parameters
        model = GridSearchCV(Lasso(tol=1e-5), tuned_parameters, cv=cv, n_jobs=2)
        model.fit(X, y)
        end = time.time()

        for mean_test_score, params in zip(model.cv_results_['mean_test_score'],
                                              model.cv_results_['params']):
            print "%0.8f for %r    [X.shape=%s, cv=%s]  %0.2f min" % \
                  (mean_test_score, params, str(X.shape), cv, (end-start)/60)


def ridge_grid_search(X, y):

    print "R^2 scores calculated on training set:"
    cv = 10
    start = time.time()
    for a in [0.11, 0.12, 0.13, 0.14, 0.15, 0.16]:
        # Set the parameters by cross-validation
        tuned_parameters = [{'alpha': [a]}]

        # Perform the grid search on the tuned parameters
        model = GridSearchCV(Ridge(), tuned_parameters, cv=cv, n_jobs=1)
        model.fit(X, y)
        end = time.time()

        for mean_test_score, params in zip(model.cv_results_['mean_test_score'],
                                              model.cv_results_['params']):
            print "%0.8f for %r    [X.shape=%s, cv=%s]  %0.2f min" % \
                  (mean_test_score, params, str(X.shape), cv, (end-start)/60)


def lsvr_grid_search(X, y):

    print "R^2 scores calculated on training set:"
    cv = 10
    start = time.time()
    for c in [9, 11, 12, 17, 25]:
        # Set the parameters by cross-validation
        tuned_parameters = [{'C': [c]}]

        # Perform the grid search on the tuned parameters
        model = GridSearchCV(LinearSVR(), tuned_parameters, cv=cv, n_jobs=2)
        model.fit(X, y)
        end = time.time()

        for mean_test_score, params in zip(model.cv_results_['mean_test_score'],
                                              model.cv_results_['params']):
            print "%0.8f for %r    [X.shape=%s, cv=%s]  %0.2f min" % \
                  (mean_test_score, params, str(X.shape), cv, (end-start)/60)


def svr_grid_search(X, y):

    print "R^2 scores calculated on training set:"
    cv = 3
    start = time.time()
    for g in [1e-2, 1e-5]:
        for c in [1, 10, 100, 1000, 10000]:
            # Set the parameters by cross-validation
            tuned_parameters = [{'kernel': ['rbf'], 'gamma': [g], 'C': [c]}]

            # Perform the grid search on the tuned parameters
            model = GridSearchCV(SVR(), tuned_parameters, cv=cv, n_jobs=2)
            model.fit(X, y)
            end = time.time()

            for mean_test_score, params in zip(model.cv_results_['mean_test_score'],
                                                  model.cv_results_['params']):
                print "%0.8f for %r    [X.shape=%s, cv=%s]  %0.2f min" % \
                      (mean_test_score, params, str(X.shape), cv, (end-start)/60)


def etr_grid_search(X, y):

    print "R^2 scores calculated on training set:"
    cv = 3
    start = time.time()
    for n in [200, 500, 1000, 2000]:
        # Set the parameters by cross-validation
        tuned_parameters = [{'n_estimators': [n]}]
        # tuned_parameters = [{'n_estimators': [200, 500, 1000],
        #                      'max_features': ['auto', 'log2'],
        #                      'min_samples_leaf': [1, 10, 50]}]

        # Perform the grid search on the tuned parameters
        model = GridSearchCV(ExtraTreesRegressor(n_jobs=2), tuned_parameters, cv=cv)
        model.fit(X, y)
        end = time.time()

        for mean_test_score, params in zip(model.cv_results_['mean_test_score'],
                                              model.cv_results_['params']):
            print "%0.8f for %r    [X.shape=%s, cv=%s]  %0.2f min" % \
                  (mean_test_score, params, str(X.shape), cv, (end-start)/60)


def rfr_grid_search(X, y):

    print "R^2 scores calculated on training set:"
    cv = 3
    start = time.time()
    for n in [200, 500, 1000, 2000]:
        # Set the parameters by cross-validation
        tuned_parameters = [{'n_estimators': [n]}]
        # tuned_parameters = [{'n_estimators': [200, 500, 1000],
        #                      'max_features': ['auto', 'log2'],
        #                      'min_samples_leaf': [1, 10, 50]}]

        # Perform the grid search on the tuned parameters
        model = GridSearchCV(RandomForestRegressor(n_jobs=2), tuned_parameters, cv=cv)
        model.fit(X, y)
        end = time.time()

        for mean_test_score, params in zip(model.cv_results_['mean_test_score'],
                                              model.cv_results_['params']):
            print "%0.8f for %r    [X.shape=%s, cv=%s]  %0.2f min" % \
                  (mean_test_score, params, str(X.shape), cv, (end-start)/60)


def etr_search(X_train, X_test, y_train, y_test):

    print "R^2 scores calculated on test set:"
    start = time.time()
    n_jobs = 2
    cv = 0
    for n in [20, 200, 500, 1000, 2000]:
        # tuned_parameters = [{'n_estimators': [200, 500, 1000],
        #                      'max_features': ['auto', 'log2'],
        #                      'min_samples_leaf': [1, 10, 50]}]
        params = {'n_estimators': n, 'n_jobs': n_jobs}
        model = ExtraTreesRegressor(n_estimators=n, n_jobs=n_jobs)
        model.fit(X_train, y_train)
        end = time.time()

        print "%0.8f for %r    [X_train.shape=%s, cv=%s]  %0.2f min" % \
        (model.score(X_test, y_test), params, str(X_train.shape), cv, (end-start)/60)


def rfr_search(X_train, X_test, y_train, y_test):

    print "R^2 scores calculated on test set:"
    start = time.time()
    n_jobs = 2
    cv = 0
    for n in [20, 200, 500, 1000, 2000]:
        # tuned_parameters = [{'n_estimators': [200, 500, 1000],
        #                      'max_features': ['auto', 'log2'],
        #                      'min_samples_leaf': [1, 10, 50]}]
        params = {'n_estimators': n, 'n_jobs': n_jobs}
        model = RandomForestRegressor(n_estimators=n, n_jobs=n_jobs)
        model.fit(X_train, y_train)
        end = time.time()

        print "%0.8f for %r    [X_train.shape=%s, cv=%s]  %0.2f min" % \
        (model.score(X_test, y_test), params, str(X_train.shape), cv, (end-start)/60)