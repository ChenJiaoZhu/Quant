# Coding: UTF-8
# !/usr/bin/python
# model_selection.py

import time
import threading
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
    for a in [0.0004, 0.0003, 0.0002, 0.003, 0.05]:
        start = time.time()
        # Set the parameters by cross-validation
        tuned_parameters = [{'alpha': [a]}]

        # Perform the grid search on the tuned parameters
        model = GridSearchCV(Lasso(tol=1e-5), tuned_parameters, cv=cv, n_jobs=2)
        model.fit(X, y)
        end = time.time()

        for mean_test_score, params in zip(model.cv_results_['mean_test_score'],
                                              model.cv_results_['params']):
            print "%0.8f for %r    [X.shape=%s, cv=%s]  %0.2f s" % \
                  (mean_test_score, params, str(X.shape), cv, (end-start))


def ridge_grid_search(X, y):

    print "R^2 scores calculated on training set:"
    cv = 10
    for a in [0.11, 0.12, 0.13, 0.14, 0.15, 0.16]:
        start = time.time()
        # Set the parameters by cross-validation
        tuned_parameters = [{'alpha': [a]}]

        # Perform the grid search on the tuned parameters
        model = GridSearchCV(Ridge(), tuned_parameters, cv=cv, n_jobs=1)
        model.fit(X, y)
        end = time.time()

        for mean_test_score, params in zip(model.cv_results_['mean_test_score'],
                                              model.cv_results_['params']):
            print "%0.8f for %r    [X.shape=%s, cv=%s]  %0.2f s" % \
                  (mean_test_score, params, str(X.shape), cv, (end-start))


def lsvr_grid_search(X, y):

    print "R^2 scores calculated on training set:"
    cv = 10
    for c in [9, 11, 12, 17, 25]:
        start = time.time()
        # Set the parameters by cross-validation
        tuned_parameters = [{'C': [c]}]

        # Perform the grid search on the tuned parameters
        model = GridSearchCV(LinearSVR(), tuned_parameters, cv=cv, n_jobs=2)
        model.fit(X, y)
        end = time.time()

        for mean_test_score, params in zip(model.cv_results_['mean_test_score'],
                                              model.cv_results_['params']):
            print "%0.8f for %r    [X.shape=%s, cv=%s]  %0.2f s" % \
                  (mean_test_score, params, str(X.shape), cv, (end-start))


def svr_grid_search(X, y):

    print "R^2 scores calculated on training set:"
    cv = 3
    for g in [1e-2, 1e-5]:
        for c in [1, 10, 100, 1000, 10000]:
            start = time.time()
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
    for n in [200, 500, 1000, 2000]:
        start = time.time()
        # Set the parameters by cross-validation
        tuned_parameters = [{'n_estimators': [n], 'max_features': ['sqrt']}]
        # tuned_parameters = [{'n_estimators': [200, 500, 1000],
        #                      'max_features': ['sqrt', 'log2'],
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
    for n in [200, 500, 1000, 2000]:
        start = time.time()
        # Set the parameters by cross-validation
        tuned_parameters = [{'n_estimators': [n], 'max_features': ['sqrt']}]
        # tuned_parameters = [{'n_estimators': [200, 500, 1000],
        #                      'max_features': ['sqrt', 'log2'],
        #                      'min_samples_leaf': [1, 10, 50]}]

        # Perform the grid search on the tuned parameters
        model = GridSearchCV(RandomForestRegressor(n_jobs=2), tuned_parameters, cv=cv)
        model.fit(X, y)
        end = time.time()

        for mean_test_score, params in zip(model.cv_results_['mean_test_score'],
                                              model.cv_results_['params']):
            print "%0.8f for %r    [X.shape=%s, cv=%s]  %0.2f min" % \
                  (mean_test_score, params, str(X.shape), cv, (end-start)/60)


def svr_search_main(X_train, X_test, y_train, y_test, g, c):

    start = time.time()
    params = {'kernel': 'rbf', 'gamma': g, 'C': c}
    model = SVR(C=c, gamma=g, kernel='rbf')
    model.fit(X_train, y_train)
    end = time.time()

    print "%0.8f for %r    [X_train.shape=%s]  %0.2f min" % \
          (model.score(X_test, y_test), params, str(X_train.shape), (end-start)/60)


def parallel(X_train, X_test, y_train, y_test):
    threads = []
    for c in [100, 1000]:
        for g in [0.1, 0.05, 0.01]:
            t = threading.Thread(target=svr_search_main, args=(X_train, X_test, y_train, y_test, g, c))
            threads.append(t)
    for thread in threads:
        thread.start()
    thread.join()


def svr_search(X_train, X_test, y_train, y_test):

    print "R^2 scores calculated on training set:"
    for c in [1, 10, 100]:
        for g in [1e-3, 1e-4, 1e-5]:
            start = time.time()
            params = {'kernel': 'rbf', 'gamma': g, 'C': c}
            model = SVR(C=c, gamma=g, kernel='rbf')
            model.fit(X_train, y_train)
            end = time.time()

            print "%0.8f for %r    [X_train.shape=%s]  %0.2f min" % \
                  (model.score(X_test, y_test), params, str(X_train.shape), (end-start)/60)


def etr_search(X_train, X_test, y_train, y_test):

    print "R^2 scores calculated on test set:"
    n_jobs = 6
    n = 1000
    cv = 0
    max_features = 0.5
    max_score = {}
    max_sc = 0
    for depth in [11, 12, 13, 14, 15, 16, 17, 18, 19]:
        for split in range(2, 20):
            for leaf in [1, 2, 3, 4, 5]:
                start = time.time()
                # tuned_parameters = [{'n_estimators': [200, 500, 1000],
                #                      'max_features': ['auto', 'log2'],
                #                      'min_samples_leaf': [1, 10, 50]}]
                params = {'n_estimators': n, 'max_features': max_features, 'max_depth': depth,
                          'min_samples_split': split, 'min_samples_leaf':leaf , 'n_jobs': n_jobs}
                model = ExtraTreesRegressor(n_estimators=n, n_jobs=n_jobs, max_features=max_features,
                                            max_depth=depth, min_samples_split=split, min_samples_leaf=leaf)
                model.fit(X_train, y_train)
                sc = model.score(X_test, y_test)
                max_score[sc] = params
                if sc > max_sc:
                    max_sc = sc
                end = time.time()

                print "%0.8f for %r    [X_train.shape=%s, cv=%s]  %0.2f min" % \
                (sc, params, str(X_train.shape), cv, (end-start)/60)
    print "The best model is:\n%0.8f for %r    [X_train.shape=%s, cv=%s]" % \
          (max_sc, max_score[max_sc], str(X_train.shape), cv)


def rfr_search(X_train, X_test, y_train, y_test):

    print "R^2 scores calculated on test set:"
    n_jobs = 6
    n = 1000
    cv = 0
    max_features = 0.5
    max_score = {}
    max_sc = 0
    for depth in range(16, 30, 3):
        for split in range(2, 20, 3):
            for leaf in [1, 2, 3, 4, 5]:
                start = time.time()
                # tuned_parameters = [{'n_estimators': [200, 500, 1000],
                #                      'max_features': ['auto', 'log2'],
                #                      'min_samples_leaf': [1, 10, 50]}]
                params = {'n_estimators': n, 'max_features': max_features, 'max_depth': depth,
                          'min_samples_split': split, 'min_samples_leaf':leaf , 'n_jobs': n_jobs}
                model = RandomForestRegressor(n_estimators=n, n_jobs=n_jobs, max_features=max_features,
                                              max_depth=depth, min_samples_split=split, min_samples_leaf=leaf)
                model.fit(X_train, y_train)
                sc = model.score(X_test, y_test)
                max_score[sc] = params
                if sc > max_sc:
                    max_sc = sc
                end = time.time()

                print "%0.8f for %r    [X_train.shape=%s, cv=%s]  %0.2f min" % \
                (sc, params, str(X_train.shape), cv, (end-start)/60)
    print "The best model is:\n%0.8f for %r    [X_train.shape=%s, cv=%s]" % \
          (max_sc, max_score[max_sc], str(X_train.shape), cv)