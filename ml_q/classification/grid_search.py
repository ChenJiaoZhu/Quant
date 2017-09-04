# Coding: UTF-8
# !/usr/bin/python
# param_search.py

from __future__ import print_function

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

def svc_grid_search(X,y):
    # Train/test split
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.2, random_state=42)
    l = int(X.shape[0] * 0.2)
    X_train, X_test = X[:-l,:], X[-l:,:]
    y_train, y_test = y[:-l], y[-l:]

    # Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                         'C': [1, 10, 100, 1000, 10000]}]

    # Perform the grid search on the tuned parameters
    model = GridSearchCV(SVC(), tuned_parameters, cv=10)
    model.fit(X_train, y_train)

    print("Optimised parameters found on training set:")
    print(model.best_estimator_, "\n")

    print("Grid scores calculated on training set:")
    for mean_test_score, params in zip(model.cv_results_['mean_test_score'],
                                          model.cv_results_['params']):
        print("%0.3f for %r" % (mean_test_score, params))

def rfc_grid_search(X,y):
    # Train/test split
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.2, random_state=42)
    l = int(X.shape[0] * 0.2)
    X_train, X_test = X[:-l,:], X[-l:,:]
    y_train, y_test = y[:-l], y[-l:]

    # Set the parameters by cross-validation
    tuned_parameters = [{'n_estimators': [200, 500, 1000],
                         'max_features': ['auto', 'log2'],
                         'min_samples_leaf': [1, 10, 50]}]

    # Perform the grid search on the tuned parameters
    model = GridSearchCV(RandomForestClassifier(n_jobs=2), tuned_parameters, cv=10)
    model.fit(X_train, y_train)

    print("Optimised parameters found on training set:")
    print(model.best_estimator_, "\n")

    print("Grid scores calculated on training set:")
    for mean_test_score, params in zip(model.cv_results_['mean_test_score'],
                                          model.cv_results_['params']):
        print("%0.3f for %r" % (mean_test_score, params))