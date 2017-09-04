# coding: UTF-8

'''
This contains some methods for feature selection
'''

def SelectFromLasso(X, X_train, X_test, y_train, y_test):

    from sklearn.linear_model import Lasso
    import numpy as np

    lasso = Lasso(alpha=0.012)
    lasso.fit(X_train, y_train)
    print "R^2 on training set: %f" % lasso.score(X_train, y_train)
    print "R^2 on test set: %f" % lasso.score(X_test, y_test)
    print lasso.coef_

def SelectFromRFR(X, X_train, X_test, y_train, y_test):

    from sklearn.ensemble import RandomForestRegressor
    import numpy as np

    rfr = RandomForestRegressor(n_estimators=200, n_jobs=2)
    rfr.fit(X_train, y_train)
    print "R^2 on training set: %f" % rfr.score(X_train, y_train)
    print "R^2 on test set: %f" % rfr.score(X_test, y_test)

    importances = rfr.feature_importances_
    indices = np.where(importances>=0.01)[0]
    X = X.iloc[:, list(indices)]
    return X

def SelectFromETR(X, X_train, X_test, y_train, y_test):

    from sklearn.ensemble import ExtraTreesRegressor
    import numpy as np

    etr = ExtraTreesRegressor(n_estimators=200, n_jobs=2)
    etr.fit(X_train, y_train)
    print "R^2 on training set: %f" % etr.score(X_train, y_train)
    print "R^2 on test set: %f" % etr.score(X_test, y_test)

    importances = etr.feature_importances_
    indices = np.where(importances>=0.009)[0]
    X = X.iloc[:, list(indices)]
    return X

def SelectFromPCA(X, y):

    from sklearn.decomposition import PCA

    pca = PCA(n_components=15)
    pca.fit(X)
    X_ = pca.transform(X)
    return X_,y

def Norm(X, y):

    from sklearn.preprocessing import StandardScaler

    model = StandardScaler()
    model.fit(X)
    X_ = model.transform(X)
    return X_,y

def ForwardStepwise(X_train, X_test, y_train, y_test):

    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.svm import SVR, LinearSVR
    import pandas as pd
    import time

    score, score_ = -999, 0
    selected = []
    clm = list(X_train.columns)
    l = 0
    for k in X_train.columns:
        if len(k)>l:
            l=len(k)
    change_of_score = pd.DataFrame(columns=['Feature', 'R^2', 'Increment', 'Time'])
    begin = time.time()
    while True:
        for n,i in enumerate(clm):
            m = '['+i+']' + ' '*(l-len(i))
            if len(selected) == 0:
                # model = SVR(C=10000.0, gamma=1e-05, kernel='rbf')
                model = LinearRegression()
                # model = Ridge(alpha = 0.09)
                # model = LinearSVR(C=17.0)
                model.fit(X_train[[i]], y_train)
                scr = model.score(X_test[[i]], y_test)
                if scr > score:
                    score = scr
                    ind = n
                end = time.time()
                print '%s finished %s out of %s. Max score: %.5f %s. Time: %.2fs' %\
                      (m, n+1, len(clm), score, str(selected)+'+'+clm[ind], end-begin)
            else:
                # model = SVR(C=10000.0, gamma=1e-05, kernel='rbf')
                model = LinearRegression()
                # model = Ridge(alpha = 0.09)
                # model = LinearSVR(C=17.0)
                x = pd.concat([X_train[selected], X_train[[i]]], axis=1)
                model.fit(x, y_train)
                scr = model.score(pd.concat([X_test[selected], X_test[[i]]], axis=1), y_test)
                if scr > score:
                    score = scr
                    ind = n
                end = time.time()
                if ind >= len(clm) or score_ == score:
                    print '%s finished %s out of %s. Max score: %.5f %s. Time: %.2fs' %\
                          (m, n+1, len(clm), score, str(selected), end-begin)
                else:
                    print '%s finished %s out of %s. Max score: %.5f %s. Time: %.2fs' %\
                          (m, n+1, len(clm), score, str(selected)+'+'+clm[ind], end-begin)
        if score > score_ + 0 or len(selected)<15:
            end = time.time()
            record = pd.DataFrame([[clm[ind], score, score - score_, end-begin]],
                                  columns=['Feature', 'R^2', 'Increment', 'Time'])
            change_of_score = change_of_score.append(record, ignore_index=True)
            change_of_score['Time_diff'] = change_of_score['Time'].diff(1)
            change_of_score.iloc[0, -1] = change_of_score['Time'][0]
            selected.append(clm[ind])
            clm.pop(ind)
            end = time.time()
            print 'Iteration %s: the increment of r^2 is %s. Time: %.2fmin' %\
                  (len(selected), score - score_, (end-begin)/60)
            score_ = score
        else:
            change_of_score.index = change_of_score.index + 1
            end = time.time()
            print 'Mission complete and the max r^2 is %.5f with %s features. Time: %.2fmin' %\
                  (score, len(selected), (end-begin)/60)
            print 'Detail about feature selection process is shown below:\n%s' % change_of_score
            break


def get_best_subset(X):

    ridge78 = ['Close', '95_ROC', '65_FI', '5_PL', '5_SMA', '5_VAR_RTN', 'High', 'Lag2', 'Low',
               'Open', '35_MEAN_RTN', '50_CCI', '50_FI', '5_D', 'PHC', '20_ROC', '20_OD',
               '20_MEAN_RTN', 'Lag4', '5_FI', '20_FI', 'Lag10', '95_EVM', '35_PL', '35_L_BB_Close',
               '50_PL', '50_L_BB_Close', '35_CCI', '80_PH', '50_EWMA', '5_CCI', '20_J', 'Lag20',
               'Lag15', 'Lag14', 'Lag11', 'Volume', '20_VH', '80_VC', '50_VC', '80_VAR_RTN',
               'Lag9', 'Lag6', '95_VH', '50_PH', '5_VH', 'Lag5', 'Lag13', 'Lag3', '65_SKEW_RTN',
               '95_U_BB_Close', '20_SKEW_RTN', '35_ROC', '65_KURT_RTN', '80_U_BB_Close', '95_PH',
               'MACD', 'PLC', '65_VL', '80_VH', '50_EVM', '35_EVM', '5_L_BB_Close', '5_U_BB_Close',
               '65_VC', '95_VD', '80_VD', 'Lag18', 'Lag8', '5_VD', '65_U_BB_Close', '65_PH',
               '65_J', 'Lag19', '5_KURT_RTN', '20_VL', '35_VC', '20_VC']
    Ridge_78 = X[ridge78]

    lasso65 = ['Close', 'Open', '5_MACDH', '20_EVM', '80_EVM', '5_FI', 'Lag1', 'POC', '35_PL',
               '35_MEAN_RTN', '65_VL', '50_ROC', '20_CCI', '20_OD', '20_VAR_RTN', '35_VAR_RTN',
               '95_VL', '80_VH', 'Lag9', 'Lag10', '5_K', 'Lag20', '5_VAR_RTN', '35_VL', 'Lag5',
               '80_FI', '95_D', '35_CCI', '35_FI', '80_RSV', '35_VH', '80_VL', 'Lag6', '50_VH',
               '95_FI', '5_D', '20_D', '5_VL', 'Lag11', '35_SKEW_RTN', '50_CCI', 'Lag15',
               '35_U_BB_Close', '95_MEAN_RTN', 'Lag12', '20_SKEW_RTN', '35_D', '20_KURT_RTN',
               'Lag19', '5_CCI', '95_VD', '65_MEAN_RTN', 'Lag14', '80_SKEW_RTN', '65_RSV', '50_J',
               'Lag13', '5_VD', 'Lag7', 'Lag8', '5_KURT_RTN', 'Lag16', '5_SKEW_RTN', '5_PL', '80_VC']
    Lasso_65 = X[lasso65]

    rfr78 = ['Close', 'High', 'Low', 'Open', '5_EWMA', '5_SMA', '5_PC', '5_PH', '5_U_BB_Close',
             '5_PL', '20_PC', '5_L_BB_Close', '20_SMA', 'Lag1', '80_VL', '20_PL', '50_J', '20_K',
             '5_VAR_RTN', '20_EVM', '20_VAR_RTN', '35_K', 'Lag20', 'Lag4', 'PLC', 'Lag12', '5_MACDH',
             'Lag10', '20_D', 'Lag7', 'Lag11', '20_L_BB_Close', '5_FI', 'Lag17', 'Lag9', '50_CCI',
             'Lag13', '5_ROC', 'Lag3', 'Lag14', 'Lag15', 'Lag8', '5_MEAN_RTN', 'Lag19', '20_SKEW_RTN',
             'PHC', '20_ROC', 'Lag18', '50_EVM', 'Lag5', '20_FI', 'Lag16', '5_EVM', 'Lag6',
             '5_KURT_RTN', '20_KURT_RTN', 'Lag2', '5_RSV', '5_SKEW_RTN', '20_MACDH', '95_FI', '35_FI',
             '95_KURT_RTN', '65_FI', '80_KURT_RTN', '35_VAR_RTN', '65_EVM', '5_VL', '35_KURT_RTN',
             '35_SKEW_RTN', '20_EWMA', '95_EMA_MACD', '95_SKEW_RTN', '20_MEAN_RTN', '50_KURT_RTN',
             '95_D', '65_KURT_RTN', '5_EMA_MACD']

    RFR_78 = X[rfr78]

    RFR_48 = X[rfr78[:48]]
    return Ridge_78, Lasso_65, RFR_78, RFR_48
