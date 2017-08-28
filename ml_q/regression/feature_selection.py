# Coding: UTF-8

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
    from sklearn.svm import SVR
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
                # model = LinearRegression()
                model = Ridge(alpha = 0.09)
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
                # model = LinearRegression()
                model = Ridge(alpha = 0.09)
                x = X_train[selected].join(X_train[i])
                model.fit(x, y_train)
                scr = model.score(X_test[selected].join(X_test[i]), y_test)
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


from Quant.ml_q import get_data
X, y, backtest_X, backtest_y_info = get_data.Get_Data(type_y = 'reg')
X_train, X_test, y_train, y_test = get_data.split_by_weigh(X, y)

ForwardStepwise(X_train, X_test, y_train, y_test)

X, y = SelectFromPCA(X, y)
X, y = Norm(X, y)
X_train, X_test, y_train, y_test = get_data.split_by_weigh(X, y)