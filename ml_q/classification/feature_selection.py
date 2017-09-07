# Coding: UTF-8

'''
This contains some methods for feature selection
'''
from sklearn.feature_selection import SelectFromModel

def SelectFromRF(X, y):

    from sklearn.ensemble import RandomForestClassifier

    rfc = RandomForestClassifier(n_estimators=500, n_jobs=2)
    model = SelectFromModel(rfc, threshold=0.05)
    model.fit(X, y)
    X_ = model.transform(X)
    return X_,y

def SelectFromETC(X, y):

    from sklearn.ensemble import ExtraTreesClassifier
    import numpy as np

    model = ExtraTreesClassifier(n_estimators=200, n_jobs=2)
    model.fit(X, y)
    importances = model.feature_importances_
    indices = np.where(importances>=0.009)[0]
    X_ = X.iloc[:, list(indices)]
    return X_,y

def SelectFromPCA(X, y):

    from sklearn.decomposition import PCA

    pca = PCA(n_components=15)
    pca.fit(X)
    X_ = pca.transform(X)
    return X_,y

from Quant.ml_q import data
datasets, X, y = data.Get_Data(type_y ='reg')
X_train, X_test, y_train, y_test = data.split_data(X, y)