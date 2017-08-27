# Coding: UTF-8

import MySQLdb
import time
import pandas as pd
import numpy as np
import tushare as ts
from sqlalchemy import create_engine

'''
This download the original stock data from tushare and 
transform this original data into our original datasets.
'''


def get_hs300s_code():
    df = ts.get_hs300s()
    engine = create_engine('mysql://root:123@127.0.0.1/hs300s?charset=utf8')
    df.to_sql('symbol', engine, if_exists='append')


def get_sql_data(sql, fetch=True, data=None, many=False, db='hs300s'):
    con = MySQLdb.connect(host='localhost',
                          user='root',
                          passwd='123',
                          db=db,
                          charset='utf8')
    with con:
        cur = con.cursor()
        if many:
            cur.executemany(sql, data)
        else:
            cur.execute(sql, data)
        if fetch:
            data_ = cur.fetchall()
            return data_


def download_original_data():
    sql = 'select code from symbol'
    data = get_sql_data(sql)
    code = sorted([i[0] for i in data])
    for k,i in enumerate(code):
        df = ts.get_k_data(i, start='2000-01-01', end='2017-08-01')
        engine = create_engine('mysql://root:123@127.0.0.1/hs300s?charset=utf8')
        df.to_sql('daily_price', engine, if_exists='append')
        print 'Added %s data into the DB. %s out of %s' % (i,k+1,len(code))


def select_stock(listing_date='2014-08-01', start='2013-08-01', end='2016-08-01'):
    sql = "select code from daily_price where `index`=1 and date<='%s'" % listing_date
    data = get_sql_data(sql)
    sorted_code = sorted([i[0] for i in data])
    price_code = []
    volume_df = pd.DataFrame()
    for n,i in enumerate(sorted_code):
        sql = "select date,close,volume from daily_price where code=%s " \
              "and date<='%s' and date>'%s'" % (i, end, start)
        data = get_sql_data(sql)
        df = pd.DataFrame(list(data), columns=['Date', 'Close', 'Volume']).set_index('Date')

        if df['Close'].mean() >=10 and df['Close'].mean() <=50:
            price_code.append(i)

        volume_df = pd.concat([volume_df, df['Volume'].rename(i)], axis=1)
        print 'Selecting stock... %s is finished. %s out of %s' % (i, n+1, len(sorted_code))

    volume_code = []
    vol_desc = volume_df.mean().describe(percentiles=[0.33, 0.66])
    for i in volume_df.columns:
        if volume_df[i].mean() >= vol_desc['33%'] and volume_df[i].mean() <= vol_desc['66%']:
            volume_code.append(i)

    codes = sorted(list(set(price_code) & set(volume_code)))
    return codes


def get_original_data(codes, start_time, start='2007-01-01'):
    original_data = []
    for k,i in enumerate(codes):
        sql = "select * from daily_price where code = %s and `date`>'%s' " % (i, start)
        data = get_sql_data(sql)
        df = pd.DataFrame(list(data), columns=['id', 'Date','Open','Close',
                          'High','Low','Volume','Code']).set_index('Date')
        # df.index = df.index.astype('datetime64')
        del df['id']
        original_data.append(df)
        end = time.time()
        print "Successfully get %s's data from DB: %s out of %s. Time: %.2fs" % \
              (i, k + 1, len(codes), end - start_time)
    return original_data


def features_extraction(data, ndays = range(5,100,15)):

    from Quant.ml_q.features.BB import BB
    from Quant.ml_q.features.CCI import CCI
    from Quant.ml_q.features.EVM import EVM
    from Quant.ml_q.features.FI import FI
    from Quant.ml_q.features import MA
    from Quant.ml_q.features.MACD import MACD
    from Quant.ml_q.features.MACD import EMA_MACD
    from Quant.ml_q.features import MMT
    from Quant.ml_q.features.PC import PC
    from Quant.ml_q.features.PVC import PVC
    from Quant.ml_q.features.ROC import ROC
    from Quant.ml_q.features.RSV import RSV
    from Quant.ml_q.features.VC import VC

    data = MACD(data)
    data = MMT.TL(data)
    data = PVC(data)
    for n in ndays:
        data = BB(data, n)
        data = CCI(data, n)
        data = EVM(data, n)
        data = FI(data, n)
        data = MA.SMA(data, n)
        data = MA.EWMA(data, n)
        data = EMA_MACD(data, n)
        data = MMT.MMT(data, n)
        data = PC(data, n)
        data = ROC(data, n)
        data = RSV(data, n)
        data = VC(data, n)
    return data


def add_target(data):
    # classification target
    data["True_cls"] = data["Close"].pct_change() * 100.0
    for i, x in enumerate(data["True_cls"]):
        if (abs(x) == 0.0):
            if i % 2 == 0:
                data.loc[data.index[i], "True_cls"] = 0.00001
            else:
                data.loc[data.index[i], "True_cls"] = -0.00001
    data["True_cls"] = np.sign(data["True_cls"])

    # regression target
    data["True_reg"] = data["Close"].copy()
    return data


def pre_processing(data):

    data.iloc[:,:-2] = data.iloc[:,:-2].shift(1)
    data.iloc[:, -2:] = data.iloc[:, -2:].shift(-1)
    data.dropna(axis=0, how='any', inplace=True)
    return data


class Normalization(object):

    def __init__(self):
        self.stdar_mean = 0
        self.stdar_std = 0
        self.norm_max = 0
        self.norm_min = 0

    def fit(self, data):

        self.stdar_mean, self.stdar_std = data.mean(), data.std()
        stdar_data = (data - self.stdar_mean) / self.stdar_std
        self.norm_max, self.norm_min = stdar_data.max(), stdar_data.min()
        return

    def transform(self, X, norm = True):

        stdar_data = (X - self.stdar_mean) / self.stdar_std
        if norm:
            stdar_data = stdar_data / (self.norm_max - self.norm_min)
        return stdar_data


def split_x_y(data):

    y = data[['True_cls','True_reg','Code']].copy()
    del data["True_cls"], data["True_reg"], data["Code"]
    return data, y


# if __name__ == "__main__":
    # get_hs300s_code()
    # download_original_data()

def Get_Data(type_y = 'cls', backtest_date = '2016-08-01'):
    # codes = select_stock()
    codes = [u'000039', u'000060', u'000061', u'000333', u'000503', u'000623',
             u'000625', u'000651', u'000728', u'000738', u'000768', u'000793',
             u'000800', u'000858', u'000917', u'002008', u'002065', u'002129',
             u'002202', u'002230', u'002415', u'002465', u'002673', u'300002',
             u'300027', u'300058', u'300104', u'300251', u'600037', u'600060',
             u'600074', u'600104', u'600118', u'600150', u'600196', u'600362',
             u'600406', u'600518', u'600547', u'600585', u'600588', u'600637',
             u'600649', u'600703', u'600718', u'600739', u'600804', u'600827',
             u'600875', u'600895', u'601088', u'601601', u'601607', u'601628',
             u'601788', u'601928']

    start_time = time.time()
    original_data = get_original_data(codes, start_time = start_time)
    datasets = []
    for k, data in enumerate(original_data):
        data = features_extraction(data)
        data = add_target(data)
        data = pre_processing(data)
        datasets.append(data)
        end = time.time()
        print "Pre-processing of %s's data is finished: %s out of %s. Time: %.2fs" % \
              (codes[k], k+1, len(original_data), end - start_time)

    datasets = pd.concat(datasets)
    train_data = datasets[datasets.index <= backtest_date].copy()
    backtest_data = datasets[datasets.index > backtest_date].copy()
    end = time.time()
    print 'Training data and backtest data were split. Time: %.2fs' % (end - start_time)

    train_X, train_y_info = split_x_y(train_data.copy())
    backtest_X, backtest_y_info = split_x_y(backtest_data.copy())
    end = time.time()
    print 'X and y_info were split. Time: %.2fs' % (end - start_time)

    norm = Normalization()
    norm.fit(train_X)
    train_X = norm.transform(train_X)
    backtest_X = norm.transform(backtest_X)
    end = time.time()
    print 'X were normalized. Time: %.2fs' % (end - start_time)

    if type_y == 'cls':
        y = train_y_info["True_cls"].copy()
    else:
        y = train_y_info["True_reg"].copy()

    return train_X, y, backtest_X, backtest_y_info


def split_by_weigh(X, y, w = 0.3):

    l = int(X.shape[0] * w)
    try:
        if X._typ == 'dataframe':
            X_train, X_test = X.iloc[:-l,:].copy(), X.iloc[-l:,:].copy()
    except:
        X_train, X_test = X[:-l, :].copy(), X[-l:, :].copy()
    y_train, y_test = y[:-l].copy(), y[-l:].copy()
    return X_train, X_test, y_train, y_test