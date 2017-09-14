# Coding: UTF-8

import tushare as ts


hs = ts.get_k_data('000300',index=True)
hs = hs[hs['date']>='2016-08-01']
hs = hs[hs['date']<'2017-08-01']
