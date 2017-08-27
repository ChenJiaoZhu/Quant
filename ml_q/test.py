# Coding: UTF-8

class Test():
    def __init__(self):
        self.mean = 0

    def me(self):
        self.mean = 1
        return

    def check(self):
        print self.mean

t = Test()
t.me()
t.check()

from Quant.ml_q import get_data as gd
gd.Normalization.fit()