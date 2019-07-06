# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 13:32:46 2018

@author: Erich Wittenbeck
"""


class CandleStick:

    def __init__(self, product_id, time, low, high, at_open, at_close, volume):
        self.product_id = product_id
        self.time = int(time)
        self.low = float(low)
        self.high = float(high)
        self.at_open = float(at_open)
        self.at_close = float(at_close)
        self.volume = float(volume)

    @classmethod
    def empty(cls, product_id, time):
        return cls(product_id, time, 0, 0, 0, 0, 0)

    def as_list(self):
        return [self.product_id,
                self.time,
                self.low,
                self.high,
                self.at_open,
                self.at_close,
                self.volume]
