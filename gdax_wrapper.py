# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 15:09:58 2018

@author: User
"""

import gdax
from time import sleep
from datetime import timedelta
from data_structs import CandleStick

# List of Products, available as of Feb. 2018
BTC_USD = "BTC-USD"
BCH_BTC = "BCH-BTC"
BCH_USD = "BCH-USD"
BCH_EUR = "BCH-EUR"
BTC_EUR = "BTC-EUR"
BTC_GBP = "BTC-GBP"
BTC_USD = "BTC-USD"
ETH_BTC = "ETH-BTC"
ETH_EUR = "ETH-EUR"
ETH_USD = "ETH-USD"
LTC_BTC = "LTC-BTC"
LTC_EUR = "LTC-EUR"
LTC_USD = "LTC-USD"


def get_historic_rates(product_id, start_date, end_date, granularity):
    candlesticks = []

    client = gdax.PublicClient()
    historic_rates = client.get_historic_rates(product_id, start_date, end_date, granularity)

    if len(historic_rates) == 0:
        raise ValueError("Invalid Time Range, probably!")
    if isinstance(historic_rates, dict):
        raise ValueError(historic_rates['message'])

    for rate in historic_rates:
        candlestick = CandleStick(product_id, *rate)
        candlesticks.append(candlestick)

    candlesticks.reverse()
    return candlesticks


def get_history_in_minutes(product_id, start_time, end_time):
    candlesticks = []
    time_delta = timedelta(hours=5)
    from_time = start_time
    to_time = start_time + time_delta

    requests = 0
    waits = 0

    print('Start fetching historic rates...')

    while from_time < end_time:
        if to_time > end_time:
            to_time = end_time
        candlesticks.extend(get_historic_rates(product_id, from_time.isoformat(), to_time.isoformat(), 60))
        from_time = to_time
        to_time += time_delta
        requests += 1
        if requests % 3 == 0:
            waits += 1
            sleep(1.1)
            if waits % 60 == 0:
                print('one minute passed')

    return candlesticks
