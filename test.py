#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 17:28:36 2022

@author: potruso
"""

import requests

import pandas as pd

from methods import generate_samples_api
from methods import create_unique_api_dataframe

printing = True
show_samples = True

# PARAMETERS for SAMPLES generation from time series
lf = 1 # look forward for establishing targets
order = 3 # order of the relative max
tau = 12 # decorrelation time, on time series of 4h it's 48h

K = 2 # consecutive highs
period = 14 # period of indicators
slow_period = 300
fast_period = 60

periods = '14400' # it means 4h ohlc data
list_apis = [
    'https://api.cryptowat.ch/markets/bitfinex/ftmusd/ohlc',
    'http://api.cryptowat.ch/markets/Kraken/adausd/ohlc',
    'https://api.cryptowat.ch/markets/bitfinex/ethusd/ohlc',
    'https://api.cryptowat.ch/markets/bitfinex/xrpusd/ohlc',
    'http://api.cryptowat.ch/markets/Kraken/dogeusd/ohlc',
    'http://api.cryptowat.ch/markets/Kraken/omgusd/ohlc',
    'http://api.cryptowat.ch/markets/binance/bnbusdt/ohlc',
    'http://api.cryptowat.ch/markets/binance/maticusdt/ohlc',
    'http://api.cryptowat.ch/markets/binance/dotusdt/ohlc',
    'http://api.cryptowat.ch/markets/binance/ltcusdt/ohlc',
    'http://api.cryptowat.ch/markets/binance/shibusdt/ohlc'
    ]
crypto_symbols = ['ftmusd','adausd','ethusd','xrpusd','dogeusd','omgusd','bnbusd','maticusd','dotusd','ltcusd','shibusd']

for i_api in list_apis:
    resp = requests.get(
        i_api,
        data = {'CloseTime':1659304800}, # setted to 1st August 2022
        params = {'periods':periods})
        
    print(i_api, ', avaiable:',resp.ok)

generate_samples_api(crypto_symbols, list_apis, periods, lf, order, tau, K, period, slow_period, fast_period, printing, show_samples)
df_hh_api = create_unique_api_dataframe()

print(df_hh_api)

from preprocessing import report_targets_distribution
from preprocessing import remove_decimals_valley_p_var
from preprocessing import apply_scaling_oscillators
from preprocessing import logscale_volumes
from preprocessing import create_Xy

#----------------PREPROCESSING------------------------------

report_targets_distribution(df_hh_api)
remove_decimals_valley_p_var(df_hh_api)
df_hh_ing = apply_scaling_oscillators(df_hh_api)

logscale_volumes(df_hh_ing)

# PARAMETERS for TARGET CLASSES
min_pattern_length = 0
cc0 = -5
cc1 = -15
rnd_labels_study = False

# Design Matrix with Labels
Xy = create_Xy(df_hh_ing, min_pattern_length, cc0, cc1, rnd_labels_study)


import seaborn as sns

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from learning import get_X_y

from joblib import load
rf_tuned = load('rf_tuned.joblib')

#------------------TESTING NEW DATA (not streaming, just newer since July2022)

X_Test, y_Test = get_X_y(Xy)

y_Pred = rf_tuned.predict(X_Test)

print(classification_report(y_Test, y_Pred, digits=3, output_dict=False, zero_division='warn'))

cm_df = pd.DataFrame(
    confusion_matrix(y_Test, y_Pred),
    index = ['actual 0', 'actual 1', 'actual 2'],
    columns = ['pred 0', 'pred 1','pred 2'])
    
sns.heatmap(cm_df, annot=True, linewidth=2., fmt='g')
