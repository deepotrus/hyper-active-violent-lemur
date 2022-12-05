#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 21:26:41 2022

@author: potruso
"""

import pandas as pd
import seaborn as sns

#---------------METHODS FROM methods.py ---------------------

from methods import generate_samples
from methods import create_unique_dataframe

#---------------GENERATING SAMPLES FROM TIMESERIES-----------

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

crypto_symbols = [
    'binance_dogeusd','binance_iotxusdt','binance_oneusd','binance_trxusd','binance_vetusd',
    'bitfinex_ltcusd','poloniex_shibusdt','binance_ftmusdt','binance_xlmusdt','coinbase_omgusd',
    'binance_xrpusd','binance_lunausdt','ftx_bnbusd','ftx_solusd','binance_adausd','bitstamp_ethusd',
    'newdata_binance_vthousdt','newdata_bitfinex_avaxusd','newdata_bitfinex_dotusd','newdata_bitfinex_egldusd',
    'newdata_bitfinex_filusd','newdata_bitfinex_icpusd','newdata_bitfinex_linkusd','newdata_bitfinex_maticusd',
    'newdata_bitfinex_thetausd','newdata_bitfinex_xmrusd','newdata_bitfinex_xtzusd',
    'newdata_ftx_algousd','newdata_ftx_atomusd','newdata_ftx_bchusd','newdata_ftx_crousd','newdata_ftx_hntusdt',
    'newdata_ftx_manausd','newdata_ftx_uniusd']

#crypto_symbols = ['binance_adausd']

generate_samples(crypto_symbols, lf, order, tau, K, period, slow_period, fast_period, printing, show_samples)
df_hh = create_unique_dataframe()

print('\n', df_hh)




#---------------METHODS FROM preprocessing.py

from preprocessing import report_targets_distribution
from preprocessing import remove_decimals_valley_p_var
from preprocessing import apply_scaling_oscillators
from preprocessing import logscale_volumes
from preprocessing import create_Xy
from preprocessing import distribution_samples_removed
from preprocessing import distribution_samples_length
from preprocessing import optimize_cc0_cc1
from preprocessing import optimize_cc0_cc1_predictor

#----------------PREPROCESSING------------------------------

report_targets_distribution(df_hh)
remove_decimals_valley_p_var(df_hh)
df_hh_ing = apply_scaling_oscillators(df_hh)

logscale_volumes(df_hh_ing)


# This helps understanding how to set cc0 and cc1 (classification criteria) that better separate classes
study_distr_pattern_length = False
normalize = False

if study_distr_pattern_length:
    optimize_cc0_cc1(df_hh_ing, normalize)
# It was observed empirically one must look at the distribution of counts instead of the normalized one.
# When studying with normalize=True one can distinguish classes for that predictor but in those cases
# the soft and hard class are basically empty, so it hasn't much value.

# Also study on continous predictors
study_distr_predictor = True
if study_distr_predictor:
    predictor = 'volat_growth'
    optimize_cc0_cc1_predictor(predictor, df_hh_ing)

# PARAMETERS for TARGET CLASSES
cc0 = -5
cc1 = -15
study_rnd_labels = False
show_distributions = False
save_plots = False

# distribution of removed samples based on length
if show_distributions:
    distribution_samples_removed(df_hh_ing, cc0, cc1)
    distribution_samples_length(df_hh_ing, cc0, cc1, save_plots)

# Choose this parameter based on distribution results
min_pattern_length = 0

# Design Matrix with Labels
Xy = create_Xy(df_hh_ing, min_pattern_length, cc0, cc1, study_rnd_labels)

#------------------METHODS

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#------------------METHODS FROM learning.py

from learning import get_X_y
from learning import set_random_search
from learning import start_training

#------------------LEARNING-----------------------------------

X, y = get_X_y(Xy)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)

# Random Search parameters
n_iter = 100
cv = 4
scoring = 'f1_weighted'
rnd_st = 42

show_rs_results = True

rs = set_random_search(n_iter, cv, scoring, rnd_st)
start_training(rs, X_train, y_train, show_rs_results)

rf_tuned = rs.best_estimator_
y_pred = rf_tuned.predict(X_test)

print(classification_report(y_test, y_pred, digits=3, output_dict=False, zero_division='warn'))

cm_df = pd.DataFrame(
    confusion_matrix(y_test, y_pred),
    index = ['actual 0', 'actual 1', 'actual 2'],
    columns = ['pred 0', 'pred 1','pred 2'])
    
sns.heatmap(cm_df, annot=True, linewidth=2., fmt='g')

#----------------SAVING TUNED MODEL------------------

from joblib import dump
dump(rf_tuned, 'rf_tuned.joblib')