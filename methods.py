#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 20:45:23 2022

@author: potruso
"""

import time
import os
import requests

import numpy as np
import pandas as pd
import pandas_ta as ta

from scipy.signal import argrelextrema
from collections import deque

pd.options.mode.chained_assignment = None  # default='warn'

#----------CONVERTING METHODS--------------------------------------

def convert_epoch_to_date(n):
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(n))

#-------------PATTERN RECOGNITION METHODS--------------------------

def getHigherHighs(data: np.array, order, K, tau):
    
    # get indices and values of all highs of given order
    high_idx = argrelextrema(data, np.greater, order=order)[0]
    highs = data[high_idx]
    
    extrema = []
    extr = 0
    ex_deque = deque()
    #last_high_idx = 0
    
    # ensure consecutive highs are higher than previous highs
    for i, idx in enumerate(high_idx):
        #print(i,idx,highs[i],'\tdelta: ', idx - extr)
        
        # implementing constant decorrelation time
        if idx - extr > tau:
            if i == 0:
                ex_deque.append(idx)
                continue

            #higher highs, condition is <
            if highs[i] < highs[i-1]:
                ex_deque.clear()
                ex_deque.append(idx)
            else:
                ex_deque.append(idx)

            if len(ex_deque) == K:
                extrema.append(ex_deque.copy())
                ex_deque.clear()
                extr = extrema[-1][1]
        else:
            ex_deque.clear()
        
    return extrema

def getLowerLows(data: np.array, order, K, tau):
    
    # Get lows
    low_idx = argrelextrema(data, np.less, order=order)[0]
    lows = data[low_idx]
    
    extrema = []
    ex_deque = deque()
    
    for i, idx in enumerate(low_idx):
        if i == 0:
            ex_deque.append(idx)
            continue
                                                            #Lower Lows, condition is >
        if lows[i] > lows[i-1]:
            ex_deque.clear()
            ex_deque.append(idx)
        else:
            ex_deque.append(idx)
            
        if len(ex_deque) == K:
            extrema.append(ex_deque.copy())
            ex_deque.clear()
            ex_deque.append(idx)
    
    return extrema

#----------------METHODS FEATURES---------------------------------

def get_p_var(i,df_i):
    p_var = [100*(df_i['Close'].iloc[k] - df_i['Close'].iloc[0])/df_i['Close'].iloc[0] for k in range(1,len(i),1)]
    
    m = p_var[0]
    
    for k in range(1,len(p_var),1):
        if p_var[k] < 0:
            if p_var[k] < m:
                m = p_var[k]
    
    return round(m,2)

def get_Vol_features(i,df_i):
    # i refers to indices among the days where hh was taking place
    # df refers to dataframe for the indices when hh is occuring
    #Vol_p_var = 100*(df_i['Volume MA'].iloc[-1] - df_i['Volume MA'].iloc[0])/df_i['Volume MA'].iloc[0]
    Vol_p_var = 100*(df_i['Volume'].iloc[-1] - df_i['Volume'].iloc[0])/df_i['Volume'].iloc[0]
    
    conta = 0
    for n in range(len(i)):
        if df_i['Volume'].iloc[n] > df_i['Volume MA'].iloc[n]:
            conta += 1
    
    Vol_sum = conta/len(i)
    
    return Vol_p_var, Vol_sum

def get_rsi_features(i,df_i):
    rsi_diff = df_i['RSI'].iloc[0] - df_i['RSI'].iloc[-1]
    return rsi_diff

def get_rvi_features(i,df_i):
    rvi_p_var = df_i['RVI'].iloc[0] - df_i['RVI'].iloc[-1]
    return rvi_p_var

def get_volat_features(i,df_i):
    volat_growth = df_i['Std Dev'].iloc[-1]/df_i['Std Dev'].iloc[0]
    return volat_growth

def get_price_features(i,df_i):
    price_p_var = 100*(df_i['Close'].iloc[-1] - df_i['Close'].iloc[0])/df_i['Close'].iloc[0]
    return price_p_var

def get_trend_features(i,df_i):
    trend_rapp = (df_i['SMA_60'].iloc[-1] - df_i['SMA_300'].iloc[-1])/(df_i['SMA_60'].iloc[0] - df_i['SMA_300'].iloc[0])
    return trend_rapp

def get_candle_resist(i,df_i):
    resist = 0
    second_max_close = df_i['Close'].iloc[-1]
    once = False

    for n in range(len(i)):
        if df_i['High'].iloc[n] > second_max_close:
            if df_i['High'].iloc[n] > resist:
                once = True
                resist = df_i['High'].iloc[n]
    if once == False:
        resist = second_max_close
        
    resist_rapp = resist/second_max_close
    return resist_rapp

def get_candle_valley(i,df_i):
    candle_valley = df_i['Low'].iloc[0]
    for n in range(1,len(i)-1,1):
        if df_i['Low'].iloc[n] < candle_valley:
            candle_valley = df_i['Low'].iloc[n]
    valley_p_var = 100*(candle_valley - df_i['Close'].iloc[0])/df_i['Close'].iloc[0]
    return valley_p_var

def get_btc_mk_cond(i,df_i):
    return df_i['btc_cond'].iloc[-1]

#--------------------METHODS for Time-Series Sample Visualization

def get_open_timeseries(i,df_i):
    open = list()

    for n in range(len(i)):
        open.append(df_i['Open'].iloc[n])

    return open

def get_high_timeseries(i,df_i):
    high = list()

    for n in range(len(i)):
        high.append(df_i['High'].iloc[n])
    
    return high

def get_low_timeseries(i,df_i):
    low = list()

    for n in range(len(i)):
        low.append(df_i['Low'].iloc[n])
    
    return low

def get_close_timeseries(i,df_i):
    close = list()

    for n in range(len(i)):
        close.append(df_i['Close'].iloc[n])
    
    return close

def get_volume_timeseries(i,df_i):
    volume = list()

    for n in range(len(i)):
        volume.append(df_i['Volume'].iloc[n])
    
    return volume

#--------------------METHODS--------------------------------------

def generate_samples(crypto_symbols, lf, order, tau, K, period, slow_period, fast_period, printing, show_samples):
    # for each symbol create dataframe with features and targets
    i_sample = 0
    for i_symb in crypto_symbols:
        data = pd.read_csv(r'/home/potruso/Documents/crypto/crypto_data/4h_data/{}.csv'.format(i_symb))
        
        
        # adding btc market condition as column
        #data['btc_cond'] = ''
        #attach_btc_market_condition(data, time_cross)
        
        # converting epoch to datetime
        for n in range(len(data)):
            data['time'][n] = convert_epoch_to_date(data['time'][n])
        
        # creating rolling std
        std_column = np.array(data['close'].rolling(period).std())
        var_column = np.array(data['close'].rolling(period).var())
        
        # add 'std_column' array as new column in DataFrame, std is also referred to as volatility
        data['std_{}'.format(period)] = std_column.tolist()
        
        # adding indicators, new columns will be named as 'indicator name_period'
        data.ta.rsi(close = 'close',length=period,append=True)
        data.ta.rvi(close = 'close',length=period,append=True)
        data.ta.sma(close = 'Volume',length=period,append=True)
        data.ta.sma(close = 'close',length=fast_period,append=True)
        data.ta.sma(close = 'close',length=slow_period,append=True)
        
        # renaming columns
        data.rename(columns = {
            'time': 'Time',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'std_{}'.format(period): 'Std Dev',
            'RSI_{}'.format(period): 'RSI',
            'RVI_{}'.format(period): 'RVI',
            'SMA_{}'.format(period): 'Volume MA'
        }, inplace = True)
        
        #data.dropna(subset = ['RSI'], inplace=True)
        
        df = data.copy(deep=True)
        
        price = np.array(df['Close'])
        dates = df.index
        
        hh = getHigherHighs(price, order, K, tau)
        ll = getLowerLows(price, order, K, tau)
        
        # output of hh = [ deque([20, 28]), deque([63, 81]), ... ,deque([5431, 5436]) ]
        
        # get indices of higher highs
        hh_idx = np.zeros((len(hh),K))
        for m in range(len(hh)):
            hh_idx[m] = np.array([int(i) for i in hh[m]])
        
        # output of hh_idx = [ [  20.   28.]
                              #[  63.   81.]
                              #     ...
                              #[5431. 5436.] ]
        
        ll_idx = np.zeros((len(ll),K))
        for m in range(len(ll)):
            ll_idx[m] = np.array([int(i) for i in ll[m]])
        
        # converting indices to int type
        hh_idx = hh_idx.astype(int)
        ll_idx = ll_idx.astype(int)
        
        # get indices of higher highs confirmation
        hh_idx_conf = hh_idx[:,K-1]
        hh_idx_conf = np.array([int(i) + order for i in hh_idx_conf]) 
        
        ll_idx_conf = ll_idx[:,K-1]
        ll_idx_conf = np.array([int(i) + order for i in ll_idx_conf]) 
        
        # deleting indices out of data range
        if hh_idx_conf[-1] >= len(df):
            hh_idx_conf = np.delete(hh_idx_conf,len(hh_idx_conf)-1)
            
        if ll_idx_conf[-1] >= len(df):
            ll_idx_conf = np.delete(ll_idx_conf,len(ll_idx_conf)-1)
        
        if printing == True:
            print('\nMaking {}'.format(i_symb))
            print('# samples for Higher Highs: ', hh_idx.shape[0])
            print('# samples for Lower Lows: ', ll_idx.shape[0])
        
        # duration_hh is a vector which contains all delta_time within hh occured
        duration_hh = hh_idx[:-1,1] - hh_idx[:-1,0]
        
        # convert duration_hh in int for indices
        duration_hh = np.array([int(i) for i in duration_hh]) 
        
        idx_after_hh = [[int(hh_idx[j,1]) + i for i in range(0,int(lf*(duration_hh[j]+1)),1)] for j in range(len(duration_hh))]
        
        # pseudo-matrix indices for the next days after the hh (not the confirmation signal) within delta_days
        idx_p_var = [[int(hh_idx[j,1]) + i for i in range(0,int(lf*(duration_hh[j]+1)),1)] for j in range(len(duration_hh))]
        # pseudo-matrix indices for days while hh is occuring
        idx_hh =  [[int(hh_idx[j,0]) + i for i in range(0,duration_hh[j]+1)] for j in range(len(duration_hh))]
        
        # initializing target values and then filling
        target_p_var = np.zeros(len(duration_hh))
    
        #getting %var and making target vector
        for h in range(0,len(duration_hh)-1,1):
            target_p_var[h] = get_p_var(idx_p_var[h],df.iloc[idx_p_var[h]])
        
        # matrix indices for the days when hh is forming
        idx_forming_hh = [[hh_idx_conf[j] - order - duration_hh[j] + i for i in range(duration_hh[j]+1)] for j in range(len(duration_hh))]
        
        # get features
        Vol_p_var = np.zeros(len(duration_hh))
        Vol_sum = np.zeros(len(duration_hh))
        for n in range(len(duration_hh)):
            Vol_p_var[n], Vol_sum[n] = get_Vol_features(idx_forming_hh[n],df.iloc[idx_forming_hh[n]])
        
        rsi_p_var = np.zeros(len(duration_hh))
        for n in range(len(duration_hh)):
            rsi_p_var[n] = get_rsi_features(idx_forming_hh[n],df.iloc[idx_forming_hh[n]])
        
        rvi_p_var = np.zeros(len(duration_hh))
        for n in range(len(duration_hh)):
            rvi_p_var[n] = get_rvi_features(idx_forming_hh[n],df.iloc[idx_forming_hh[n]])
    
        volat_p_var = np.zeros(len(duration_hh))
        for n in range(len(duration_hh)):
            volat_p_var[n] = get_volat_features(idx_forming_hh[n],df.iloc[idx_forming_hh[n]])
        
        trend_rapp = np.zeros(len(duration_hh))
        for n in range(len(duration_hh)):
            trend_rapp[n] = get_trend_features(idx_forming_hh[n],df.iloc[idx_forming_hh[n]])
        
        price_p_var = np.zeros(len(duration_hh))
        for n in range(len(duration_hh)):
            price_p_var[n] = get_price_features(idx_forming_hh[n],df.iloc[idx_forming_hh[n]])
        
        resist_p_var = np.zeros(len(duration_hh))
        for n in range(len(duration_hh)):
            resist_p_var[n] = get_candle_resist(idx_hh[n],df.iloc[idx_hh[n]])
        
        valley_p_var = np.zeros(len(duration_hh))
        for n in range(len(duration_hh)):
            valley_p_var[n] = get_candle_valley(idx_hh[n],df.iloc[idx_hh[n]])
        
        if show_samples == True:
            df_ohlc = list()
            
            # create matrix of indices of all samples gathering indices from forming hh and after hh
            M_idx_pattern = list()
            for j in range(len(duration_hh)):
                idx_pattern = list()
                for i in range(0,2*order):
                    idx_pattern.append(idx_forming_hh[j][0] - 2*order + i)
                    
                for i in range(0, len(idx_forming_hh[j]) - 1):
                    idx_pattern.append(idx_forming_hh[j][i])
                
                for i in range(len(idx_after_hh[j])):
                    idx_pattern.append(idx_after_hh[j][i])
                
                M_idx_pattern.append(idx_pattern)
            
            for n in range(len(duration_hh)): # for all samples in this market
                open = get_open_timeseries(M_idx_pattern[n], df.iloc[M_idx_pattern[n]])
                high = get_high_timeseries(M_idx_pattern[n], df.iloc[M_idx_pattern[n]])
                low = get_low_timeseries(M_idx_pattern[n], df.iloc[M_idx_pattern[n]])
                close = get_close_timeseries(M_idx_pattern[n], df.iloc[M_idx_pattern[n]])
                volume = get_volume_timeseries(M_idx_pattern[n], df.iloc[M_idx_pattern[n]])
                
                # values of labels
                #target_p_var = get_p_var(idx_after_hh[n], df.iloc[idx_after_hh[n]])
                
                zipped = list(zip(open, high, low, close, volume))
                
                df_ohlc = pd.DataFrame(zipped, columns=['open', 'high', 'low', 'close', 'Volume'])
                
                df_ohlc.to_csv('/home/potruso/Documents/crypto/crypto1 - classic ML/HAVL/samples/sample_{}.csv'.format(i_sample), index = False)
                i_sample += 1
                
        #btc_cond  = []
        #for n in range(len(delta_days)):
        #    btc_cond.append(get_btc_mk_cond(idx_hh[n],df.iloc[idx_hh[n]]))
        #btc_cond = np.array((btc_cond))
        
        data_learning = np.vstack((duration_hh,price_p_var,resist_p_var,valley_p_var,
                                   rsi_p_var,rvi_p_var,volat_p_var,trend_rapp,Vol_p_var,Vol_sum,target_p_var)).T
        # creating list of column names
        column_values = ['delta_days', 'price_p_var','resist_rapp','valley_p_var',
                         'rsi_diff','rvi_diff','volat_growth','trend_rapp','Vol_p_var','Vol_sum','target_p_var']
        
        # creating list of index names
        index_values = df.iloc[hh_idx[:-1,0]].index
        
        # creating the dataframe
        df_learning_hh = pd.DataFrame(data = data_learning, 
                                      index = index_values, 
                                      columns = column_values)
        
        # exporting dataframe
        df_learning_hh.to_csv('df_hh_{}.csv'.format(i_symb))
        
        
def create_unique_dataframe():
    df_hh_doge = pd.read_csv("df_hh_binance_dogeusd.csv")
    df_hh_doge = df_hh_doge.drop(columns='Unnamed: 0')
    os.remove('df_hh_binance_dogeusd.csv')
    
    df_hh_ada = pd.read_csv("df_hh_binance_adausd.csv")
    df_hh_ada = df_hh_ada.drop(columns='Unnamed: 0')
    os.remove('df_hh_binance_adausd.csv')
    
    df_hh_ftm = pd.read_csv("df_hh_binance_ftmusdt.csv")
    df_hh_ftm = df_hh_ftm.drop(columns='Unnamed: 0')
    os.remove('df_hh_binance_ftmusdt.csv')
    
    df_hh_iotx = pd.read_csv("df_hh_binance_iotxusdt.csv")
    df_hh_iotx = df_hh_iotx.drop(columns='Unnamed: 0')
    os.remove('df_hh_binance_iotxusdt.csv')
    
    df_hh_ltc = pd.read_csv("df_hh_bitfinex_ltcusd.csv")
    df_hh_ltc = df_hh_ltc.drop(columns='Unnamed: 0')
    os.remove('df_hh_bitfinex_ltcusd.csv')
    
    df_hh_one = pd.read_csv("df_hh_binance_oneusd.csv")
    df_hh_one = df_hh_one.drop(columns='Unnamed: 0')
    os.remove('df_hh_binance_oneusd.csv')
    
    df_hh_shib = pd.read_csv("df_hh_poloniex_shibusdt.csv")
    df_hh_shib = df_hh_shib.drop(columns='Unnamed: 0')
    os.remove('df_hh_poloniex_shibusdt.csv')
    
    df_hh_trx = pd.read_csv("df_hh_binance_trxusd.csv")
    df_hh_trx = df_hh_trx.drop(columns='Unnamed: 0')
    os.remove('df_hh_binance_trxusd.csv')
    
    df_hh_vet = pd.read_csv("df_hh_binance_vetusd.csv")
    df_hh_vet = df_hh_vet.drop(columns='Unnamed: 0')
    os.remove('df_hh_binance_vetusd.csv')
    
    df_hh_xlm = pd.read_csv("df_hh_binance_xlmusdt.csv")
    df_hh_xlm = df_hh_xlm.drop(columns='Unnamed: 0')
    os.remove('df_hh_binance_xlmusdt.csv')
    
    df_hh_omg = pd.read_csv("df_hh_coinbase_omgusd.csv")
    df_hh_omg = df_hh_omg.drop(columns='Unnamed: 0')
    os.remove('df_hh_coinbase_omgusd.csv')
    
    df_hh_luna = pd.read_csv("df_hh_binance_lunausdt.csv")
    df_hh_luna = df_hh_luna.drop(columns='Unnamed: 0')
    os.remove('df_hh_binance_lunausdt.csv')
    
    df_hh_xrp = pd.read_csv("df_hh_binance_xrpusd.csv")
    df_hh_xrp = df_hh_xrp.drop(columns='Unnamed: 0')
    os.remove('df_hh_binance_xrpusd.csv')
    
    df_hh_bnb = pd.read_csv("df_hh_ftx_bnbusd.csv")
    df_hh_bnb = df_hh_bnb.drop(columns='Unnamed: 0')
    os.remove('df_hh_ftx_bnbusd.csv')
    
    df_hh_sol = pd.read_csv("df_hh_ftx_solusd.csv")
    df_hh_sol = df_hh_sol.drop(columns='Unnamed: 0')
    os.remove('df_hh_ftx_solusd.csv')
    
    #------------------------------------------------new data
    
    df_hh_eth = pd.read_csv("df_hh_bitstamp_ethusd.csv")
    df_hh_eth = df_hh_eth.drop(columns='Unnamed: 0')
    os.remove('df_hh_bitstamp_ethusd.csv')
    
    df_hh_vtho = pd.read_csv("df_hh_newdata_binance_vthousdt.csv")
    df_hh_vtho = df_hh_vtho.drop(columns='Unnamed: 0')
    os.remove('df_hh_newdata_binance_vthousdt.csv')
    
    df_hh_avax = pd.read_csv("df_hh_newdata_bitfinex_avaxusd.csv")
    df_hh_avax = df_hh_avax.drop(columns='Unnamed: 0')
    os.remove('df_hh_newdata_bitfinex_avaxusd.csv')
    
    df_hh_dot = pd.read_csv("df_hh_newdata_bitfinex_dotusd.csv")
    df_hh_dot = df_hh_dot.drop(columns='Unnamed: 0')
    os.remove('df_hh_newdata_bitfinex_dotusd.csv')
    
    df_hh_egld = pd.read_csv("df_hh_newdata_bitfinex_egldusd.csv")
    df_hh_egld = df_hh_egld.drop(columns='Unnamed: 0')
    os.remove('df_hh_newdata_bitfinex_egldusd.csv')
    
    df_hh_fil = pd.read_csv("df_hh_newdata_bitfinex_filusd.csv")
    df_hh_fil = df_hh_fil.drop(columns='Unnamed: 0')
    os.remove('df_hh_newdata_bitfinex_filusd.csv')
    
    df_hh_icp = pd.read_csv("df_hh_newdata_bitfinex_icpusd.csv")
    df_hh_icp = df_hh_icp.drop(columns='Unnamed: 0')
    os.remove('df_hh_newdata_bitfinex_icpusd.csv')
    
    df_hh_link = pd.read_csv("df_hh_newdata_bitfinex_linkusd.csv")
    df_hh_link = df_hh_link.drop(columns='Unnamed: 0')
    os.remove('df_hh_newdata_bitfinex_linkusd.csv')
    
    df_hh_matic = pd.read_csv("df_hh_newdata_bitfinex_maticusd.csv")
    df_hh_matic = df_hh_matic.drop(columns='Unnamed: 0')
    os.remove('df_hh_newdata_bitfinex_maticusd.csv')
    
    df_hh_theta = pd.read_csv("df_hh_newdata_bitfinex_thetausd.csv")
    df_hh_theta = df_hh_theta.drop(columns='Unnamed: 0')
    os.remove('df_hh_newdata_bitfinex_thetausd.csv')
    
    df_hh_xmr = pd.read_csv("df_hh_newdata_bitfinex_xmrusd.csv")
    df_hh_xmr = df_hh_xmr.drop(columns='Unnamed: 0')
    os.remove('df_hh_newdata_bitfinex_xmrusd.csv')
    
    df_hh_xtz = pd.read_csv("df_hh_newdata_bitfinex_xtzusd.csv")
    df_hh_xtz = df_hh_xtz.drop(columns='Unnamed: 0')
    os.remove('df_hh_newdata_bitfinex_xtzusd.csv')
    
    df_hh_algo = pd.read_csv("df_hh_newdata_ftx_algousd.csv")
    df_hh_algo = df_hh_algo.drop(columns='Unnamed: 0')
    os.remove('df_hh_newdata_ftx_algousd.csv')
    
    df_hh_atom = pd.read_csv("df_hh_newdata_ftx_atomusd.csv")
    df_hh_atom = df_hh_atom.drop(columns='Unnamed: 0')
    os.remove('df_hh_newdata_ftx_atomusd.csv')
    
    df_hh_bch = pd.read_csv("df_hh_newdata_ftx_bchusd.csv")
    df_hh_bch = df_hh_bch.drop(columns='Unnamed: 0')
    os.remove('df_hh_newdata_ftx_bchusd.csv')
    
    df_hh_cro = pd.read_csv("df_hh_newdata_ftx_crousd.csv")
    df_hh_cro = df_hh_cro.drop(columns='Unnamed: 0')
    os.remove('df_hh_newdata_ftx_crousd.csv')
    
    df_hh_hnt = pd.read_csv("df_hh_newdata_ftx_hntusdt.csv")
    df_hh_hnt = df_hh_hnt.drop(columns='Unnamed: 0')
    os.remove('df_hh_newdata_ftx_hntusdt.csv')
    
    df_hh_mana = pd.read_csv("df_hh_newdata_ftx_manausd.csv")
    df_hh_mana = df_hh_mana.drop(columns='Unnamed: 0')
    os.remove('df_hh_newdata_ftx_manausd.csv')
    
    df_hh_uni = pd.read_csv("df_hh_newdata_ftx_uniusd.csv")
    df_hh_uni = df_hh_uni.drop(columns='Unnamed: 0')
    os.remove('df_hh_newdata_ftx_uniusd.csv')
    
    frames_hh = [df_hh_doge, df_hh_ada, df_hh_ftm, df_hh_iotx, df_hh_ltc, df_hh_one, df_hh_shib, df_hh_trx,
             df_hh_vet, df_hh_xlm, df_hh_omg, df_hh_luna, df_hh_xrp, df_hh_bnb, df_hh_sol,
             df_hh_eth, df_hh_vtho, df_hh_avax, df_hh_dot, df_hh_egld, df_hh_fil, df_hh_icp,
             df_hh_link, df_hh_matic, df_hh_theta, df_hh_xmr, df_hh_xtz, df_hh_algo, df_hh_atom,
             df_hh_bch, df_hh_cro, df_hh_hnt, df_hh_mana, df_hh_uni]
    
    df_hh = pd.concat(frames_hh,ignore_index=True)
    
    # deleting rows with nan values in trend_rapp column
    df_hh = df_hh[df_hh['trend_rapp'].notna()]

    # dropping NaN values from indicators that weren't still ready
    #df_learning_hh.dropna(subset = ['rsi_diff'], inplace=True)
    #df_learning_hh.dropna(subset = ['rvi_diff'], inplace=True)
    
    # removing inf and empty values by replacing them with nan
    nan_value = float("NaN")
    df_hh.replace('', nan_value, inplace=True)
    df_hh.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_hh.dropna(subset = ['Vol_p_var'], inplace=True)
    df_hh.dropna(subset = ['volat_growth'], inplace=True)
    
    # drop extremas values in features
    df_hh = df_hh[df_hh['price_p_var'] != 0]
    df_hh = df_hh[df_hh['target_p_var'] != 0]
    df_hh = df_hh[df_hh['resist_rapp'] != 0]
    df_hh = df_hh[df_hh['Vol_p_var'] != -100]

    return df_hh





def generate_samples_api(crypto_symbols, list_apis, periods, lf, order, tau, K, period, slow_period, fast_period, printing, show_samples):
    i_sample = 0
    i_symb = 0
    for i_api in list_apis:
        
        resp = requests.get(
            i_api,
            data = {'CloseTime':1659304800}, # setted to 1st August 2022
            params = {'periods':periods})
        
        if resp.ok == True:
            data_json = resp.json()
            data = pd.DataFrame(data_json['result'][periods], columns=['time','open','high','low','close','Volume','NA'])
            data.drop(columns='NA', inplace = True)
            
            
            for n in range(len(data)):
                data['time'][n] = convert_epoch_to_date(data['time'][n])

            std_column = np.array(data['close'].rolling(period).std())
            var_column = np.array(data['close'].rolling(period).var())

            data['std_{}'.format(period)] = std_column.tolist()

            data.ta.rsi(close = 'close',length=period,append=True)
            data.ta.rvi(close = 'close',length=period,append=True)
            data.ta.sma(close = 'Volume',length=period,append=True)
            data.ta.sma(close = 'close',length=fast_period,append=True)
            data.ta.sma(close = 'close',length=slow_period,append=True)

            data.rename(columns = {
                'time': 'Time',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'std_{}'.format(period): 'Std Dev',
                'RSI_{}'.format(period): 'RSI',
                'RVI_{}'.format(period): 'RVI',
                'SMA_{}'.format(period): 'Volume MA'
            }, inplace = True)
            
            df = data.copy(deep=True)
            
            price = np.array(df['Close'])
            dates = df.index
            
            hh = getHigherHighs(price, order, K, tau)
            ll = getLowerLows(price, order, K, tau)
            
            # output of hh = [ deque([20, 28]), deque([63, 81]), ... ,deque([5431, 5436]) ]
            
            # get indices of higher highs
            hh_idx = np.zeros((len(hh),K))
            for m in range(len(hh)):
                hh_idx[m] = np.array([int(i) for i in hh[m]])
            
            # output of hh_idx = [ [  20.   28.]
                                  #[  63.   81.]
                                  #     ...
                                  #[5431. 5436.] ]
            
            ll_idx = np.zeros((len(ll),K))
            for m in range(len(ll)):
                ll_idx[m] = np.array([int(i) for i in ll[m]])
            
            # converting indices to int type
            hh_idx = hh_idx.astype(int)
            ll_idx = ll_idx.astype(int)
            
            # get indices of higher highs confirmation
            hh_idx_conf = hh_idx[:,K-1]
            hh_idx_conf = np.array([int(i) + order for i in hh_idx_conf]) 
            
            ll_idx_conf = ll_idx[:,K-1]
            ll_idx_conf = np.array([int(i) + order for i in ll_idx_conf]) 
            
            # deleting indices out of data range
            if hh_idx_conf[-1] >= len(df):
                hh_idx_conf = np.delete(hh_idx_conf,len(hh_idx_conf)-1)
                
            if ll_idx_conf[-1] >= len(df):
                ll_idx_conf = np.delete(ll_idx_conf,len(ll_idx_conf)-1)
            
            if printing == True:
                print('\nMaking {}'.format(i_api))
                print('# samples for Higher Highs: ', hh_idx.shape[0])
                print('# samples for Lower Lows: ', ll_idx.shape[0])
            
            # duration_hh is a vector which contains all delta_time within hh occured
            duration_hh = hh_idx[:-1,1] - hh_idx[:-1,0]
            
            # convert duration_hh in int for indices
            duration_hh = np.array([int(i) for i in duration_hh]) 
            
            idx_after_hh = [[int(hh_idx[j,1]) + i for i in range(0,int(lf*(duration_hh[j]+1)),1)] for j in range(len(duration_hh))]
            
            # pseudo-matrix indices for the next days after the hh (not the confirmation signal) within delta_days
            idx_p_var = [[int(hh_idx[j,1]) + i for i in range(0,int(lf*(duration_hh[j]+1)),1)] for j in range(len(duration_hh))]
            # pseudo-matrix indices for days while hh is occuring
            idx_hh =  [[int(hh_idx[j,0]) + i for i in range(0,duration_hh[j]+1)] for j in range(len(duration_hh))]
            
            # initializing target values and then filling
            target_p_var = np.zeros(len(duration_hh))
        
            #getting %var and making target vector
            for h in range(0,len(duration_hh)-1,1):
                target_p_var[h] = get_p_var(idx_p_var[h],df.iloc[idx_p_var[h]])
            
            # matrix indices for the days when hh is forming
            idx_forming_hh = [[hh_idx_conf[j] - order - duration_hh[j]+i for i in range(duration_hh[j]+1)] for j in range(len(duration_hh))]
            
            # get features
            Vol_p_var = np.zeros(len(duration_hh))
            Vol_sum = np.zeros(len(duration_hh))
            for n in range(len(duration_hh)):
                Vol_p_var[n], Vol_sum[n] = get_Vol_features(idx_forming_hh[n],df.iloc[idx_forming_hh[n]])
            
            rsi_p_var = np.zeros(len(duration_hh))
            for n in range(len(duration_hh)):
                rsi_p_var[n] = get_rsi_features(idx_forming_hh[n],df.iloc[idx_forming_hh[n]])
            
            rvi_p_var = np.zeros(len(duration_hh))
            for n in range(len(duration_hh)):
                rvi_p_var[n] = get_rvi_features(idx_forming_hh[n],df.iloc[idx_forming_hh[n]])
        
            volat_p_var = np.zeros(len(duration_hh))
            for n in range(len(duration_hh)):
                volat_p_var[n] = get_volat_features(idx_forming_hh[n],df.iloc[idx_forming_hh[n]])
            
            trend_rapp = np.zeros(len(duration_hh))
            for n in range(len(duration_hh)):
                trend_rapp[n] = get_trend_features(idx_forming_hh[n],df.iloc[idx_forming_hh[n]])
            
            price_p_var = np.zeros(len(duration_hh))
            for n in range(len(duration_hh)):
                price_p_var[n] = get_price_features(idx_forming_hh[n],df.iloc[idx_forming_hh[n]])
            
            resist_p_var = np.zeros(len(duration_hh))
            for n in range(len(duration_hh)):
                resist_p_var[n] = get_candle_resist(idx_hh[n],df.iloc[idx_hh[n]])
            
            valley_p_var = np.zeros(len(duration_hh))
            for n in range(len(duration_hh)):
                valley_p_var[n] = get_candle_valley(idx_hh[n],df.iloc[idx_hh[n]])
            
            if show_samples == True:
                df_ohlc = list()
                
                # create matrix of indices of all samples gathering indices from forming hh and after hh
                M_idx_pattern = list()
                for j in range(len(duration_hh)):
                    idx_pattern = list()
                    for i in range(0,2*order):
                        idx_pattern.append(idx_forming_hh[j][0] - 2*order + i)
                        
                    for i in range(0, len(idx_forming_hh[j]) - 1):
                        idx_pattern.append(idx_forming_hh[j][i])
                    
                    for i in range(len(idx_after_hh[j])):
                        idx_pattern.append(idx_after_hh[j][i])
                    
                    M_idx_pattern.append(idx_pattern)
                
                for n in range(len(duration_hh)): # for all samples in this market
                    open = get_open_timeseries(M_idx_pattern[n], df.iloc[M_idx_pattern[n]])
                    high = get_high_timeseries(M_idx_pattern[n], df.iloc[M_idx_pattern[n]])
                    low = get_low_timeseries(M_idx_pattern[n], df.iloc[M_idx_pattern[n]])
                    close = get_close_timeseries(M_idx_pattern[n], df.iloc[M_idx_pattern[n]])
                    volume = get_volume_timeseries(M_idx_pattern[n], df.iloc[M_idx_pattern[n]])
                    
                    # values of labels
                    #target_p_var = get_p_var(idx_after_hh[n], df.iloc[idx_after_hh[n]])
                    
                    zipped = list(zip(open, high, low, close, volume))
                    
                    df_ohlc = pd.DataFrame(zipped, columns=['open', 'high', 'low', 'close', 'Volume'])
                    
                    df_ohlc.to_csv('/home/potruso/Documents/crypto/crypto1 - classic ML/HAVL/samples_test/sample_{}.csv'.format(i_sample), index = False)
                    i_sample += 1

            data_testing_hh = np.vstack((duration_hh,price_p_var,resist_p_var,valley_p_var,
                                       rsi_p_var,rvi_p_var,volat_p_var,trend_rapp,Vol_p_var,Vol_sum,target_p_var)).T
            # creating list of column names
            column_values = ['delta_days', 'price_p_var','resist_rapp','valley_p_var',
                             'rsi_diff','rvi_diff','volat_growth','trend_rapp','Vol_p_var','Vol_sum','target_p_var']
            
            # creating list of index names
            index_values = df.iloc[hh_idx[:-1,0]].index
            
            # creating the dataframe
            data_testing_hh = pd.DataFrame(data = data_testing_hh, 
                                          index = index_values, 
                                          columns = column_values)
            
            # exporting dataframe
            data_testing_hh.to_csv('df_hh_{}.csv'.format(crypto_symbols[i_symb]))
            i_symb += 1


crypto_symbols = ['ethusd','xrpusd','dogeusd','omgusd','bnbusd','maticusd','dotusd','ltcusd','shibusd']


def create_unique_api_dataframe():
    df_hh_ada = pd.read_csv("df_hh_adausd.csv")
    df_hh_ada = df_hh_ada.drop(columns='Unnamed: 0')
    os.remove('df_hh_adausd.csv')
    
    df_hh_ftm = pd.read_csv("df_hh_ftmusd.csv")
    df_hh_ftm = df_hh_ftm.drop(columns='Unnamed: 0')
    os.remove('df_hh_ftmusd.csv')
    
    df_hh_eth = pd.read_csv("df_hh_ethusd.csv")
    df_hh_eth = df_hh_eth.drop(columns='Unnamed: 0')
    os.remove('df_hh_ethusd.csv')
    
    df_hh_xrp = pd.read_csv("df_hh_xrpusd.csv")
    df_hh_xrp = df_hh_xrp.drop(columns='Unnamed: 0')
    os.remove('df_hh_xrpusd.csv')
    
    df_hh_doge = pd.read_csv("df_hh_dogeusd.csv")
    df_hh_doge = df_hh_doge.drop(columns='Unnamed: 0')
    os.remove('df_hh_dogeusd.csv')
    
    df_hh_omg = pd.read_csv("df_hh_omgusd.csv")
    df_hh_omg = df_hh_omg.drop(columns='Unnamed: 0')
    os.remove('df_hh_omgusd.csv')
    
    df_hh_bnb = pd.read_csv("df_hh_bnbusd.csv")
    df_hh_bnb = df_hh_bnb.drop(columns='Unnamed: 0')
    os.remove('df_hh_bnbusd.csv')
    
    df_hh_matic = pd.read_csv("df_hh_maticusd.csv")
    df_hh_matic = df_hh_matic.drop(columns='Unnamed: 0')
    os.remove('df_hh_maticusd.csv')
    
    df_hh_dot = pd.read_csv("df_hh_dotusd.csv")
    df_hh_dot = df_hh_dot.drop(columns='Unnamed: 0')
    os.remove('df_hh_dotusd.csv')
    
    df_hh_ltc = pd.read_csv("df_hh_ltcusd.csv")
    df_hh_ltc = df_hh_ltc.drop(columns='Unnamed: 0')
    os.remove('df_hh_ltcusd.csv')
    
    df_hh_shib = pd.read_csv("df_hh_shibusd.csv")
    df_hh_shib = df_hh_shib.drop(columns='Unnamed: 0')
    os.remove('df_hh_shibusd.csv')
    
    frames_hh = [
        df_hh_ada,
        df_hh_ftm,
        df_hh_eth,
        df_hh_xrp,
        df_hh_doge,
        df_hh_omg,
        df_hh_bnb,
        df_hh_matic,
        df_hh_dot,
        df_hh_ltc,
        df_hh_shib
        ]
    
    df_hh = pd.concat(frames_hh, ignore_index=True)
    
    # deleting rows with nan values in trend_rapp column
    df_hh = df_hh[df_hh['trend_rapp'].notna()]

    # dropping NaN values from indicators that weren't still ready
    #df_learning_hh.dropna(subset = ['rsi_diff'], inplace=True)
    #df_learning_hh.dropna(subset = ['rvi_diff'], inplace=True)
    
    # removing inf and empty values by replacing them with nan
    nan_value = float("NaN")
    df_hh.replace('', nan_value, inplace=True)
    df_hh.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_hh.dropna(subset = ['Vol_p_var'], inplace=True)
    df_hh.dropna(subset = ['volat_growth'], inplace=True)
    
    # drop extremas values in features
    df_hh = df_hh[df_hh['price_p_var'] != 0]
    df_hh = df_hh[df_hh['target_p_var'] != 0]
    df_hh = df_hh[df_hh['resist_rapp'] != 0]
    df_hh = df_hh[df_hh['Vol_p_var'] != -100]

    return df_hh
































