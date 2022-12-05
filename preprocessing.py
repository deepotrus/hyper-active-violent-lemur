#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 22:27:43 2022

@author: potruso
"""
import random

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

def report_targets_distribution(df_hh):
    cc0 = -2
    cc1 = -4
    cc2 = -6
    cc3 = -8
    cc4 = -10
    cc5 = -15
    cc6 = -20
    cc7 = -30
    cc8 = -40
    
    conta_1 = 0
    conta_2 = 0
    conta_3 = 0
    conta_4 = 0
    conta_5 = 0
    conta_6 = 0
    conta_7 = 0
    conta_8 = 0
    conta_9 = 0
    conta_10 = 0
    
    for n in range(len(df_hh)):
        t = df_hh['target_p_var'].iloc[n]
        if t > cc0:
            conta_1 += 1
        elif t > cc1:
            conta_2 += 1
        elif t > cc2:
            conta_3 += 1
        elif t > cc3:
            conta_4 += 1
        elif t > cc4:
            conta_5 += 1
        elif t > cc5:
            conta_6 += 1
        elif t > cc6:
            conta_7 += 1
        elif t > cc7:
            conta_8 += 1
        elif t > cc8:
            conta_9 += 1
        else:
            conta_10 += 1
    
    print('class € [0,{}]  \t\t'.format(cc0),conta_1,'  \t', round(100*conta_1/len(df_hh),1),'%')
    print('class € [{},{}]  \t\t'.format(cc0,cc1),conta_2,'  \t', round(100*conta_2/len(df_hh),1),'%')
    print('class € [{},{}]  \t\t'.format(cc1,cc2),conta_3,'  \t', round(100*conta_3/len(df_hh),1),'%')
    print('class € [{},{}]  \t\t'.format(cc2,cc3),conta_4,'  \t', round(100*conta_4/len(df_hh),1),'%')
    print('class € [{},{}]  \t\t'.format(cc3,cc4),conta_5,'  \t', round(100*conta_5/len(df_hh),1),'%')
    print('class € [{},{}]  \t\t'.format(cc4,cc5),conta_6,'  \t', round(100*conta_6/len(df_hh),1),'%')
    print('class € [{},{}]  \t\t'.format(cc5,cc6),conta_7,'  \t', round(100*conta_7/len(df_hh),1),'%')
    print('class € [{},{}]  \t\t'.format(cc6,cc7),conta_8,'  \t', round(100*conta_8/len(df_hh),1),'%')
    print('class € [{},{}]  \t\t'.format(cc7,cc8),conta_9,'  \t', round(100*conta_9/len(df_hh),1),'%')
    print('class € [{},-100] \t\t'.format(cc8),conta_10,'  \t', round(100*conta_10/len(df_hh),1),'%')

def remove_decimals_valley_p_var(df):
    for n in range(len(df)):
        if df['valley_p_var'].iloc[n] - int(df['valley_p_var'].iloc[n]) > -0.5:
            df['valley_p_var'].iloc[n] = int(df['valley_p_var'].iloc[n])
        else:
            df['valley_p_var'].iloc[n] = int(df['valley_p_var'].iloc[n])-1
            
            
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

def apply_scaling_oscillators(df):
    stdscaler = StandardScaler()
    
    df[["stds_rsi_diff"]] = stdscaler.fit_transform(df[["rsi_diff"]])
    df = df.drop('rsi_diff',axis = 1)
    
    df[["stds_rvi_diff"]] = stdscaler.fit_transform(df[["rvi_diff"]])
    df = df.drop('rvi_diff',axis = 1)
    
    return df

def logscale_volumes(df):
    df['pc_Vol_log'] = ''
    df['pc_Vol_log'] = np.log(100 + df['Vol_p_var'])
    df.drop(columns = 'Vol_p_var', inplace = True)
    

def randomize_labels(df):
    retr_class = df['retr_class'].tolist()
    df.pop('retr_class') # pop out the true labels
    randomized_labels = random.sample(retr_class, len(retr_class))
    
    # replace retr_class with randomized labels
    df['retr_class'] = randomized_labels
    
    return df

def conta_samples_excluded(df, min_pattern_length, cc0, cc1):
    # algo for detecting how many hard retracements had been rejected

    conta_hard = 0
    conta_soft = 0
    conta_normal = 0
    
    conta_hard1 = 0
    conta_soft1 = 0
    conta_normal1 = 0
    
    for n in range(len(df)):
        if df['target_p_var'].iloc[n] <= cc1:
            conta_hard += 1
        elif df['target_p_var'].iloc[n] > cc0:
            conta_soft += 1
        else:
            conta_normal += 1
            
    for n in range(len(df[df['delta_days'] > min_pattern_length])):
        if df[df['delta_days'] > min_pattern_length]['target_p_var'].iloc[n] <= cc1:
            conta_hard1 += 1
        elif df[df['delta_days'] > min_pattern_length]['target_p_var'].iloc[n] > cc0:
            conta_soft1 += 1
        else:
            conta_normal1 += 1
    
    return (conta_soft - conta_soft1)/conta_soft, (conta_normal - conta_normal1)/conta_normal, (conta_hard - conta_hard1)/conta_hard

def distribution_samples_length(df, cc0, cc1, save_plots, normalize = False):
    N_norm = 0
    N_soft = 0
    N_hard = 0
    
    # normalization
    if normalize:
        for n in range(len(df)):
            if df['target_p_var'].iloc[n] <= cc1:
                N_hard += 1
            elif df['target_p_var'].iloc[n] > cc0:
                N_soft += 1
            else:
                N_norm += 1
    
    l_soft = list()
    l_norm = list()
    l_hard = list()
    l_4h_candles = range(0,30)
    for length in l_4h_candles:
        conta_hard = 0
        conta_soft = 0
        conta_norm = 0
        for n in range(len(df[df['delta_days'] == length])):
            if df[df['delta_days'] == length]['target_p_var'].iloc[n] <= cc1:
                conta_hard += 1
            elif df[df['delta_days'] == length]['target_p_var'].iloc[n] > cc0:
                conta_soft += 1
            else:
                conta_norm += 1
        if normalize:
            l_soft.append(conta_soft/N_soft)
            l_norm.append(conta_norm/N_norm)
            l_hard.append(conta_hard/N_hard)
        else:
            l_soft.append(conta_soft)
            l_norm.append(conta_norm)
            l_hard.append(conta_hard)
    my_dict = {
        '4h_candles': l_4h_candles,
        'soft': l_soft,
        'norm': l_norm,
        'hard': l_hard
        }
    
    _ = pd.DataFrame(my_dict)
    
    if save_plots:
        sns.set(style="whitegrid", color_codes=True, font_scale = 1)
        sns.lineplot(data = _, x = '4h_candles', y = 'soft', color='green')
        sns.lineplot(data = _, x = '4h_candles', y = 'norm', color='yellow')
        sns.lineplot(data = _, x = '4h_candles', y = 'hard', color='red')
        plt.ylabel('% samples')
        plt.xlabel('4h candles')
        if normalize:
            plt.ylim(0, 0.25)
        else:
            plt.yscale('log')
            plt.ylim(0, 300)
        plt.savefig(r'/home/potruso/Documents/crypto/crypto1 - classic ML/HAVL/distrbs_pattern_length/cc0={}, cc1={}.png'.format(cc0,cc1))
        plt.clf()
    else:    
        sns.set(style="whitegrid", color_codes=True, font_scale = 1)
        sns.lineplot('4h_candles', 'soft', data=_, color='green')
        sns.lineplot('4h_candles', 'norm', data=_, color='yellow')
        sns.lineplot('4h_candles', 'hard', data=_, color='red')
        plt.ylabel('% samples')
        plt.xlabel('4h candles')
        plt.show()
        plt.clf()

def optimize_cc0_cc1(df, normalize):
    l_cc0 = [-1,-2,-3,-4,-5,-6,-7,-8]
    l_cc1 = [-9,-10,-12,-13,-14,-15,-16,-17,-18,-19,-20,-21,-22,-23,-24,-25]
    save_plots = True
    
    for cc0 in l_cc0:
        for cc1 in l_cc1:
            distribution_samples_length(df, cc0, cc1, save_plots, normalize)

def distribution_samples_removed(df_hh_ing, cc0, cc1):
    l_ex_soft = list()
    l_ex_norm = list()
    l_ex_hard = list()
    l_ex_4h_candles = range(0,11)
    for length in l_ex_4h_candles:
        excluded = conta_samples_excluded(df_hh_ing, length, cc0, cc1)
        l_ex_soft.append(excluded[0])
        l_ex_norm.append(excluded[1])
        l_ex_hard.append(excluded[2])
    
    my_dict = {
        'ex_4h_candles': l_ex_4h_candles,
        'ex_soft': l_ex_soft,
        'ex_norm': l_ex_norm,
        'ex_hard': l_ex_hard
        }
    
    df_ex = pd.DataFrame(my_dict)
    
    sns.set(style="whitegrid", color_codes=True, font_scale = 1)
    sns.lineplot('ex_4h_candles', 'ex_soft', data=df_ex, color='green')
    sns.lineplot('ex_4h_candles', 'ex_norm', data=df_ex, color='yellow')
    sns.lineplot('ex_4h_candles', 'ex_hard', data=df_ex, color='red')
    plt.ylabel('excluded samples %')
    plt.xlabel('excluded 4h candles')
    plt.show()

def create_Xy(df, min_pattern_length, cc0, cc1, rnd_labels_study):
    # FOR 3-CLASSIFICATION
    
    # min_pattern_length decided on the distribution of samples removed
    df = df[df['delta_days'] > min_pattern_length]
    
    conditions  = [
        df['target_p_var'] > cc0,
        (df['target_p_var'] > cc1) & (df['target_p_var'] <= cc0),
        df['target_p_var'] <= cc1
    ]
    choices     = [0,1,2]
    
    df["retr_class"] = np.select(conditions, choices, default=np.nan)
    
    # drop target_p_var column
    df = df.drop('target_p_var', axis = 1)

    if rnd_labels_study == True:
        return randomize_labels(df)
    
    return df


def distribution_samples(predictor, df, cc0, cc1):
    ''' Class Distributions for a continous predictor'''
    
    df[['log_{}'.format(predictor)]] = ''
    df['log_{}'.format(predictor)] = np.log(100 + df['{}'.format(predictor)])
    
    # standard scaling of the predictor    
    scaler = MinMaxScaler()
    df[['std_{}'.format(predictor)]] = scaler.fit_transform(df[['{}'.format(predictor)]])
    
    # creating hue variable for histogram
    conditions  = [
        df['target_p_var'] > cc0,
        (df['target_p_var'] > cc1) & (df['target_p_var'] <= cc0),
        df['target_p_var'] <= cc1
    ]
    choices     = ['soft','norm','hard']
    
    df["retr_class"] = np.select(conditions, choices, default=np.nan)
    
    palette = {
        'norm': 'yellow',
        'soft': 'green',
        'hard': 'red'
    }
    
    # create histogram
    sns.set(style="whitegrid", color_codes=True, font_scale = 1)
    sns.histplot(data = df, x = 'log_{}'.format(predictor),
                 hue = 'retr_class', palette = palette,
                 bins = 15, element = 'poly', fill = False
                 )
    plt.yscale('log')
    
    plt.savefig(r'/home/potruso/Documents/crypto/crypto1 - classic ML/HAVL/distrbs_predictor/cc0={}, cc1={}.png'.format(cc0,cc1))
    plt.clf()

def optimize_cc0_cc1_predictor(predictor, df):
    l_cc0 = [-1,-2,-3,-4,-5,-6,-7,-8]
    l_cc1 = [-9,-10,-12,-13,-14,-15,-16,-17,-18,-19,-20,-21,-22,-23,-24,-25]
    
    for cc0 in l_cc0:
        for cc1 in l_cc1:
            distribution_samples(predictor, df, cc0, cc1)












































