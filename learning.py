#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 09:51:37 2022

@author: potruso
"""

import random
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
    
def get_X_y(Xy):
    y = np.array(Xy['retr_class'])
    y = y.astype(int)
    Xy = Xy.drop('retr_class', axis = 1)
    X = np.array(Xy)
    
    return X, y
    

def set_random_search(n_iter, cv, scoring, rnd_st):
    n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 20)]
    max_features = ['log2', 'sqrt']
    max_depth = [int(x) for x in np.linspace(start = 1, stop = 20, num = 20)]
    min_samples_split = [int(x) for x in np.linspace(start = 2, stop = 50, num = 50)]
    min_samples_leaf = [int(x) for x in np.linspace(start = 2, stop = 50, num = 50)]
    bootstrap = [True, False]
    
    hyprms = {
        'n_estimators': n_estimators,
        'max_features': max_features,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'bootstrap': bootstrap}
    
    rfc = RandomForestClassifier(class_weight = 'balanced')
    rs = RandomizedSearchCV(
        rfc,
        hyprms,
        n_iter = n_iter,
        cv = cv,
        scoring = scoring,
        verbose = 1,
        n_jobs = 1,
        random_state = rnd_st)
    
    return rs

def plot_rs_results(rs):
    rs_df = pd.DataFrame(rs.cv_results_).sort_values('rank_test_score').reset_index(drop=True)
    rs_df = rs_df.drop([
                'mean_fit_time', 
                'std_fit_time', 
                'mean_score_time',
                'std_score_time', 
                'params', 
                'split0_test_score', 
                'split1_test_score', 
                'split2_test_score', 
                'std_test_score'],
                axis=1)
    
    fig, axs = plt.subplots(ncols=3, nrows=2)
    sns.set(style="whitegrid", color_codes=True, font_scale = 2)
    fig.set_size_inches(25,20)
    
    sns.barplot(x='param_n_estimators', y='mean_test_score', data=rs_df, ax=axs[0,0], color='lightgrey')
    axs[0,0].set_ylim([.45,.80])
    axs[0,0].set_title(label = 'n_estimators', size=30, weight='bold')
    axs[0,0].tick_params(axis='x', rotation=90, labelsize=12)
    
    sns.barplot(x='param_min_samples_split', y='mean_test_score', data=rs_df, ax=axs[0,1], color='coral')
    axs[0,1].set_ylim([.45,.80])
    axs[0,1].set_title(label = 'min_samples_split', size=30, weight='bold')
    axs[0,1].tick_params(axis='x', rotation=90, labelsize=12)
    
    sns.barplot(x='param_min_samples_leaf', y='mean_test_score', data=rs_df, ax=axs[0,2], color='lightgreen')
    axs[0,2].set_ylim([.45,.80])
    axs[0,2].set_title(label = 'min_samples_leaf', size=30, weight='bold')
    axs[0,2].tick_params(axis='x', rotation=90, labelsize=12)
    
    sns.barplot(x='param_max_features', y='mean_test_score', data=rs_df, ax=axs[1,0], color='wheat')
    axs[1,0].set_ylim([.45,.80])
    axs[1,0].set_title(label = 'max_features', size=30, weight='bold')
    
    sns.barplot(x='param_max_depth', y='mean_test_score', data=rs_df, ax=axs[1,1], color='lightpink')
    axs[1,1].set_ylim([.45,.80])
    axs[1,1].set_title(label = 'max_depth', size=30, weight='bold')
    axs[1,1].tick_params(axis='x', rotation=90, labelsize=12)
    
    sns.barplot(x='param_bootstrap',y='mean_test_score', data=rs_df, ax=axs[1,2], color='skyblue')
    axs[1,2].set_ylim([.45,.80])
    axs[1,2].set_title(label = 'bootstrap', size=30, weight='bold')
    
    plt.show()
    

def start_training(rs, X_train, y_train, show_rs_results):
    start_time = time.time()
    rs.fit(X_train, y_train)
    elapsed_time = time.time() - start_time
    print(round(elapsed_time), 's')
    
    if show_rs_results == True:
        plot_rs_results(rs)
        print(rs.best_params_)