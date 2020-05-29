#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 22:29:21 2020

@author: ruozhou_zhang
"""
import numpy as np
import os
import pandas as pd
import time
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from scipy.io import loadmat
from sklearn.ensemble import BaggingClassifier


def extract_mat(x):
    v = list(x.keys())[-1]
    return x[v]

def get_f(file_dir):
    '''Argument: 
        file_dir: The whole direction contain the exact mat file
       
       Return:
        a np.array contains the featrues of single X'''
    a = extract_mat(loadmat(file_dir))
    b = cdist(a, a)
    r = b[np.triu_indices(b.shape[1], 1)].flatten()
    return r

def feature_extraction(dir_x):
    fea_start = time.time()
    filenames = list(os.listdir(dir_x + '/'))
    filenames.sort()
    X = np.array(list(map(get_f, ((dir_x + '/' + i) for i in filenames))))
    fea_end = time.time()
    fea_time = fea_end - fea_start
    
    print('Feature Extraction Completed!')
    print(f'Feature Extraction Cost: {fea_time: 0.2f} Seconds')
    return X

def BaggingLR(X_path, y_path):
    X = feature_extraction(X_path)
    y = pd.read_csv(y_path).emotion_idx
    
    lr = LogisticRegression(C = 1,
                            penalty = 'l2',
                            fit_intercept = False)
    Bag_lr = BaggingClassifier(lr,
                               n_estimators = 70,
                               n_jobs = 5,
                               bootstrap_features = True,
                               verbose = 7)
    start = time.time()
    Bag_lr.fit(X,y)
    end = time.time()
    
    Train_time = end - start
    print(f'The Time for train is: {Train_time: 0.2f} Seconds')
    return Bag_lr
    

# =============================================================================
# Train_dir = '../data/points/'
# X = feature_extraction(Train_dir)
# y = pd.read_csv("../data/label.csv").emotion_idx
# 
# X_train, X_test, y_train, y_test = train_test_split(X, y,
#                                                     test_size = .2,
#                                                     stratify = y)
# 
# param_grid = {"C": [0.001, 0.01, 1, 5, 10, 25]}
# gs_lr = GridSearchCV(LogisticRegression(penalty = 'l2'),
#                       param_grid,
#                       verbose = 5)
# best_gs = gs_lr.fit(X_train, y_train)
# 
# lr = LogisticRegression(C = 1, penalty = 'l2')
# lr.fit(X_train, y_train)
# lr.score(X_test, y_test)
# 
# Bag_lr = BaggingClassifier(LogisticRegression(fit_intercept = False),
#                             n_estimators = 70,
#                             n_jobs = 5,
#                             bootstrap_features = True,
#                             verbose = 7)
# 
# bag_lrfit = Bag_lr.fit(X_train, y_train)
# bag_lrfit.score(X_test, y_test)
# =============================================================================

