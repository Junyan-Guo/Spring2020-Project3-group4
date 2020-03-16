#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 19:14:39 2020

@author: ruozhou_zhang
"""

import numpy as np
import os
import pandas as pd
import time
from scipy.spatial.distance import cdist
from sklearn.svm import SVC
from scipy.io import loadmat
# from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import BaggingClassifier


"""
    To Use This you should Make sure that:
        1. For Train_set: 
            1) label.csv should have a column named emotion_idx contains y_train
            2) Should have a file named points that contains the .mat file contains image point coordinates
            3) label.csv and file points should be under the same path
        2. For Test_set:
            1) Should have a file named points that contains the .mat file contains image point coordinates
"""


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

def BaggingSVM(X_path, y_path):
    X = feature_extraction(X_path)
    y = pd.read_csv(y_path).emotion_idx
    
    start_SVM = time.time()
    S_svm = SVC(C = 0.1,
                kernel = 'linear',
                shrinking = True,
                decision_function_shape = 'ovo')
    Bagg_SVM = BaggingClassifier(S_svm,
                                 n_estimators = 80,
                                 n_jobs = 5,
                                 bootstrap_features = True)
    Bagg_SVM.fit(X,y)
    end_SVM = time.time()
    
    Train_time = end_SVM - start_SVM
    print(f'The Time for train is: {Train_time: 0.2f} Seconds')
    return Bagg_SVM


def Test_on(test_model, Test_set_dir):
    X = feature_extraction(Test_set_dir)
    y_hat = test_model.predict(X)
    return y_hat


TheModel = BaggingSVM('../data/points', '../data/label.csv')
y_hat = Test_on(TheModel, '../data/points')


# =============================================================================
# 
# Train_dir = '../data/points/'
# X = feature_extraction(Train_dir)
# y = pd.read_csv("../data/label.csv").emotion_idx
# groups = y
# 
# X_train, X_test, y_train, y_test = train_test_split(X, y,
#                                                     test_size = .2,
#                                                     stratify = y)
# 
# # My single SVM model
# start_SVM = time.time()
# S_svm = SVC(C = 0.1,
#             kernel = 'linear',
#             shrinking = True,
#             decision_function_shape = 'ovo')
# S_svm.fit(X_train, y_train)
# end_SVM = time.time()
# Single_SVM_Train_Time = end_SVM - start_SVM
# Single_SVM_Train_Error = S_svm.score(X_train, y_train)
# Single_SVM_Test_Error = S_svm.score(X_test, y_test)
# 
# S_svm_cv = cross_validate(S_svm,
#                           X, 
#                           y,
#                           verbose = 5,
#                           cv = 5,
#                           groups = groups,
#                           return_train_score = True)
# 
# 
# # My Bagging SVM model
# B_svm = SVC(C = .00001,
#             kernel = 'linear', shrinking = True,
#             decision_function_shape = 'ovo')
# Bagg_SVM = BaggingClassifier(B_svm,
#                              n_estimators = 80,
#                              n_jobs = 5,
#                              bootstrap_features = True)
# Bagg_SVM.fit(X_train, y_train)
# Bagg_SVM.score(X_test, y_test)
# 
# 
# =============================================================================
