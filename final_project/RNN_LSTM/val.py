#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 21:03:14 2018

@author: oliviajin
"""

import h5py
import numpy as np
import torch
from sklearn.decomposition import PCA
import pywt
from sklearn.cross_validation import train_test_split


def data_filter(X):
    ca3, cd3, cd2, cd1 = pywt.wavedec(X, 'coif1', level=3)
    cd3 = np.zeros((cd3.shape[0]))
    #ca, cd = pywt.dwt(X,'coif1')
    #ca2, cd2, cd1 = pywt.wavedec(X, 'coif1', level=2)
    #cd2 = np.zeros((cd2.shape[0]))
    out = pywt.idwt(ca3, cd3, 'coif1')
    return out


def pca(X, y, feature_num, index):
    data = X.transpose(0, 2, 1)
    pca = PCA(n_components=feature_num)
    X_pca = np.zeros((data.shape[0], data.shape[1], feature_num))
    trial = 0
    error_num = 0
    trail_num = data.shape[0]
    for i in range(trail_num):
        try:
            X_pca[trial] = pca.fit_transform(data[i])
            trial += 1
        except:
            y = np.delete(y, i - error_num)
            error_num += 1
    out = X_pca[:trial, :, :]
    out = torch.from_numpy(out)
    out = out.float()
    
    y = y - 769
    y = torch.from_numpy(y)
    y = y.long()
    return out, y


def preprocess(file_name, index):
# if True:
    mat_dataset = file_name
    # mat_dataset = 'project_datasets3/A01T_slice.mat'
    A01T = h5py.File(mat_dataset, 'r')
    X = np.copy(A01T['image'])
    y = np.copy(A01T['type'])
    y = y[0, 0:X.shape[0]:1]
    y = np.asarray(y, dtype=np.int32)
    
    X  = X[:, :22, :]
    X, X_t, y, y_t = train_test_split(X, y, test_size=50, random_state=0)
    X, X_v, y, y_v = train_test_split(X, y, test_size=20, random_state=0)

    # Extract raw signal features and Extend data
    X_train = np.zeros((X.shape[0], X.shape[1], 254))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X_train[i][j] = data_filter(X[i][j])

    X_val = np.zeros((X_v.shape[0], X_v.shape[1], 254))
    for i in range(X_v.shape[0]):
        for j in range(X_v.shape[1]):
            X_val[i][j] = data_filter(X_t[i][j])

    X_test = np.zeros((X_t.shape[0], X_t.shape[1], 254))
    for i in range(X_t.shape[0]):
        for j in range(X_t.shape[1]):
            X_test[i][j] = data_filter(X_t[i][j])

    # PCA decomposition
    X_train, y_train = pca(X_train, y, 10, index)
    X_val, y_val = pca(X_val, y_v, 10, index)
    X_test, y_test = pca(X_test, y_t, 10, index)
    
    print(X_train.shape)
    print(y_train.shape)
    print(X_val.shape)
    print(y_val.shape)
    print(X_test.shape)
    print(y_test.shape)

    return X_train, y_train, X_val, y_val, X_test, y_test
