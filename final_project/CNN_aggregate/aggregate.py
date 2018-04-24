#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 03:17:38 2018

@author: oliviajin
"""

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 15:33:34 2018

@author: oliviajin
"""
import h5py
import numpy as np
import torch
from sklearn.decomposition import PCA
import random
from sklearn.cross_validation import train_test_split

datasets = ['project_datasets/A01T_slice.mat', 'project_datasets/A02T_slice.mat', 'project_datasets/A03T_slice.mat',
            'project_datasets/A04T_slice.mat', 'project_datasets/A05T_slice.mat', 'project_datasets/A06T_slice.mat',
            'project_datasets/A07T_slice.mat', 'project_datasets/A08T_slice.mat', 'project_datasets/A09T_slice.mat']

def random_data_process(x_data, extend):
    output = np.zeros((extend, x_data.shape[0], 300))  # index
    np.random.seed(0)
    for k in range(extend):  # index
        start_pos = random.randint(0, 699)
        end_pos = start_pos + 300
        new_data = x_data[:, start_pos:end_pos]
        output[k] = new_data
    return output


flag = True


def extend_data(file_name, extend):
    # mat_dataset = 'project_datasets/A01T_slice.mat'
    mat_dataset = file_name
    A01T = h5py.File(mat_dataset, 'r')
    X = np.copy(A01T['image'])
    y = np.copy(A01T['type'])
    y = y[0, 0:X.shape[0]:1]
    y = np.asarray(y, dtype=np.int32)

    # Exclude EOG signal
    X = X[:, :22, :]
    # X_test = X[:50, :, :]
    # X = X[50:, :, :]
    # y_test = y[:50]
    # y = y[50:]
    X, X_test, y, y_test = train_test_split(X, y, test_size=50, random_state=0)
    X, X_val, y, y_val = train_test_split(X, y, test_size=30, random_state=0)
    X_val = torch.from_numpy(X_val)
    X_test = torch.from_numpy(X_test)
    X = torch.from_numpy(X)
    # trial_num = X.shape[0]

    # Extract raw signal features and Extend data
    # train
    trial_num = X.shape[0]
    for i in range(trial_num):
        select_data = random_data_process(X[i], extend)
        if i == 0:
            data = select_data
        else:
            data = np.vstack((data, select_data))
    for i in range(extend):  # index
        if i == 0:
            label = y
        else:
            label = np.vstack((label, y))
    label = label.transpose()
    label = label.reshape(-1, 1)

    # validation
    val_num = X_val.shape[0]
    for i in range(val_num):
        select_data = random_data_process(X_val[i], extend)
        if i == 0:
            val_data = select_data
        else:
            val_data = np.vstack((val_data, select_data))
    X_val = val_data
    for i in range(extend):  # index
        if i == 0:
            val_label = y_val
        else:
            val_label = np.vstack((val_label, y_val))
    val_label = val_label.transpose()
    val_label = val_label.reshape(-1, 1)

    # test
    test_num = X_test.shape[0]
    for i in range(test_num):
        select_data = random_data_process(X_test[i], extend)
        if i == 0:
            test_data = select_data
        else:
            test_data = np.vstack((test_data, select_data))
    X_test = test_data
    for i in range(extend):  # index
        if i == 0:
            test_label = y_test
        else:
            test_label = np.vstack((test_label, y_test))
    test_label = test_label.transpose()
    test_label = test_label.reshape(-1, 1)

    # PCA decomposition
    if True:

        # PCA features
        # train
        data = data.transpose(0, 2, 1)
        feature_num = 10
        pca = PCA(n_components=feature_num)
        X_pca = np.zeros((data.shape[0], data.shape[1], feature_num))
        trial = 0
        error_num = 0
        trail_num = data.shape[0]
        train_num = []
        for i in range(trail_num):
            try:
                X_pca[trial] = pca.fit_transform(data[i])
                trial += 1
            except:
                label = np.delete(label, i - error_num)
                error_num += 1
            if i % extend == 0 and i > 0:
                train_num.append(extend - error_num)
                error_num = 0
        train_num.append(extend - error_num)
        out = X_pca[:trial, :, :]
        out = torch.from_numpy(out)
        out = out.float()

        # validation
        data = X_val
        data = data.transpose(0, 2, 1)
        feature_num = 10
        pca = PCA(n_components=feature_num)
        X_pca = np.zeros((data.shape[0], data.shape[1], feature_num))
        trial = 0
        error_num = 0
        trail_num = data.shape[0]
        val_num = []
        for i in range(trail_num):
            try:
                X_pca[trial] = pca.fit_transform(data[i])
                trial += 1
            except:
                val_label = np.delete(val_label, i - error_num)
                error_num += 1
            if i % extend == 0 and i > 0:
                val_num.append(extend - error_num)
                error_num = 0
        val_num.append(extend - error_num)
        X_val = X_pca[:trial, :, :]
        X_val = torch.from_numpy(X_val)
        X_val = X_val.float()

        # test
        data = X_test
        data = data.transpose(0, 2, 1)
        feature_num = 10
        pca = PCA(n_components=feature_num)
        X_pca = np.zeros((data.shape[0], data.shape[1], feature_num))
        trial = 0
        error_num = 0
        trail_num = data.shape[0]
        test_num = []
        for i in range(trail_num):
            try:
                X_pca[trial] = pca.fit_transform(data[i])
                trial += 1
            except:
                test_label = np.delete(test_label, i - error_num)
                error_num += 1
            if i % extend == 0 and i > 0:
                test_num.append(extend - error_num)
                error_num = 0
        test_num.append(extend - error_num)
        X_test = X_pca[:trial, :, :]
        X_test = torch.from_numpy(X_test)
        X_test = X_test.float()

    else:
        out = data
        out = out.transpose(0, 2, 1)
        out = torch.from_numpy(out)
        out = out.float()

    label = label - 769
    label = label.reshape((label.shape[0]))
    label = torch.from_numpy(label)
    label = label.long()

    label_val = val_label - 769
    label_val = label_val.reshape((label_val.shape[0]))
    label_val = torch.from_numpy(label_val)
    label_val = label_val.long()

    label_test = test_label - 769
    label_test = label_test.reshape((label_test.shape[0]))
    label_test = torch.from_numpy(label_test)
    label_test = label_test.long()


    return out, X_val, X_test, label, label_val, label_test, train_num, val_num, test_num

#if True:
#    extend = 3
def preprocess(extend):
    for i, file_name in enumerate(datasets):
        print('Loading', file_name)
        if i == 0:
            X_train, X_val, X_test, y_train, y_val, y_test, train_num, val_num, test_num = extend_data(file_name, extend)
        else:
            X1, X2, X3, y1, y2, y3, n1, n2, n3 = extend_data(file_name, extend)
            X_train = np.vstack((X_train, X1))
            X_val = np.vstack((X_val, X2))
            X_test = np.vstack((X_test, X3))
            y_train = np.hstack((y_train, y1))
            y_val = np.hstack((y_val, y2))
            y_test = np.hstack((y_test, y3))
            train_num = np.hstack((train_num, n1))
            val_num = np.hstack((val_num, n2))
            test_num = np.hstack((test_num, n3))
    
    X_train = torch.from_numpy(X_train)
    X_train = X_train.float()
    X_val = torch.from_numpy(X_val)
    X_val = X_val.float()
    X_test = torch.from_numpy(X_test)
    X_test = X_test.float()
    
    y_train = torch.from_numpy(y_train)
    y_train = y_train.long()
    y_val = torch.from_numpy(y_val)
    y_val = y_val.long()
    y_test = torch.from_numpy(y_test)
    y_test = y_test.long()
    
    train_num = train_num.tolist()
    val_num = val_num.tolist()
    test_num = test_num.tolist()
    
    print(X_train.shape)
    print(X_val.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_val.shape)
    print(y_test.shape)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, train_num, val_num, test_num
