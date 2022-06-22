# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 09:36:45 2022

@author: QUYNH ANH
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

#read file raw data

data_train = pd.read_csv(r"C:\Users\QUYNH ANH\Desktop\LAB\Datasets\ICS-datasets\SWaT.A1.2\raw\SWaT_Dataset_Normal_v0.csv", header = 0, index_col = None)
data_test = pd.read_csv(r"C:\Users\QUYNH ANH\Desktop\LAB\Datasets\ICS-datasets\SWaT.A1.2\raw\SWaT_Dataset_Attack_v0.csv", header = 0, index_col = None)

col_remove = ['Timestamp', 'P202', 'P401', 'P404', 'P502', 'P601', 'P603']

#drop columns

data_train = data_train.drop(columns = col_remove, axis = 1)
data_test = data_test.drop(columns = col_remove, axis = 1)

#nomarlize

data_test.replace('A ttack', 'Attack', inplace = True)
data_test.replace(['Normal','Attack'],[0, 1], inplace= True)
data_train.replace(['Normal','Attack'],[0, 1], inplace= True)

scaler = MinMaxScaler()
#train
data_train_ = scaler.fit_transform(data_train.iloc[:, :-1])
#test
data_test_ = scaler.transform(data_test.iloc[: , :-1])
index_anomaly = np.array(data_test.index[data_test['Normal/Attack'] == 1].tolist())

#save npz flie

np.savez('pre_processing.npz', training = data_train_, test = data_test_, idx_anomaly = index_anomaly)