# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 22:03:07 2022

@author: QUYNH ANH
"""
import pandas as pd
from lstm import LstmModel
from dataloader import CustomDataset
import torch
import torch.nn as nn
from scipy.stats import norm
import numpy as np
#from torch import optim 
from torch.utils.data import DataLoader
import math

config = {
    'l_win': 120,
    'n_features' : 45,
    'hidden_size' : 128,
    'num_layers' : 2,
    'batch_size' : 100,
    'lr': 0.001
    }

def return_anomaly_by_threshold(test_anomaly_metric, threshold):
    # test_list = np.squeeze(np.ndarray.flatten(test_anomaly_metric))
    idx_error = np.zeros(len(test_anomaly_metric))
    
    for i, data in enumerate(test_anomaly_metric):
        if data > threshold:
            idx_error[i] = 1
    
    return idx_error

def KQp(data, q):
    data2 = np.sort(data)  # sap xep tang dan
    n = np.shape(data2)[0]  # kich thuoc
    p = 1-q  # q tu xet, dat bang smth 0.05 0.025 0.01
    h = math.sqrt((p*q)/(n+1))
    KQ = 0
    for i in range(1, n+1):
        a = ((i/n)-p)/h
        b = (((i-1)/n)-p)/h
        TP = (norm.cdf(a)-norm.cdf(b))*data2[i-1]  # normcdf thu trong matlab
        KQ = KQ+TP
    # KQp = KQ;
    return KQ

def count_TP_FP_FN(anomaly, test_labels):
    n_TP = 0
    n_FP = 0
    n_FN = 0
    #n_detection = len(idx_detected_anomaly)
    # for i in range(n_detection):
    for i, data in enumerate(anomaly):
        # if test_labels[idx_detected_anomaly[i]] == 1:
        if test_labels[i] == 1:
            if data == 1:
                n_TP = n_TP + 1
            else:
                n_FN = n_FN + 1
        else:
            if data == 1:
                n_FP = n_FP + 1

    n_TN = len(test_labels) - n_TP - n_FP - n_FN
    return n_TP, n_FP, n_FN, n_TN


def compute_precision_and_recall(anomaly, test_labels):
    # compute true positive
    n_TP, n_FP, n_FN, n_TN = count_TP_FP_FN(anomaly, test_labels)

    if n_TP + n_FP == 0:
        precision = 1
    else:
        precision = n_TP / (n_TP + n_FP)
    recall = n_TP / (n_TP + n_FN)
    if precision + recall == 0:
        F1 = 0
    else:
        F1 = 2 * (precision * recall)/(precision + recall)
    accuracy = (n_TP + n_TN) / (n_TP + n_FN + n_FP + n_TN)

    return precision, recall, F1, accuracy, n_TP, n_FP, n_FN


def select_KQp_threshold(predic_loss, test_labels):
    q_list = [0.99, 0.9, 0.1, 0.02, 0.01, 0.0095]
    n_threshold = len(q_list)
    precision_aug = np.zeros(n_threshold)
    recall_aug = np.zeros(n_threshold)
    F1_aug = np.zeros(n_threshold)
    acc_aug = np.zeros(n_threshold)
    q_best = 0
    for i in range(n_threshold):
        q = q_list[i]
        temp_thres = KQp(predic_loss, q)
        anomaly = return_anomaly_by_threshold(predic_loss, temp_thres)
        
        precision_aug[i], recall_aug[i], F1_aug[i], acc_aug[i], _, _, _ = compute_precision_and_recall(anomaly, test_labels)

    
    
    idx_best_q = np.argmax(F1_aug)
    q_best = q_list[idx_best_q]

    # auc = plot_roc_curve(fpr_aug, recall_aug, config)
    return q_best, acc_aug[idx_best_q], precision_aug[idx_best_q], recall_aug[idx_best_q], F1_aug[idx_best_q]

@torch.no_grad()
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_set = CustomDataset(config, mode="test")
    test_dataloader = DataLoader(test_set, 
                                  batch_size = config['batch_size'], 
                                  shuffle = False, 
                                  num_workers = 0)
    model = LstmModel(config)
    model.load_state_dict(torch.load(r"C:\Users\QUYNH ANH\lstm_trained_50epoch.pt"))
    model.float()
    model.eval()
    model.to(device)
    loss = nn.MSELoss()
    n_test = len(test_set)
    pred_loss = np.zeros(n_test)
    data = np.load(r"C:\Users\QUYNH ANH\pre_processing1.npz")
    test_labels = data['test'][120:1120, -1]
    
    for i, (data, sample) in enumerate(test_dataloader):
        h0 = model.init_hidden(len(data))
        data_ = data.float().to(device)
        sample_ = sample.float().to(device)
        predict = model(data_, h0)
        print(i)
        for j in range(len(data)):
            try:
                pred_loss[i*len(data) + j] = loss(predict[j,:], sample_[j,:])
            except:
                pass
    
    q, accuracy, precision, recall, F1 = select_KQp_threshold(pred_loss, test_labels)
    
    print(accuracy, precision, recall, F1)
    
if __name__ == '__main__':
    main()
    