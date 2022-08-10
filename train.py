# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 15:48:51 2022

@author: QUYNH ANH
"""

import time
#import logging
import pandas as pd
import torch
from torch import nn 
from torch import optim
import pickle 


is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
class Trainer():
    def __init__(self, model, device, train_data, optimize, config, mode='train'):
        self.model = model
        self.train_data = train_data
        self.device = device
        self.config = config
        self.optim = optimize
        self.batch_size = config['batch_size']
        self.min_loss = float('inf')
        self.train_loss_list = []
        self.best_optimizer = None
        
    def train_epoch(self, criterion, optim, epoch_idx):
        
        self.model.train()
        train_loss = 0.0
        
        for i, (data, sample) in enumerate(self.train_data):
            print(i)
            self.model.zero_grad()
            x = self.model.init_hidden(len(data))
            out = self.model(data.to(device).float(), x)
            loss = criterion(out, sample.to(device).float())
            loss.backward()
            optim.step()
            train_loss += loss.item()
            
        train_loss = train_loss/len(self.train_data)
        self.train_loss_list.append(train_loss)
        print("Epoch {} done, Train loss: {}".format(epoch_idx, train_loss))
        return self.model
            
    def train(self):
        self.model.to(self.device)
        start = time.perf_counter()
        print("Start training")
        model_opt = self.optim.Adam(self.model.parameters(), lr = self.config['lr'])
        criterion = nn.MSELoss()
        
        for epoch in range(0, self.config["epoch_n"]):
            print("Training epoch {}".format(epoch + 1))
            self.train_epoch(criterion, model_opt, epoch)
            
        df_loss = pd.DataFrame(self.train_loss_list)
        df_loss.to_csv("train_loss_50epoch.csv", mode = 'a', index = False)
        print("time training: {}".format(time.perf_counter() - start))
        torch.save(self.model.state_dict(), 'lstm_trained_50epoch.pt')
