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
    def __init__(self, model, device, optimize, config, mode='train'):
        self.model = model
        #self.train_data = train_data
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
        x = self.model.init_hidden()
        for i, data in enumerate(self.train_data):
            print(i)
            label = data[:,:,-1]
            self.model.zero_grad()
            
            out = self.model(data[:,:,:-1].to(device).float(), x)
            loss = criterion(out, label.to(device).float())
            loss.backward()
            optim.step()
            train_loss += loss.item()
            
        train_loss = train_loss/len(self.train_data)
        self.train_loss_list.append(train_loss)
        print("Epoch {} done, Train loss: {}".format(epoch_idx, train_loss))
        return self.model
            
    def train(self, train_data):
        self.train_data = train_data
        self.model.to(self.device)
        start = time.perf_counter()
        print("Start training")
        model_opt = self.optim.Adam(self.model.parameters(), lr = self.config['lr'])
        criterion = nn.MSELoss()
        for epoch in range(0, self.config["epoch_n"]):
            print("Training epoch {}".format(epoch + 1))
            self.train_epoch(criterion, model_opt, epoch + 1)
            
        df_loss = pd.DataFrame(self.train_loss_list)
        df_loss.to_csv("train_loss1.csv", mode = 'a', index = False)
        print("time training: {}".format(time.perf_counter() - start))
        torch.save(self.model.state_dict(), 'lstm_trained.pt')
       # return self.model
    
    def test(self, test_data):
        model = self.model
        model.eval()
        model.to(self.device)
        output = []
        sample = []
        criterion = nn.CrossEntropyLoss().to(self.device)
        with torch.no_grad():
            for data in test_data:
                data_ = data[:-1].to(self.device).float()
                target = data[-1].to(self.device).float()
                x = model.init_hidden()
                pred = model(data_.to(device).float(), x)
                output.append(pred)
                sample.append(test_data[:,:, -1])
        return output, sample
                
                
                
        
        
                
            
    
        
        