# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 09:22:17 2022

@author: QUYNH ANH
"""
import torch
import torch.nn as nn

class Lstm_Model(nn.Module):
    def __init__(self, config, dropout=0.2):
        super().__init__()
        self.config = config
        self.n_features = config['n_features']
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.batch_size = config['batch_size']
        self.dropout = dropout
        
        self.lstm = nn.LSTM(
            input_size= self.n_features,
            hidden_size= self.hidden_size,
            batch_first=True,
            num_layers= self.num_layers,
            dropout=self.dropout)
        
        self.linear = nn.Linear(self.hidden_size, self.config['l_win'])
        self.sigmoid = nn.Sigmoid()
       
    def forward(self, input_data, x):
        self.lstm.flatten_parameters()
        
        output, (self.hidden, _) = self.lstm(input_data, x)
        output_linear = self.linear(self.hidden[-1])
        
        return self.sigmoid(output_linear)
        
    def init_hidden(self):
        
        hn = next(self.parameters()).data
        hidden = hn.new(self.num_layers, self.config['batch_size'], self.hidden_size).zero_()
        cell = hn.new(self.num_layers, self.config['batch_size'], self.hidden_size).zero_()
        
        return (hidden, cell)

