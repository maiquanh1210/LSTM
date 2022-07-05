# -*- coding: utf-8 -*-


import pandas as pd
from lstm import Lstm_Model
from train import Trainer
from dataloader import CustomDataset
import torch
#import torch.nn as nn
import pandas as pd
import numpy as np
#from torch import optim 
from torch.utils.data import DataLoader
import time
config = {
    'l_win': 120,
    'n_features' : 45,
    'hidden_size' : 128,
    'num_layers' : 2,
    'batch_size' : 100,
    'lr': 0.001
    }
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_set = CustomDataset(config=config)
    
    len_data = int((len(train_set)/config['batch_size']))*config['batch_size']
    train_dataloader = DataLoader(train_set[:len_data], 
                                  batch_size = config['batch_size'], 
                                  shuffle = True, 
                                  num_workers = 0)
    config['epoch_n'] = len(train_dataloader)
    model = Lstm_Model(config)
    model.float()
    model.to(device)
    optimize = torch.optim
    model_trainer = Trainer(model, device, optimize, config)
    print("ok")
    model_trainer.train(train_dataloader)
    test_data = CustomDataset(config, mode = 'test')
    test_dataloader = DataLoader(test_data[:len_data],
                                 batch_size = config['batch_size'],
                                 shuffle=False,
                                 num_workers=0)
    output, label = model_trainer.test(test_dataloader)
    
    df_loss = pd.DataFrame({'output': output, 'label': label})
    df_loss.to_csv("train_loss1.csv", mode = 'a', index = False)
    
    
    
    
    
    
    
    


