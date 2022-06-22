# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 09:44:20 2022

@author: QUYNH ANH
"""

import logging
import numpy as np
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    
    def __init__(self, config, mode = 'train'):
        super().__init__()
        self.config = config
        self.mode = mode
        self.load_dataset(config['dataset'])
        
    def __len__(self):
        return self.rolling_windows.shape[0]
    def __getitem__(self, index):
        input = target = self.rolling_windows[index, :-1, :]
        sample = {"input": input, "target": target}
        return sample
    def load_dataset(self, dataset):
        data_dir = self.config['data_dir']
        self.data = np.load(data_dir + dataset + ".npz")
        
        if self.mode != "test":
            if len(self.data['training'].shape) == 1:
                data = np.expand_dims(self.data['training'], -1)
            else:
                data = self.data['training']
                
            if int(data.shape[0]*0.1) > self.config['l_win']:
                if self.mode == 'train':
                    data = data[: int(data.shape[0]*0.9), :]
                    logging.info("TRAINING DATA SHAPE: {}".format(data.shape))
                else:
                    data = data[int(data.shape[0]*0.9):, :]
            else:
                if self.mode == "validate":
                    self.rolling_windows = np.zero((0,0,0))
                    return logging.info("VALIDATE DATA SHAPE: {}".format(data.shape))
        else:
            if len(data['test'].shape) == 1:
                data = np.expand_dims(data['test'], -1)
            else:
                data = self.data['test']
            logging.info("TEST DATA SHAPE: {}".format(data.shape))
            
        self.rolling_windows = np.lib.stride_tricks.sliding_window_view(data, self.config['l_win'], axis = 0, writeable = True).transpose(0,2,1)


    
