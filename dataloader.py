# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 09:44:20 2022

@author: QUYNH ANH
"""

import numpy as np
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, config, mode = 'train'):
        super().__init__()
        self.config = config
        self.mode = mode
        self.load_dataset()
        
    def __len__(self):
        return self.rolling_windows.shape[0]
    def __getitem__(self, index):
        return self.rolling_windows[index], self.sample[index]
    def load_dataset(self):
        #data_dir = self.config['data_dir']
        self.data = np.load(r"C:\Users\QUYNH ANH\pre_processing1.npz")
        
        if self.mode != "test":
            data = self.data['training'][:10000,: -1]
            self.sample = self.data['training'][self.config['l_win']: 10000 + self.config['l_win'],:-1]
            
        else:
            data = self.data['test'][:1000,: -1]
            self.sample = self.data['test'][120:1120,:-1]
        
        self.rolling_windows = np.lib.stride_tricks.sliding_window_view(data, self.config['l_win'], axis = 0, writeable = True).transpose(0,2,1)
        
