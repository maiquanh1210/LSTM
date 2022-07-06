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
        return self.rolling_windows[index]
    def load_dataset(self):
        #data_dir = self.config['data_dir']
        self.data = np.load(r"C:\Users\QUYNH ANH\pre_processing1.npz")
        
        if self.mode != "test":
            data = self.data['training'][:10000]
        else:
            data = self.data['test'][:200]
        self.rolling_windows = np.lib.stride_tricks.sliding_window_view(data, self.config['l_win'], axis = 0, writeable = True).transpose(0,2,1)
