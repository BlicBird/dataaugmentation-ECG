from dataclasses import dataclass
from operator import getitem
import torch
import os
from os.path import join as opj
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.io import loadmat


class ECGDataset_fake(torch.utils.data.Dataset):
    def __init__(self, file_path,):
        super().__init__()

        self.file_path = file_path
        self.df = self.load_data()
        self.X, self.Y = self.get_tensors(self.df)
    
    def load_data(self):
        
        df_load = pd.read_pickle(self.file_path)
        return df_load
    
    
    def np21h(self, y):
        # Numpy to one-hot.
        oh = np.zeros((y.shape[0], 10))
        oh[np.arange(y.shape[0]), y.astype(int)] = 1
        return oh

    def get_tensors(self, df):
        # Create the tensors that are loaded from.
        Xall = np.stack(df['ecg'].values)
        y = df['label'].astype(int).values.T
        # yall = np.stack(df['label'].values).T

        # yall = np.nan_to_num(yall)
        # y1hot = self.np21h(yall[:,0]) + self.np21h(yall[:,1]) + self.np21h(yall[:,2])
        y1hot = self.np21h(y)

        # # Remove the first column, which represents NaNs.
        y1hot = y1hot[:,1:]

        # return Xall, y
        return Xall, y1hot
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,idx):
        return self.X[idx], self.Y[idx]