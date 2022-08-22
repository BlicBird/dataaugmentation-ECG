### SEGMENTATION ###


from dataclasses import dataclass
from operator import getitem
import torch
import os
from os.path import join as opj
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.io import loadmat


class ECGDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, segment_length):
        super().__init__()

        self.file_path = file_path
        self.segment_length = segment_length
        self.df = self.load_data()
        self.df['size'] = self.df['ecg'].apply(lambda x : np.array(x).shape[1])
        self.df = self.df[self.df['size'] > self.segment_length]
        self.df['segmented_ecg'] = self.df['ecg'].apply(lambda x: self.split(x,self.segment_length))
        self.df = self.df.explode('segmented_ecg').reset_index()
        self.X, self.Y = self.get_tensors(self.df)
    
    def load_data(self):
        files = [opj(self.file_path, x) for x in os.listdir(self.file_path) if '.mat' in x]
        
        #Create dataframe
        df = pd.DataFrame({thing:[] for thing in ['name', 'sex', 'ecg']})
        
        for thing in ['name', 'sex']:
            df[thing] = df[thing].astype(str)
    
        for thing in ['ecg']:
            df[thing] = df[thing].astype(object)
        
        # Reference file containing labels
        ref = pd.read_csv(opj(self.file_path, 'REFERENCE.csv')).rename(columns={
            'First_label':'label0',
            'Second_label':'label1',
            'Third_label':'label2'
        })
        
        for i, f in tqdm(enumerate(files), total=len(files)):
            name = f[:f.index('.')].rsplit('/')[-1]
            data = loadmat(f)
            ecg = data['ECG'][0][0][2].astype('float32')
            sex = data['ECG'][0][0][0][0].lower()
            other = data['ECG'][0][0][1]
    
            assert type(ecg)==np.ndarray
            assert ecg.shape[0]==12
            assert sex in ['male', 'female']
    
            df.at[i, 'name'] = name
            df.at[i, 'sex'] = sex
            df.at[i, 'ecg'] = ecg
        
        df = pd.merge(left=df, right=ref, left_on='name', right_on='Recording', how='inner').drop(columns='Recording')

        return df
    
    def split(self,array, segment_size):
        rmv = len(array[0]) % segment_size
        times = (len(array[0]) - rmv) / segment_size
        if rmv != 0:
            array = array[:,:-rmv]
        return np.hsplit(array,times)
    
    def np21h(self, y):
        # Numpy to one-hot.
        oh = np.zeros((y.shape[0], 10))
        oh[np.arange(y.shape[0]), y.astype(int)] = 1
        return oh

    def get_tensors(self, df):
        # Create the tensors that are loaded from.
        # Create the tensors that are loaded from.
        Xall = np.stack(df['segmented_ecg'].values)
        # y = df['label0'].astype(int).values
        yall = np.stack([
            df['label0'].values,
            df['label1'].values,
            df['label2'].values
            ]).T

        yall = np.nan_to_num(yall)
        y1hot = self.np21h(yall[:,0]) + self.np21h(yall[:,1]) + self.np21h(yall[:,2])

        # Remove the first column, which represents NaNs.
        y1hot = y1hot[:,1:]

        return Xall, y1hot
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,idx):
        return self.X[idx], self.Y[idx]



