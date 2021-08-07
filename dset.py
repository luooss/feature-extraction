import os
import random
import torch
from torch.utils.data import Dataset
import scipy.io as sio
import numpy as np


class ArtDataset(Dataset):
    def __init__(self, feature=None, freq_band='all', choice=None):
        """Initialize dataset.

        Args:
            feature (str): None, 'psd', 'de'
            freq_band (str): 'all', 'delta', 'theta', 'alpha', 'beta', 'gamma'
            choice (list): Indices to choose
        """
        super().__init__()
        bands = ['all', 'delta', 'theta', 'alpha', 'beta', 'gamma']
        if feature == None:
            data_path = r"D:\bcmi\EMBC\eeg_process\npydata\huatong_data.npy"
            freq_in_use = bands.index(freq_band)
            self.data = np.load(data_path)[freq_in_use]
        elif feature == 'psd':
            data_path = r"D:\bcmi\EMBC\eeg_process\npydata\huatong_data_psd.npy"
            self.data = np.load(data_path)
        elif feature == 'de':
            data_path = r"D:\bcmi\EMBC\eeg_process\npydata\huatong_data_de.npy"
            self.data = np.load(data_path)
        
        label_path = r"D:\bcmi\EMBC\eeg_process\npydata\huatong_label.npy"
        self.label = np.load(label_path)

        if not choice == None:
            self.data = self.data[choice]
            self.label = self.label[choice]
        
        self.data = torch.from_numpy(self.data).to(torch.float)
        self.label = torch.from_numpy(self.label).to(torch.long)
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        return self.data[index], self.label[index]


class SEED_IV(Dataset):
    labels = [[1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3],
              [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1],
              [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]]
    freqs = {'delta':0,
             'theta':1,
             'alpha':2,
             'beta':3,
             'gamma':4}

    def __init__(self, dir_path, feature, smooth_method, freq_bands, subjects, DANN=None):
        """Initialize dataset.

        Args:
            dir_path (str): directory path
            feature (str): either 'de' or 'psd'
            smooth_method (str): either 'movingAve' or 'LDS'
            freq_bands (array): delta, theta, alpha, beta, gamma
            subjects (array): 1-15
        """
        super().__init__()
        self.DANN = DANN
        data_list = []
        label_list = []
        self.fbs = [self.freqs[band] for band in freq_bands]
        fpattern = feature + '_' + smooth_method
        for session in range(0, 3):
            for fname in os.listdir(os.path.join(dir_path, str(session+1))):
                if int(fname[:-4]) in subjects:
                    datap = sio.loadmat(os.path.join(dir_path, str(session+1), fname))
                    for k, v in datap.items():
                        if k.startswith(fpattern):
                            trial_number = int(k[len(fpattern):]) - 1
                            x = torch.from_numpy(v)[:, :, self.fbs].permute(1, 0, 2).flatten(start_dim=1)
                            data_list.append(x)
                            label_list += [self.labels[session][trial_number]] * x.size()[0]
        
        self.data = torch.cat(data_list, 0).to(torch.float)
        self.emotion_label = torch.tensor(label_list).to(torch.long)
        if self.DANN == 'source':
            self.domain_label = torch.zeros(self.data.size()[0]).to(torch.long)
        elif self.DANN == 'target':
            self.domain_label = torch.ones(self.data.size()[0]).to(torch.long)

        print('SEED-IV dataset preparation done:\n{}, {}, {}, {}'.format(fpattern, str(freq_bands), str(subjects), str(self.data.size())))
        unique_label, label_count = torch.unique(self.emotion_label, sorted=True, return_counts=True)
        label_count = torch.true_divide(label_count, label_count.sum())
        print('Label count: {}, {}'.format(str(unique_label), str(label_count)))
    
    def __len__(self):
        return self.data.size()[0]
    
    def __getitem__(self, index):
        if self.DANN is None:
            return self.data[index], self.emotion_label[index]
        else:
            return self.data[index], self.emotion_label[index], self.domain_label[index]
