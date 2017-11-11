import h5py
import math
import torch
import torch.utils.data as data
import numpy as np
import pandas as pd

class loadTrainDataset(data.Dataset):
    def __init__(self, path):
        self.file = pd.read_csv(path)
        self.nb_samples = len(self.file['pixel0'][:])

        self.y = self.file['label']
        self.x = self.file[[i for i in self.file.columns if i != 'label']]
        self.pic = []
        for i in range(self.x.shape[0]):
            img = self.x.ix[i].values.reshape((28,28))
            img = np.multiply(img, 1.0 / 255.0)
            self.pic.append(img)

    def __getitem__(self, index):
        return self.pic[index], self.y[index]

    def __len__(self):
        return self.nb_samples
    
    
    
class loadTestDataset(data.Dataset):
    def __init__(self, path):
        self.file = pd.read_csv(path)
        self.nb_samples = len(self.file['pixel0'][:])

        self.pic = []
        for i in range(self.file.shape[0]):
            img = self.file.ix[i].values.reshape((28,28))
            img = np.multiply(img, 1.0 / 255.0)
            self.pic.append(img)

    def __getitem__(self, index):
        return self.pic[index]


    def __len__(self):
        return self.nb_samples
