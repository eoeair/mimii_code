import torch
import pickle
import numpy as np

class Us_Feeder_snr(torch.utils.data.Dataset):
    """ Feeder for label inputs """

    def __init__(self, data_path):
        self.data_path = data_path
        self.load_data()
    
    def load_data(self):
        with open(self.data_path, 'rb') as file:
            data = pickle.load(file)
        
        self.data = []

        for i,snr in enumerate(data):
            for j in snr:
                self.data.append(j)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index][0]

class Us_Feeder_device(torch.utils.data.Dataset):
    """ Feeder for label inputs """

    def __init__(self, data_path, snr):
        self.data_path = data_path
        self.snr = snr
        self.load_data()
    
    def load_data(self):
        with open(self.data_path, 'rb') as file:
            self.data = pickle.load(file)
        
    def __len__(self):
        return len(self.data[self.snr])

    def __getitem__(self, index):
        # get data
        return self.data[self.snr][index][0]

class Us_Feeder_label(torch.utils.data.Dataset):
    def __init__(self, data_path, snr):
        self.data_path = data_path
        self.snr = snr
        self.load_data()
    
    def load_data(self):
        with open(self.data_path, 'rb') as file:
            self.data = pickle.load(file)
        
    def __len__(self):
        return len(self.data[self.snr])

    def __getitem__(self, index):
        return self.data[self.snr][index][0] 
    
class Feeder_snr(torch.utils.data.Dataset):
    """ Feeder for label inputs """

    def __init__(self, data_path):
        self.data_path = data_path
        self.load_data()
    
    def load_data(self):
        # data: mfcc snr device label
        with open(self.data_path, 'rb') as file:
            data = pickle.load(file)
        
        self.data = []
        self.label = []

        for i,snr in enumerate(data):
            for j in snr:
                self.data.append(j)
                self.label.append(i)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # get data
        return self.data[index][0], self.label[index]

class Feeder_device(torch.utils.data.Dataset):
    """ Feeder for label inputs """

    def __init__(self, data_path, snr):
        self.data_path = data_path
        self.snr = snr
        self.load_data()
    
    def load_data(self):
        # data: mfcc snr device label
        with open(self.data_path, 'rb') as file:
            self.data = pickle.load(file)
        
    def __len__(self):
        return len(self.data[self.snr])

    def __getitem__(self, index):
        # get data
        return self.data[self.snr][index][0], self.data[self.snr][index][2]

class Feeder_label(torch.utils.data.Dataset):
    """ Feeder for label inputs """

    def __init__(self, data_path, snr):
        self.data_path = data_path
        self.snr = snr
        self.load_data()
    
    def load_data(self):
        # data: mfcc snr device label
        with open(self.data_path, 'rb') as file:
            self.data = pickle.load(file)
        
    def __len__(self):
        return len(self.data[self.snr])

    def __getitem__(self, index):
        # get data
        return self.data[self.snr][index][0], self.data[self.snr][index][2]