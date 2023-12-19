from turtle import mode
from urllib import robotparser
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np



class CustomDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = {
            'data': torch.FloatTensor(self.data[idx]),
            'targets': torch.FloatTensor([self.targets[idx]])
        }
        return sample
    


def data_loader(robot_choice, mode_choice, test_size, batch_size, shuffle=False):

    if robot_choice == "3DoF-3R":

        if mode_choice == "FK":

            data = pd.read_csv('../ea-based-nn-ik-solver/data_3DoF-3R_N.csv')
            data = np.array(data).astype(np.float32)
            joints = data[:,-3:]
            pos = data[:,:-3]

            FK_X_train, FK_X_test, FK_y_train, FK_y_test = train_test_split(joints, pos, test_size=test_size, random_state=42)

            FK_train_dataset = CustomDataset(FK_X_train, FK_y_train)
            FK_train_loader = DataLoader(FK_train_dataset, batch_size=batch_size, shuffle=shuffle)

            FK_test_dataset = CustomDataset(FK_X_test, FK_y_test)
            FK_test_loader = DataLoader(FK_test_dataset, batch_size=batch_size, shuffle=shuffle)

            return FK_train_loader, FK_test_loader, joints.shape[1], pos.shape[1]
        
        if mode_choice == "IK":

            data = pd.read_csv('../ea-based-nn-ik-solver/data_3DoF-3R_N.csv')
            data = np.array(data).astype(np.float32)
            joints = data[:,-3:]
            pos = data[:,:-3]

            IK_X_train, IK_X_test, IK_y_train, IK_y_test = train_test_split(pos, joints, test_size=test_size, random_state=42)

            IK_train_dataset = CustomDataset(IK_X_train, IK_y_train)
            IK_train_loader = DataLoader(IK_train_dataset, batch_size=batch_size, shuffle=shuffle)

            IK_test_dataset = CustomDataset(IK_X_test, IK_y_test)
            IK_test_loader = DataLoader(IK_test_dataset, batch_size=batch_size, shuffle=shuffle)

            return IK_train_loader, IK_test_loader, pos.shape[1], joints.shape[1]
        
        if mode_choice == "IKFK":

            data = pd.read_csv('../ea-based-nn-ik-solver/data_3DoF-3R_N.csv')
            data = np.array(data).astype(np.float32)
            joints = data[:,-3:]
            pos = data[:,:-3]

            IKFK_X_train, IKFK_X_test, IKFK_y_train, IKFK_y_test = train_test_split(pos, pos, test_size=test_size, random_state=42)

            IKFK_train_dataset = CustomDataset(IKFK_X_train, IKFK_y_train)
            IKFK_train_loader = DataLoader(IKFK_train_dataset, batch_size=batch_size, shuffle=shuffle)

            IKFK_test_dataset = CustomDataset(IKFK_X_test, IKFK_y_test)
            IKFK_test_loader = DataLoader(IKFK_test_dataset, batch_size=batch_size, shuffle=shuffle)

            return IKFK_train_loader, IKFK_test_loader, pos.shape[1], joints.shape[1]


