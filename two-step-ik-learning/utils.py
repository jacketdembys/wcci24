import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import pandas as pd
import random
import sklearn
import time
import math
import matplotlib.pyplot as plt
import os
import sys
import argparse

from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import decomposition
from sklearn import manifold
from tqdm import tqdm
from scipy import stats


def get_DH(robot_choice, t):
    # columns: t, d, a, alpha
    if robot_choice == "2DoF-2R":
        DH = torch.tensor([[t[0], 0, 1, 0],
                           [t[1], 0, 1, 0]])
    elif robot_choice == "3DoF-3R":
        DH = torch.tensor([[t[0], 0, 1, 0],
                           [t[1], 0, 1, torch.pi/2],
                           [t[2], 0, 0, 0]])
    elif robot_choice == "3DoF-3R-2":
        DH = torch.tensor([[t[0], 380/1000, 0, 0],
                           [t[1],        0, 280/1000, -torch.pi/2],
                           [t[2],        0, 280/1000, 0]])
    elif robot_choice == "4DoF-2RPR":
        DH = torch.tensor([[t[0], 400/1000, 250/1000, 0],
                           [t[1],        0, 150/1000, torch.pi],
                           [   0,     t[2],        0, 0],
                           [t[3], 150/1000,        0, 0]])
    elif robot_choice == "6DoF-6R-Puma260":
        DH = torch.tensor([[t[0],           0,          0,        -torch.pi/2],
                           [t[1],  125.4/1000, 203.2/1000,                  0],
                           [t[2],           0,  -7.9/1000,         torch.pi/2],
                           [t[3],  203.2/1000,          0,        -torch.pi/2],
                           [t[4],           0,          0,         torch.pi/2],
                           [t[5],   63.5/1000,          0,                  0]])
    elif robot_choice == "7DoF-7R":
        DH = torch.tensor([[t[0], 400/1000, 250/1000, 0],
                           [t[1],        0, 150/1000, torch.pi],
                           [   0,     t[2],        0, 0],
                           [t[3], 150/1000,        0, 0]])
    return DH


def A_matrix(t,d,a,al):
    # the inputs of torch.sin and torch.cos are expressed in rad
    A = torch.tensor([[torch.cos(t), -torch.sin(t)*torch.cos(al),  torch.sin(t)*torch.sin(al), a*torch.cos(t)],
                      [torch.sin(t),  torch.cos(t)*torch.cos(al), -torch.cos(t)*torch.sin(al), a*torch.sin(t)],
                      [           0,               torch.sin(al),               torch.cos(al),              d],
                      [           0,                           0,                           0,              1]])
    return A


def forward_kinematics(DH):

    n_DoF = DH.shape[0]
    T = torch.eye(4,4)
    for i in range(n_DoF):
        A = A_matrix(*DH[i,:])
        T = torch.matmul(T, A)
    
    return T


def reconstruct_pose(y_preds, robot_choice):
    # y_preds = torch.from_numpy(y_preds)
    n_samples = y_preds.shape[0]
    # print(n_samples)
    pose = []
    for i in range(n_samples):
        t = y_preds[i,:]
        DH = get_DH(robot_choice, t)
        T = forward_kinematics(DH)
        if robot_choice == "3DoF-3R":
            # x,y,t1,t2,t3 where x,y (m) and t (rad)
            pose.append(T[:2,-1].numpy())
          
    X_pred = np.array(pose)
    # print(X_pred.shape)
    return torch.from_numpy(X_pred)