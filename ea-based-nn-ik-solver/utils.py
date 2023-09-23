import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# get DH matrix based on robot choice
def get_DH(robot_choice, t):
    # columns: t, d, a, alpha
    if robot_choice == "3DoF-3R":
        DH = torch.tensor([[t[0], 0, 1, 0],
                           [t[1], 0, 1, torch.pi/2],
                           [t[2], 0, 0, 0]])
    elif robot_choice == "4DoF-2RPR":
        DH = torch.tensor([[t[0], 400/1000, 250/1000, 0],
                           [t[1],        0, 150/1000, torch.pi],
                           [   0,     t[2],        0, 0],
                           [t[3], 150/1000,        0, 0]])
    return DH

# A matrix
def A_matrix(t,d,a,al):
    # the inputs of torch.sin and torch.cos are expressed in rad
    A = torch.tensor([[torch.cos(t), -torch.sin(t)*torch.cos(al),  torch.sin(t)*torch.sin(al), a*torch.cos(t)],
                      [torch.sin(t),  torch.cos(t)*torch.cos(al), -torch.cos(t)*torch.sin(al), a*torch.sin(t)],
                      [           0,               torch.sin(al),               torch.cos(al),              d],
                      [           0,                           0,                           0,              1]])
    return A

# Forward Kinematics
def forward_kinematics(DH):

    n_DoF = DH.shape[0]
    T = torch.eye(4,4)
    for i in range(n_DoF):
        A = A_matrix(*DH[i,:])
        T = torch.matmul(T, A)
    
    return T

# weight initialization
def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0/np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)


def weights_init_normal_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0/np.sqrt(n)
        m.weight.data.normal_(0.0,y)
        m.bias.data.fill_(0)


def weights_init_xavier_uniform_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0/np.sqrt(n)        
        nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
        m.bias.data.fill_(0)

def weights_init_xavier_normal_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0/np.sqrt(n)
        nn.init.xavier_normal_(m.weight.data)
        m.bias.data.fill_(0)


def weights_init_xavier_kaiming_uniform_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0/np.sqrt(n)
        nn.init.kaiming_uniform_(m.weight.data)
        m.bias.data.fill_(0)

def weights_init_xavier_kaiming_normal_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        n = m.in_features
        y = 1.0/np.sqrt(n)
        nn.init.kaiming_normal_(m.weight.data)
        m.bias.data.fill_(0)