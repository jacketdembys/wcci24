# Libraries
# import libraries
from mpl_toolkits.mplot3d import Axes3D

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
#import os
import sys
import wandb

from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import decomposition
from sklearn import manifold
from tqdm import tqdm
from scipy import stats
from torchviz import make_dot
from utils import *




# set parameters and configurations
robot_choice = "4DoF-2RPR"
seed = True                                                                   # seed random generators for reproducibility
visualize_joints = True                                                       # visualize joint distribution in dataset     
visualize_workspace = True                                                    # visualize workspace (positions)
visualize_losses = True                                                       # visuallze training and validation losses
visualize_normalized_workspace = True                                         # visualize normalized workspace (positions - debugging purposes)
visualize_workspace_results = True                                            # visualize results in workapce
print_inference_summary = True                                                # perform and print inference summary after training is done
print_epoch = True  
batch_size = 128                                                              # desired batch size
init_type = "default"                                                         # weights init method (default, uniform, normal, xavier_uniform, xavier_normal)
hidden_layer_sizes = [128,128,128,128]                                           # architecture to employ
learning_rate = 1e-4                                                          # learning rate
optimizer_choice = "SGD"                                                      # optimizers (SGD, Adam, Adadelta, RMSprop)
loss_choice = "l2"                                                            # l2, l1, lfk
network_type = "MLP"                                                    # MLP, ResMLP, DenseMLP, FouierMLP 
EPOCHS = 10000                                                                # total training epochs



if __name__ == '__main__':
    print("Testing the installation of all packages with docker and kubernetes")


