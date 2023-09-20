# Libraries
import torch
import numpy as np
import pandas as pd

# A matrix
def A_matrix(t,d,a,al):
    A = torch.tensor([[torch.cos(t), -torch.sin(t)*torch.cos(al),  torch.sin(t)*torch.sin(al), a*torch.cos(t)],
                      [torch.sin(t),  torch.cos(t)*torch.cos(al), -torch.cos(t)*torch.sin(al), a*torch.sin(t)],
                      [           0,               torch.sin(al),               torch.cos(al),              d],
                      [           0,                           0,                           0,              1]])
    return A