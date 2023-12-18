# import libraries
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
import matplotlib.pyplot as plt
import os
import sys
import pygad
import pygad.torchga as torchga
import wandb
import argparse

from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import decomposition
from sklearn import manifold
from tqdm import tqdm
from utils import *

parser = argparse.ArgumentParser(description="A script with argparse options")


parser.add_argument("--runname", type=str, required=True)
parser.add_argument("--projectname", type=str, required=True)
parser.add_argument("--robotchoice", type=str, required=False)
parser.add_argument("--numgeneration", type=int, default=10)
parser.add_argument("--batchsize", type=int, default=4)
parser.add_argument("--savingstep", type=int, default=10)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--seed", help="Enable verbose mode", action="store_true")
parser.add_argument("--numsolutions", type=int, default=100)
parser.add_argument("--numparents", type=int, default=5)

args = parser.parse_args()

arg_robotchoice = args.robotchoice
arg_numgeneration = args.numgeneration
arg_runname = args.runname
arg_projectname = args.projectname
arg_savingstep = args.savingstep
arg_batchsize = args.batchsize
arg_numsolutions = args.numsolutions
arg_numparents = args.numparents

wandb.init(
        # set the wandb project where this run will be logged
        project=arg_projectname, name=arg_runname
        
        # track hyperparameters and run metadata
    #     config={
    #     "learning_rate": 0.02,
    #     "architecture": "CNN",
    #     "dataset": "CIFAR-100",
    #     "epochs": 20,
    #     }
    )

if args.seed:
    SEED = 3
    
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True





robot_choice = "3DoF-3R"
if robot_choice == "3DoF-3R":
    n_DoF = 3
    input_dim = 2
    output_dim = 3
data = pd.read_csv('./data_3DoF-3R_N.csv')



def load_dataset(data, n_DoF, batch_size=1):
    
    X = data[:,:2]
    y = data[:,2:]

        
    #y = data[:,:2]
    #X = data[:,2:]
        
    # split in train and test sets
    X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(X, 
                                                                y, 
                                                                test_size = 0.1,
                                                                random_state = 1)

    sc_in = MinMaxScaler(copy=True, feature_range=(-1, 1))
    sc_out = MinMaxScaler(copy=True, feature_range=(-1, 1))
    
    X_train = sc_in.fit_transform(X_train_i)
    X_test = sc_in.transform(X_test_i)  

    #y_train = sc_out.fit_transform(y_train)
    #y_test = sc_out.transform(y_test) 

    print(X_train.shape)
    print(y_train_i.shape)

    train_data = LoadIKDataset(X_train, y_train_i)
    test_data = LoadIKDataset(X_test, y_test_i)

    train_data_loader = DataLoader(dataset=train_data,
                                   batch_size=batch_size,
                                   shuffle=False,
                                   drop_last=False)

    test_data_loader = DataLoader(dataset=test_data,
                                   batch_size=1,
                                   shuffle=False)

    return train_data_loader, test_data_loader, X_test_i, y_test_i, X_train, y_train_i


def reconstruct_pose(y_preds, robot_choice):
    y_preds = torch.from_numpy(y_preds)
    n_samples = y_preds.shape[0]
    pose = []
    for i in range(n_samples):
        t = y_preds[i,:]
        DH = get_DH(robot_choice, t)
        T = forward_kinematics(DH)
        pose.append(T[:2,-1].numpy())
          
    X_pred = np.array(pose)
    return X_pred

class LoadIKDataset(Dataset):
    def __init__(self, inputs_array, outputs_array):
        x_temp = inputs_array
        y_temp = outputs_array

        self.x_data = torch.tensor(x_temp, dtype=torch.float32) 
        self.y_data = torch.tensor(y_temp, dtype=torch.float32) 

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        current_input = self.x_data[idx, :]
        current_output = self.y_data[idx, :]
        
        sample = {'input': current_input,
                  'output': current_output}
        return sample
    

    def __len__(self):
        return len(self.x_data)


data_a = np.array(data) 
train_data_loader, test_data_loader, X_test, y_test, X_train, y_train = load_dataset(data_a, 
                                                                                     n_DoF,
                                                                                     batch_size=arg_batchsize)


def fitness_func(ga_instance, solution, sol_idx):
    global data_inputs, data_outputs, torch_ga, model, loss_function
    data_inputs = data_inputs.float()

    model_weights_dict = torchga.model_weights_as_dict(model=model,
                                                       weights_vector=solution)

    # Use the current solution as the model parameters.
    model.load_state_dict(model_weights_dict)

    predictions = model(data_inputs)
    X_pred = reconstruct_pose(predictions.detach().numpy(), robot_choice)
    data_outputs = data_outputs.float()
    X_pred = torch.from_numpy(X_pred)
    # print(X_pred.shape)
#     abs_error = loss_function(predictions, data_outputs).detach().numpy() + 0.00000001
    abs_error = loss_function(X_pred, data_inputs).detach().numpy()

#     solution_fitness = 1.0 / abs_error
    solution_fitness = -abs_error

    return solution_fitness

def callback_generation(ga_instance):
    global test_data_inputs, test_data_outputs, torch_ga, model, loss_function, test_error_function

    test_data_inputs = test_data_inputs.float()

    model_weights_dict = torchga.model_weights_as_dict(model=model,
                                                       weights_vector=ga_instance.best_solution()[0])
    
    model.load_state_dict(model_weights_dict)

    predictions = model(test_data_inputs)
    X_pred = reconstruct_pose(predictions.detach().numpy(), robot_choice)
    test_data_outputs = test_data_outputs.float()
    X_pred = torch.from_numpy(X_pred)

    test_error = test_error_function(X_pred, test_data_inputs).detach().numpy()



    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))
    wandb.log({"Metrics/Generation": ga_instance.generations_completed, "Metrics/Fitness": ga_instance.best_solution()[1]})
    wandb.log({"Metrics/Test_Error": test_error})




input_layer = torch.nn.Linear(2, 16)
relu_layer = torch.nn.ReLU()
sigmoid_layer = torch.nn.Sigmoid()
h1 = torch.nn.Linear(16, 32)
h2 = torch.nn.Linear(32, 64)
h3 = torch.nn.Linear(64, 32)
h4 = torch.nn.Linear(32, 16)
output_layer = torch.nn.Linear(16, 3)

model = torch.nn.Sequential(input_layer,
                            sigmoid_layer,
                            h1,
                            sigmoid_layer,
                            h2,
                            sigmoid_layer,
                            h3,
                            sigmoid_layer,
                            h4,
                            sigmoid_layer,
                            output_layer)
# print(model)

# Create an instance of the pygad.torchga.TorchGA class to build the initial population.
torch_ga = torchga.TorchGA(model=model,
                           num_solutions=arg_numsolutions)

loss_function = torch.nn.MSELoss()
test_error_function = torch.nn.L1Loss()

# Data inputs


training_data = X_train
testing_data = X_test

training_targets = y_train
testing_targets = y_test

data_inputs = torch.from_numpy(training_data)
data_outputs = torch.from_numpy(training_targets)

test_data_inputs = torch.from_numpy(testing_data)
test_data_outputs = torch.from_numpy(testing_targets)

# data_inputs = training_data
# data_outputs = training_targets

# data_inputs = torch.tensor([[0.02, 0.1, 0.15],
#                             [0.7, 0.6, 0.8],
#                             [1.5, 1.2, 1.7],
#                             [3.2, 2.9, 3.1]])

# # Data outputs
# data_outputs = torch.tensor([[0.1],
#                              [0.6],
#                              [1.3],
#                              [2.5]])

# Prepare the PyGAD parameters. Check the documentation for more information: https://pygad.readthedocs.io/en/latest/pygad.html#pygad-ga-class
num_generations = arg_numgeneration # Number of generations.
num_parents_mating = arg_numparents # Number of solutions to be selected as parents in the mating pool.
initial_population = torch_ga.population_weights # Initial population of network weights

ga_instance = pygad.GA(num_generations=num_generations, 
                       num_parents_mating=num_parents_mating, 
                       initial_population=initial_population,
                       fitness_func=fitness_func,
                       on_generation=callback_generation)

ga_instance.run()

# After the generations complete, some plots are showed that summarize how the outputs/fitness values evolve over generations.
# ga_instance.plot_fitness(title="PyGAD & PyTorch - Iteration vs. Fitness", linewidth=4)

# # Returning the details of the best solution.
# solution, solution_fitness, solution_idx = ga_instance.best_solution()
# print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
# print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

# # Fetch the parameters of the best solution.
# best_solution_weights = torchga.model_weights_as_dict(model=model,
#                                                       weights_vector=solution)
# model.load_state_dict(best_solution_weights)
# predictions = model(data_inputs)
# print("Predictions : \n", predictions.detach().numpy())

# abs_error = loss_function(predictions, data_outputs)
# print("Absolute Error : ", abs_error.detach().numpy())


