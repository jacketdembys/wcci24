%African Vulture Optimization alghorithm

% Read the following publication first and cite if you use it

% @article{abdollahzadeh2021african,
%   title={African Vultures Optimization Algorithm: A New Nature-Inspired Metaheuristic Algorithm for Global Optimization Problems},
%   author={Abdollahzadeh, Benyamin and Gharehchopogh, Farhad Soleimanian and Mirjalili, Seyedali},
%   journal={Computers \& Industrial Engineering},
%   pages={107408},
%   year={2021},
%   publisher={Elsevier},
%   url = {https://www.sciencedirect.com/science/article/pii/S0360835221003120}
% }

%% Code modified by Jacket Demby's (PhD Student - ViGIR) to see how AVOA may perform as an IK Solver
clear all 
close all
clc

rng(3)


my_objective    = "poserpyss";       % position, orientation, poserpyss
robot           = "kinova6";        % panda, kinova6
pop_size        = 500;              % Population size
max_iter        = 1000;             % stoppoing condition based on number of iteration
Weights         = [0.8 0.2];        % weights between
pose_tolerance  = [(1/1000),deg2rad(1)]; 
show_iter_info  = true;  

% Define your objective function's details here
%fobj = @ObjectiveFunction;
fobj = @(Q,D,U,d,ot,w,robot) IK_cost_b2_vectorized(Q,D,U,d,ot,w,robot);

if robot == "kinova6"
    variables_no =  6;
    lower_bound =   deg2rad([-360,  50,  19, -360, -360, -360]); % can be a vector too
    upper_bound =   deg2rad([ 360, 310, 341,  360,  360,  360]); % can be a vector too
    data = table2array(readtable(strcat("from-cec-ideas/cec_data_points_RRRRRR_3.csv")));
end



% Problem definition
problem.fobj = fobj;
problem.variables_no = variables_no;
problem.lower_bound = lower_bound;
problem.upper_bound = upper_bound;
problem.DesiredTolerance = pose_tolerance;
problem.ObjectiveType = my_objective;

if problem.ObjectiveType == "position"
    problem.Dimension = 3;    
    problem.DesiredPose = data(1,1:3);
else    
    problem.Dimension = 6;   
    data(1,1:3) = data(1,1:3);
    problem.DesiredPose = data(1,1:6);
end
problem.Robot = robot;

% Parameters choice
params.pop_size = pop_size;
params.max_iter = max_iter;
params.Weights = Weights; 
params.ShowIterInfo = show_iter_info;

start_time = tic;      
[outputs]=AVOA(problem, params);
end_time = toc(start_time);

Best_vulture1_F = outputs.BestCost;
Best_vulture1_X = outputs.BestSolution;
convergence_curve = outputs.ConvergenceCurve;

fprintf("Elapsed time: %d seconds\n\n", end_time)

figure(1) 

% Best optimal values for the decision variables 
subplot(1,2,1)
parallelcoords(Best_vulture1_X)
xlabel('Decision variables')
ylabel('Best estimated values ')
grid on
box on

% Best convergence curve
subplot(1,2,2)
plot(convergence_curve);
title('Convergence curve of AVOA')
xlabel('Current_iteration');
ylabel('Objective value');
grid on
box on

