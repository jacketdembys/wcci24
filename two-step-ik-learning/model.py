import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(hidden_sizes)):
            if i == 0:
                layers.append(nn.Linear(input_size, hidden_sizes[i]))
            else:
                layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)



class IK_Network(nn.Module):
    def __init__(self, input_size, s1_hudden_list, s2_hidden_list, middle_state_size, output_size,model_choice,second_network_path=None):
        super(IK_Network, self).__init__()
        # The first MLP maps inputs to middle states
        self.mlp1 = MLP(input_size, s1_hudden_list, middle_state_size)  # Example hidden sizes

        # The second MLP maps middle states to outputs
        # self.mlp2 = MLP(middle_state_size, s2_hidden_list, output_size)  # Example hidden sizes

        # self.resmlp = ResMLP(middle_state_size, s2_hidden_list, output_size)

        # Load the second network weights if provided
        if model_choice == "MLP":
            self.second_net = MLP(middle_state_size, s2_hidden_list, output_size)
            self.second_net.load_state_dict(torch.load(second_network_path))
            # Freeze the second network
            for param in self.second_net.parameters():
                param.requires_grad = False
        
        if model_choice == "ResMLP":
            self.second_net = ResMLP(middle_state_size, s2_hidden_list, output_size)
            self.second_net.load_state_dict(torch.load(second_network_path))
            for param in self.second_net.parameters():
                param.requires_grad = False
        
        if model_choice == "Jacket-MLP":
            self.second_net = Jacket_MLP(middle_state_size, s2_hidden_list, output_size)
            self.second_net.load_state_dict(torch.load(second_network_path))
            # Freeze the second network
            for param in self.second_net.parameters():
                param.requires_grad = False


    def forward(self, x):
        middle_state = self.mlp1(x)
        output = self.second_net(middle_state)
        return output, middle_state
    


class ResMLP(nn.Module):
    def __init__(self, input_dim, h_sizes, output_dim):
        super().__init__()

        self.name = "ResMLP [{}, {}, {}]".format(str(input_dim), str(h_sizes).replace("[","").replace("]",""), str(output_dim))
        self.input_dim = input_dim
        self.h_sizes = h_sizes
        self.output_dim = output_dim
        
        self.input_fc = nn.Linear(self.input_dim, self.h_sizes[0])
        self.hidden_fc_1 = nn.Linear(self.h_sizes[0], self.h_sizes[1])
        self.hidden_fc_2 = nn.Linear(self.h_sizes[1], self.h_sizes[2])
        self.hidden_fc_3 = nn.Linear(self.h_sizes[2], self.h_sizes[3])
        self.output_fc = nn.Linear(self.h_sizes[len(self.h_sizes)-1], self.output_dim)       

        self.selu_activation = nn.SELU()
        self.relu_activation = nn.ReLU()
        self.prelu_activation = nn.PReLU()
        self.lrelu_activation = nn.LeakyReLU()
        self.sigmoid_activation = nn.Sigmoid()
        self.batch_norm_fc = nn.BatchNorm1d(20000)

    def forward(self, x):

        x = self.input_fc(x)
        x = self.relu_activation(x)  # ReLU(), Sigmoid(), LeakyReLU(negative_slope=0.1)

        h1 = self.hidden_fc_1(x)
        h1 = self.relu_activation(h1)

        h2 = self.hidden_fc_2(h1)
        h2 = self.relu_activation(h2)

        h3 = self.hidden_fc_3(h2+h1)
        h3 = self.relu_activation(h3)

        o = self.output_fc(h3+h2+h1)
        x_temp = o

        return o 
    


class Jacket_MLP(nn.Module):

    def __init__(self, input_dim, h_sizes, output_dim):
        super().__init__()
        self.name = "MLP [{}, {}, {}]".format(str(input_dim), str(h_sizes).replace("[","").replace("]",""), str(output_dim))
        self.input_dim = input_dim
        self.h_sizes = h_sizes
        self.output_dim = output_dim     
        self.input_fc = nn.Linear(self.input_dim, self.h_sizes[0])
        self.relu_activation = nn.ReLU()      
        self.hidden_fc = nn.ModuleList()

        for i in range(len(self.h_sizes)-1):

            self.hidden_fc.append(nn.Linear(self.h_sizes[i], self.h_sizes[i+1]))

        

        self.output_fc = nn.Linear(self.h_sizes[len(self.h_sizes)-1], self.output_dim)



        

        

    def forward(self, x):
        x = self.input_fc(x)
        x = self.relu_activation(x)
        for i in range(len(self.h_sizes)-1):
            x = self.hidden_fc[i](x)
            x = self.relu_activation(x)
        x = self.output_fc(x)
        x_temp = x
        
        return x