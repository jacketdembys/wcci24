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
    def __init__(self, input_size, s1_hudden_list, s2_hidden_list, middle_state_size, output_size, second_network_path=None):
        super(IK_Network, self).__init__()
        # The first MLP maps inputs to middle states
        self.mlp1 = MLP(input_size, s1_hudden_list, middle_state_size)  # Example hidden sizes

        # The second MLP maps middle states to outputs
        self.mlp2 = MLP(middle_state_size, s2_hidden_list, output_size)  # Example hidden sizes

        # Load the second network weights if provided
        if second_network_path:
            self.mlp2.load_state_dict(torch.load(second_network_path))
            # Freeze the second network
            for param in self.mlp2.parameters():
                param.requires_grad = False

    def forward(self, x):
        middle_state = self.mlp1(x)
        output = self.mlp2(middle_state)
        return output
    


