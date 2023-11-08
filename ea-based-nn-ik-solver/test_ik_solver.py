import numpy as np

layers = 1
neurons = 500

hidden_layer_sizes = np.zeros((1,layers))          

for neuron in range(100, neurons+100, 100):

    hidden_layer_sizes[:,:] = neuron
    print(hidden_layer_sizes)