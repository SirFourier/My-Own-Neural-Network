# A layer class for each layer in a Neural Network
import numpy as np


# Create a Layer class
class Layer:

    # Specify number of nodes from the previous layer and the number of nodes in this layer
    def __init__(self, previous_nodes, nodes):
        # initialise weights using He initialisation (2/previous nodes before)
        self.weights = np.random.normal(0.0, pow(nodes, -0.5), (nodes, previous_nodes))
        self.dJ_dW = np.full((nodes, previous_nodes), 0.0)

        # initialise biases using zeros (transpose into 2D (1D) column matrix)
        self.biases = np.full(nodes, 0.0).reshape(-1, 1)
        self.dJ_dB = np.full(nodes, 0.0).reshape(-1, 1)

        # create empty outputs (transpose into 2D (1D) column matrix)
        self.outputs = np.full(nodes, 0.0).reshape(-1, 1)
        self.act_outputs = np.full(nodes, 0.0).reshape(-1, 1)

    def calculate_act_outputs(self, act_inputs, act):
        # store outputs before activation function for backpropagation
        self.outputs = (self.weights @ act_inputs) + self.biases
        self.act_outputs = act(self.outputs)
        return self.act_outputs
