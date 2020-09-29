# A layer class for each layer in a Neural Network
import numpy as np


# Create a Layer class
class Layer:

    # Specify number of nodes from the previous layer and the number of nodes in this layer
    def __init__(self, previous_nodes, nodes):
        # initialise weights with standard deviation of sqrt(nodes)
        self.weights = np.random.normal(0.0, pow(nodes, -0.5), (nodes, previous_nodes))
        self.dJ_dW = np.full((nodes, previous_nodes), 0.0)

        # initialise biases using zeros
        self.biases = np.full((nodes, 1), 0.0)
        self.dJ_dB = np.full((nodes, 1), 0.0)

        # create empty outputs
        self.outputs = np.full((nodes, 1), 0.0)
        self.act_outputs = np.full((nodes, 1), 0.0)

    def calculate_act_outputs(self, act_inputs, act):
        # store outputs before activation function for backpropagation
        self.outputs = (self.weights @ act_inputs) + self.biases
        self.act_outputs = act(self.outputs)
        return self.act_outputs
