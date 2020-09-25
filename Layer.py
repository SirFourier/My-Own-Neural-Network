# A layer class for each layer in a Neural Network

# Import numpy for optimised matrix maths
import numpy as np


# Create a Layer class
class Layer:

    # Specify number of nodes from the previous layer and the number of nodes in this layer
    def __init__(self, previous_nodes, nodes):
        # initialise weights using He initialisation
        self.weights = np.random.normal(0.0, pow(2/previous_nodes, -0.5), (nodes, previous_nodes))
        self.d_weights = np.full((nodes, previous_nodes), 0.0)

        # initialise biases using zeros (transpose into 2D (1D) column matrix)
        self.biases = np.full(nodes, 0.0).reshape(-1, 1)
        self.d_biases = np.full(nodes, 0.0).reshape(-1, 1)

        # create empty outputs (transpose into 2D (1D) column matrix)
        self.outputs = np.full(nodes, 0.0).reshape(-1, 1)
        self.act_outputs = np.full(nodes, 0.0).reshape(-1, 1)

    def calculate_act_outputs(self, act_inputs, act):
        # store outputs before activation function for backpropagation
        self.outputs = np.matmul(self.weights, act_inputs) + self.biases
        self.act_outputs = act(self.outputs)
        return self.act_outputs
