# My own Artificial Neural Network
# This is a two-layer feed-forward neural network
# A Neural Network is an interconnected system of perceptrons
# Using sigmoid activation function

# Import numpy for optimised matrix maths
import numpy as np

# Import layer class
from Layer import Layer


# Create a Feed Forward Neural Network class
class NeuralNetwork:

    # specify number of input, hidden nodes, output nodes, and hidden layers
    def __init__(self, input_nodes, hidden_nodes, output_nodes, hidden_layers):
        # define layout (input layer + hidden layers + output layer)
        layout = [input_nodes] + [hidden_nodes for _ in range(hidden_layers)] + [output_nodes]

        # create list of layers based on number of nodes current layer and the previous layer
        self.layers = [Layer(previous, current) for previous, current in zip(layout, layout[1:])]

        # define derivative of activation function
        self.act_prime = np.vectorize(lambda x: self.act(x) * (1 - self.act(x)))

        # define derivative of cost function
        self.cost_prime = np.vectorize(lambda x, y: 2 * (x - y))

    # define activation function (sigmoid)
    @staticmethod
    def act(x):
        return 1 / (1 + np.exp(-x))

    # define derivative of activation function
    def act_prime(self, x):
        return self.act(x) * (1 - self.act(x))

    # define cost function
    @staticmethod
    def cost_prime(target, prediction):
        return 2 * (prediction - target)

    # define threshold of output
    @staticmethod
    @np.vectorize
    def threshold(x):
        if x > 0.9:
            return 1.0
        elif x < 0.1:
            return 0.0
        else:
            return 0

    # test
    def test(self, inputs, targets, lr):
        # Create random weights and biases for layer 1
        W1 = np.random.normal(0.0, pow(2 / 2, -0.5), (3, 2))
        b1 = np.full((3, 1), 0)

        # Create random weights and biases for layer 2
        W2 = np.random.normal(0.0, pow(2 / 3, -0.5), (3, 3))
        b2 = np.full((3, 1), 0)

        # Create random weights and biases for layer 3
        W3 = np.random.normal(0.0, pow(2 / 3, -0.5), (2, 3))
        b3 = np.full((2, 1), 0)

        for _ in range(1000):
            for i in range(4):
                # Convert inputs list into numpy array
                A0 = np.array(inputs[i]).reshape(-1, 1)

                # convert targets list into numpy array
                Y = np.array(targets[i]).reshape(-1, 1)

                # Feed forward through each layer
                Z1 = W1 @ A0 + b1
                A1 = self.act(Z1)

                Z2 = W2 @ A1 + b2
                A2 = self.act(Z2)

                Z3 = W3 @ A2 + b3
                A3 = self.act(Z3)

                # calculate dJ_dW3
                dJ_A3 = self.cost_prime(Y, A3)
                dA3_dZ3 = self.act_prime(Z3)
                dZ3_dW3 = A2.T
                dJ_dW3 = (dJ_A3 * dA3_dZ3) @ dZ3_dW3

                # calculate dJ_dW2
                dJ_dA2 = W3.T @ (dJ_A3 * dA3_dZ3)
                dA2_dZ2 = self.act_prime(Z2)
                dZ_dW2 = A1.T
                dJ_dW2 = (dJ_dA2 * dA2_dZ2) @ dZ_dW2

                # calculate dJ_dW1
                dJ_dA1 = W2.T @ (dJ_dA2 * dA2_dZ2)
                dA1_dZ1 = self.act_prime(Z1)
                dZ_dW1 = A0.T
                dJ_dW1 = (dJ_dA1 * dA1_dZ1) @ dZ_dW1

                # adjust all weights
                W1 += lr * dJ_dW1
                W2 += lr * dJ_dW2
                W3 += lr * dJ_dW3

        for i in range(4):
            # Convert inputs list into numpy array
            A0 = np.array(inputs[i]).reshape(-1, 1)

            # Feed forward through each layer
            Z1 = W1 @ A0 + b1
            A1 = self.act(Z1)

            Z2 = W2 @ A1 + b2
            A2 = self.act(Z2)

            Z3 = W3 @ A2 + b3
            A3 = self.act(Z3)

            print(self.threshold(A3.T))


# if this script is the main script executed
if __name__ == "__main__":
    # define Neural Network parameters
    input_nodes = 2
    hidden_nodes = 3
    output_nodes = 2
    hidden_layers = 1

    # create Neural Network object
    my_ANN = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, hidden_layers)

    # train Network for an AND gate
    inputs_list = [[0, 0], [0, 1], [1, 0], [1, 1]]
    targets_list = [[1, 1], [1, 0], [0, 1], [0, 0]]
    learning_rate = 0.5

    my_ANN.test(inputs_list, targets_list, learning_rate)
