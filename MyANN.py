# My own Artificial Neural Network
# This is a two-layer feed-forward neural network
# A Neural Network is an interconnected system of perceptrons
# Using sigmoid activation function

import numpy


# Create a Neural Network class
class NeuralNetwork:

    # specify number of input, hidden layers, hidden nodes, output nodes and activation function
    def __init__(self, input_nodes, hidden_layers, hidden_nodes, output_nodes, activation_function):
        self.i_nodes = input_nodes
        self.h_nodes = hidden_nodes
        self.o_nodes = output_nodes
        self.h_layers = hidden_layers
        self.a_function = numpy.vectorize(activation_function)

        # note each hidden layer will have the same number of hidden_nodes

        # weights layer 1 = input layer        -> hidden layer 1
        # weights layer 2 = hidden layer 1     -> hidden layer 2
        # weights layer N = hidden layer (N-1) -> output layer
        # therefore, number of weights layers = number of hidden layers + 1

        # define weights layer list where each element stores the matrix of weights for that layer
        self.w_layer_list = []

        # define bias list for each weights layer
        self.b_layer_list = []

        # initialise our weights using a normal distribution of random numbers
        # define weights matrix for input -> hidden layer 1
        w_i_to_h1 = numpy.random.normal(0.0, pow(self.h_nodes, -0.5), (self.h_nodes, self.i_nodes))
        b_i_to_h1 = [0 for _ in range(self.i_nodes)]
        self.w_layer_list.append(w_i_to_h1)
        self.b_layer_list.append(b_i_to_h1)

        # number of weight layers not including first and last = number of hidden layers - 1
        # define the rest of the weights layer except the last one
        for _ in range(self.h_layers - 1):
            w_h_to_h = numpy.random.normal(0.0, pow(self.h_nodes, -0.5), (self.h_nodes, self.h_nodes))
            b_h_to_h = [0 for _ in range(self.i_nodes)]
            self.w_layer_list.append(w_h_to_h)
            self.b_layer_list.append(b_h_to_h)

        # define weights matrix for hidden layer (N-1) -> output layer
        w_hl_to_o = numpy.random.normal(0.0, pow(self.o_nodes, -0.5), (self.o_nodes, self.h_nodes))
        b_hl_to_o = [0 for _ in range(self.h_nodes)]
        self.w_layer_list.append(w_hl_to_o)
        self.b_layer_list.append(b_hl_to_o)

    # calculate output nodes (also known as feed forwarding)
    def feed_forward(self, inputs):
        # convert inputs list into numpy array
        input_layer = numpy.array(inputs)

        # calculate first hidden layer
        first_hidden_layer = numpy.dot(self.w_layer_list[0], input_layer) + self.b_layer_list[0]
        first_hidden_layer = self.a_function(first_hidden_layer)
        previous_hidden_layer = first_hidden_layer

        # calculate the rest of the hidden layers
        for i in range(1, self.h_layers):
            current_hidden_layer = numpy.dot(self.w_layer_list[i], previous_hidden_layer) + self.b_layer_list[i]
            current_hidden_layer = self.a_function(current_hidden_layer)
            previous_hidden_layer = current_hidden_layer

        # calculate output layer
        output_layer = numpy.dot(self.w_layer_list[-1], previous_hidden_layer) + self.b_layer_list[-1]
        output_layer = self.a_function(output_layer)

        return output_layer

    # train the network using backpropagation
    def train(self, loss_function):
        pass


# a activation function
def rectified_linear(x):
    return 0 if x < 0 else 0


# a loss function
def sum_of_squares_error(predicted, target):
    return sum((target - predicted) ** 2)


# main program
def main():
    input_nodes = 2
    hidden_layers = 2
    hidden_nodes = 2
    output_nodes = 1
    activation_function = rectified_linear
    loss_function = sum_of_squares_error

    my_ANN = NeuralNetwork(input_nodes, hidden_layers, hidden_nodes, output_nodes, activation_function)


# if this script is the main script executed
if __name__ == "__main__":
    main()
