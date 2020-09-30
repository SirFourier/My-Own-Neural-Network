# My own Feed-Forward Artificial Neural Network with variable number of hidden layers
import numpy as np
from Layer import Layer
from mnist import MNIST


# Create an artificial neural network
class NeuralNetwork:
    # specify number of input nodes, hidden nodes, output nodes, and hidden layers
    def __init__(self, input_nodes, hidden_nodes, output_nodes, hidden_layers):
        # define layout (input layer + hidden layers + output layer)
        layout = [input_nodes] + [hidden_nodes for _ in range(hidden_layers)] + [output_nodes]

        # create list of layers based on number of nodes in current layer and the previous layer
        self.layers = [Layer(previous, current) for previous, current in zip(layout, layout[1:])]

    # define activation function (sigmoid)
    @staticmethod
    def act(x):
        return 1. / (1. + np.exp(-x))

    # define derivative of activation function (sigmoid prime)
    def act_prime(self, x):
        return self.act(x) * (1. - self.act(x))

    # define cost function (outputs a single number: lower number means more accurate)
    @staticmethod
    def cost(target, prediction):
        return 0.5 * np.sum((target - prediction) ** 2)

    # define derivative of cost function with respect to each prediction (our error function)
    @staticmethod
    def cost_prime(target, prediction):
        return prediction - target

    # perform feed forward
    def feed_forward(self, input_layer):
        # convert inputs to column vector and make it current layer
        current_layer = input_layer.reshape(-1, 1)

        # calculate activations through all the layers
        for layer in self.layers:
            current_layer = layer.calculate_act_outputs(current_layer, self.act)

        # return final output layer
        return current_layer

    # train network
    def train(self, inputs, targets, learn_rate, epochs):
        # define number of training sets
        n_sets = len(inputs)

        # train network
        for _ in range(epochs):
            for i in range(n_sets):
                # convert targets to column vector
                Y = targets[i].reshape(-1, 1)

                # feed forward
                self.feed_forward(inputs[i])

                # --------------------------------backpropagate with gradient descent------------------------------ #
                # calculate dJ_dW and dJ_dB of output layer
                dJ_dZ = self.cost_prime(Y, self.layers[-1].act_outputs) * self.act_prime(self.layers[-1].outputs)
                self.layers[-1].dJ_dW = dJ_dZ @ self.layers[-2].act_outputs.T
                self.layers[-1].dJ_dB = dJ_dZ

                # backpropagate up to but not including first hidden layer and calculate dJ_dW and dJ_dB
                for j in range(1, len(self.layers) - 1):
                    dJ_dZ = (self.layers[-j].weights.T @ dJ_dZ) * self.act_prime(self.layers[-j-1].outputs)
                    self.layers[-j-1].dJ_dW = dJ_dZ @ self.layers[-j-2].act_outputs.T
                    self.layers[-j-1].dJ_dB = dJ_dZ

                # calculate dJ_dW and dJ_dB of first hidden layer
                dJ_dZ = (self.layers[1].weights.T @ dJ_dZ) * self.act_prime(self.layers[0].outputs)
                self.layers[0].dJ_dW = dJ_dZ @ inputs[i].reshape(1, -1)
                self.layers[0].dJ_dB = dJ_dZ

                # adjust all weights and biases
                for layer in self.layers:
                    layer.weights -= learn_rate * layer.dJ_dW
                    layer.biases -= learn_rate * layer.dJ_dB


if __name__ == "__main__":
    # ----------------------------------------Handwritten Digits----------------------------------------------- #
    # load MNIST dataset
    mn_data = MNIST('./MNIST')
    mn_data.gz = True
    mn_images, mn_labels = mn_data.load_training()

    # normalise image data from 0-255 to 0.01 to 1
    images_array = (np.array(mn_images) / 255.0 * 0.99) + 0.01

    # convert labels into array of 10 elements with all zeros while the labeled element is replaced by 1
    labels_list = []
    for label in mn_labels:
        empty = np.full(10, 0.01)
        empty[label] = 0.99
        labels_list.append(empty)
    labels_array = np.array(labels_list)

    # instantiate Neural Network
    my_ANN = NeuralNetwork(input_nodes=784, hidden_nodes=38, output_nodes=10, hidden_layers=2)

    # train network
    my_ANN.train(images_array, labels_array, learn_rate=0.1, epochs=1)

    # test network
    mn_test_images, mn_test_labels = mn_data.load_testing()
    test_images_array = (np.array(mn_test_images) / 255.0 * 0.99) + 0.01
    test_labels_array = np.array(mn_test_labels)
    correct_labels = 0
    for test_label, test_image in zip(test_labels_array, test_images_array):
        guess = np.argmax(my_ANN.feed_forward(test_image))
        if test_label == guess:
            correct_labels += 1
    n_tests = len(test_labels_array)
    score = correct_labels / n_tests * 100
    print(f"The network was {score} % correct on {n_tests} test images!")

    # # -----------------------------------------XOR gate----------------------------------------------- #
    # # instantiate Neural Network
    # my_ANN = NeuralNetwork(input_nodes=2, hidden_nodes=2, output_nodes=1, hidden_layers=1)
    #
    # inputs_array = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # targets_array = np.array([0, 1, 1, 0])
    #
    # cost_data = my_ANN.train(inputs_array, targets_array, learn_rate=0.3, epochs=5000)
    #
    # for inputs in inputs_array:
    #     print(my_ANN.feed_forward(inputs))
