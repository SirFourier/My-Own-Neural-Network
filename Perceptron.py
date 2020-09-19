# A Perceptron is a simple linear classifier, which means the output
# is a linear summation of the inputs.

# A simple mathematical example:
# f(x) = 1 if w.x + b > 0
# f(x) = 0 otherwise
# where w = vector of weights
# w.x is the dot product of w and x
# b = bias

# For N inputs, f(x) becomes the sum of theta*f(x)
# where theta = threshold function.

# Using numpy for matrix maths and scipy for activation function
import numpy


# Create a single Perceptron class
class Perceptron:

    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    @classmethod
    def set_activation_function(cls, function):
        cls.activation_function = function

    def calculate_output(self, inputs):
        summed = numpy.dot(inputs, self.weights)
        summed += self.bias
        output = self.activation_function(summed)
        return output
