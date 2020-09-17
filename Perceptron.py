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

# Using numpy for matrix maths
import numpy


# Create a single Perceptron class
class Perceptron:

    def __init__(self, input_list, weights_list, bias):
        # convert input and weights lists into numpy arrays
        self.__inputs = numpy.array(input_list)
        self.__weights = numpy.array(weights_list)
        self.__bias = bias
        self.__summed = 0

    # perform dot product of inputs and weights
    def sum(self):
        self.__summed = numpy.dot(self.__inputs, self.__weights)

    # add the bias
    def add_bias(self):
        self.__summed += self.__bias

    # calculate output
    def calculate_output(self):
        output = 1 if self.__summed > 0 else 0
        return output


# Our main function
def main():
    # define inputs, weights, and bias
    inputs = [1.0, 0.0]
    weights = [1.0, 1.0]
    bias = -1

    # create a Perceptron object
    perceptron = Perceptron(inputs, weights, bias)

    # print results
    print(f"Inputs:  {inputs}")
    print(f"Weights: {weights}")
    print(f"Bias:    {bias}")
    print(f"Result:  {perceptron.calculate_output()}")


if __name__ == "__main__":
    main()
