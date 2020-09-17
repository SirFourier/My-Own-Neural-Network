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

# Using numpy for matrix maths and matplotlib for displaying data
import numpy
import matplotlib.pyplot as plt


# Create a single Perceptron class
class Perceptron:
    def __init__(self, input_list, weights_list, bias):
        # convert input and weights lists into numpy arrays
        self.__inputs = numpy.array(input_list)
        self.__weights = numpy.array(weights_list)
        self.__bias = bias
        self.__summed = 0

    # perform dot product of inputs and weights
    def __sum_dot(self):
        self.__summed = numpy.dot(self.__inputs, self.__weights)

    # add the bias
    def __add_bias(self):
        self.__summed += self.__bias

    # calculate output
    def calculate_output(self):
        self.__sum_dot()
        self.__add_bias()
        output = 1 if self.__summed > 0 else 0
        return output


# display truth table of all possible boolean inputs
def truth_table(weights, bias):
    for i in range(2):
        for j in range(2):
            perceptron = Perceptron([i, j], weights, bias)
            print(f"Inputs: {i, j} Result: {perceptron.calculate_output()}")

    print()


# Our main function
def main():
    # print truth able using the following weights as inputs
    # weights = [1.0, 1.0] and bias = -1 generates an AND function
    # weights = [1.0, 1.0] and bias =  0 generates an OR function
    truth_table([1.0, 1.0], -1)

    # make a plot in XKCD style
    fig = plt.xkcd()

    plt.scatter(0, 0, s=50, color="red", zorder=3)
    plt.scatter(0, 1, s=50, color="red", zorder=3)
    plt.scatter(1, 0, s=50, color="red", zorder=3)
    plt.scatter(1, 1, s=50, color="green", zorder=3)

    # weights and bias
    w1 = 1.0
    w2 = 1.0
    b = -1

    # plot line
    x_data = numpy.arange(-2.0, 2.1, 0.1)
    y_data = (-w1/w2)*x_data - b/w2
    plt.plot(x_data, y_data)

    plt.xlim(-2, 2)
    plt.ylim(-2, 2)

    plt.xlabel("Input 1")
    plt.ylabel("Input 2")
    plt.title("State Space of Input Vector")

    plt.grid(True, linewidth=1, linestyle=':')

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
