# Automate the process of going through multiple iterations of training different ANNs with different parameters
import MyANN

input_nodes = 784
hidden_nodes = 15
output_nodes = 10
hidden_layers = 2
learn_rate = 0.1
epochs = 1

changing_variable = "learning rate"
max_iterations = 20

if __name__ == "__main__":
    print(f"Training a sequence of ANNs with varying {changing_variable}...")
    for learn_rate in range(11, max_iterations + 1):
        print("\nStarting new training session.")
        learn_rate /= 10
        MyANN.handwritten_digits(input_nodes, hidden_nodes, output_nodes, hidden_layers, learn_rate, epochs)
    print(f"Completed entire training sequence of ANNs with varying number of {max_iterations}.")
