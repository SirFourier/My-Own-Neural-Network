# My own Neural Network
A Feed Forward Artificial Neural Network from first principles (using numpy) with variable number of hidden layers. This is tested on the MNIST dataset.

## Install and run
-  ``git clone https://github.com/SirFourier/My-Own-Neural-Network``
-  ``pip install -r requirements.txt``
-  ``python MyANN.py``

## Motivation
I wanted to have a go at creating an Artificial Neural Network (ANN) from first principles to better understand how it works. Therefore, I am not using machine learning libraries such as Tensorflow or PyTorch. This is so I can learn the underlying maths thats involved.

## How to use the Neural Network class
The class takes 4 ``__init__`` arguments:
  - ``my_ANN = NeuralNetwork(input_nodes=784, hidden_nodes=38, output_nodes=10, hidden_layers=2)``

To train the network is to call upon the ``.train`` method:
  - ``my_ANN.train(inputs, targets, learn_rate=0.1, epochs=1)``
  - ``inputs`` and ``targets`` must be a numpy array
  - ``epochs``= number of times to train the same training set
  
To use the trained network is to call upon the ``.feed_forward`` method:
  - ``prediction = my_ANN.feed_forward(test_inputs)``
  - ``test_inputs`` must be a numpy array

## The Maths involved
