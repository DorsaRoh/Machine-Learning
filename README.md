# ML from Scratch

## Overview

This repository contains implementations of machine learning algorithms and data structures/algorithms from scratch.


- **Machine Learning**: Implementations of ML algorithms without external libraries (using only numpy)
- **Data Structures & Algorithms**: Fundamental data structures and algorithms, with additional practice questions
- **Biological Parallels**: Exploration of analogies to biological systems and nature


# 1. Neural Networks from Scratch

## Introduction

This README file provides an in-depth explanation of building a neural network from scratch for MNIST digit classification. The MNIST dataset consists of images of single digits (0-9). Our goal is to create a neural network that can accurately classify these digits.

## Table of Contents

1. [Architecture](#architecture)
2. [The Math Behind Neural Networks](#the-math-behind-neural-networks)
3. [Implementation](#implementation)
4. [Training the Network](#training-the-network)
5. [Evaluation](#evaluation)
6. [Conclusion](#conclusion)

## Architecture

![Neural Network Architecture](root/assets/Neural-Networks/1-NN.png)

Our neural network consists of the following layers:

- Input layer: 784 neurons (one for each pixel in the 28x28 MNIST images)
- Hidden layer 1: 128 neurons
- Hidden layer 2: 64 neurons
- Output layer: 10 neurons (one for each digit 0-9)

Total parameters: 13,002 weights and biases

The network processes information as follows:

1. Input layer: Each neuron holds one pixel of an image. The activation value (between 0-1) represents the brightness of the pixel.
2. Hidden layers: These layers extract features from the input.
   - First hidden layer: May correspond to "little edges" in the images
   - Second hidden layer: May correspond to patterns from the edges (e.g., loops, lines)
3. Output layer: Each neuron represents a digit (0-9). The neuron with the highest activation indicates the network's prediction.

Note: While we hope the hidden layers correspond to these features, in reality, the network optimizes to minimize the cost function and may not explicitly capture these patterns.

![Weights Visualization](root/assets/Neural-Networks/3-weights.png)

## The Math Behind Neural Networks

### Activation Functions

We use the ReLU (Rectified Linear Unit) activation function for our hidden layers. ReLU is defined as:

```
ReLU(x) = max(0, x)
```

This function introduces non-linearity into our network, allowing it to learn complex patterns.

### Weighted Sum and Bias

For each neuron, we calculate the weighted sum of its inputs plus a bias term:

```
weighted_sum = w1*a1 + w2*a2 + w3*a3 + ... + wn*an + bias
```

Where:
- wi: weight for input i
- ai: activation from the previous layer's neuron i
- bias: a learnable offset

![Weighted Sum Visualization](root/assets/Neural-Networks/4-weightedSum.png)

The weights determine what pattern the neuron is looking for, while the bias affects how easily the neuron activates.

![Weights and Bias Visualization](root/assets/Neural-Networks/5-weightsAndBias.png)

### Forward Propagation

Forward propagation is the process of passing input through the network to get an output. For each layer:

1. Calculate the weighted sum for each neuron
2. Apply the activation function to get the neuron's output

This process is repeated for each layer until we reach the output layer.

### Backpropagation

Backpropagation is the algorithm used to train the network. It calculates the gradient of the loss function with respect to each weight by applying the chain rule to iteratively compute gradients for each layer.

![Backpropagation Visualization](root/assets/Neural-Networks/10-backprop.png)

The key steps are:

1. Compute the error at the output layer
2. Propagate this error backwards through the network
3. Update weights and biases using the computed gradients

![Backpropagation Diagram](root/assets/Neural-Networks/11-backprop-2.png)

## Implementation

Here's a basic implementation of our neural network in Python:

```python
import numpy as np

class Neuron:
    def __init__(self, num_inputs):
        self.weights = np.random.randn(num_inputs, 1) * 0.01
        self.bias = np.zeros((1, 1))

    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, inputs):
        self.last_input = inputs
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        self.last_output = self.relu(weighted_sum)
        return self.last_output

class Layer:
    def __init__(self, num_neurons, num_inputs_per_neuron):
        self.neurons = [Neuron(num_inputs_per_neuron) for _ in range(num_neurons)]

    def forward(self, inputs):
        return np.array([neuron.forward(inputs) for neuron in self.neurons]).T

class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Layer(layer_sizes[i+1], layer_sizes[i]))

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def train(self, X, y, learning_rate, epochs):
        # Training code here (implement backpropagation)
        pass
```

## Training the Network

To train the network:

1. Initialize the network with random weights and biases
2. For each epoch:
   a. Perform forward propagation on a batch of training examples
   b. Compute the loss
   c. Perform backpropagation to compute gradients
   d. Update weights and biases using gradient descent

## Evaluation

After training, we can evaluate our network on a separate test set to measure its performance. Common metrics include accuracy, precision, recall, and F1-score.

## Conclusion

Building a neural network from scratch provides deep insights into how these powerful models work. While frameworks like TensorFlow and PyTorch make it easier to implement complex networks, understanding the fundamentals is crucial for developing intuition and debugging more advanced models.

Remember, this implementation is meant for educational purposes. For real-world applications, use established libraries that offer optimized performance and additional features.

Happy coding and happy learning!
