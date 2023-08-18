import numpy as np

class ANN:
    def __init__(self, layers):
        # Initialize the weights and biases randomly
        self.weights = [np.random.randn(x, y) for x, y in zip(layers[:-1], layers[1:])]
        self.biases = [np.random.randn(y) for y in layers[1:]]

    def feedforward(self, inputs):
        # Propagate the inputs through the network
        for w, b in zip(self.weights, self.biases):
            inputs = sigmoid(np.dot(inputs, w) + b)
        return inputs

    def backprop(self, inputs, targets, learning_rate):
        # Initialize the gradients for the weights and biases
        weight_grads = [np.zeros(w.shape) for w in self.weights]
        bias_grads = [np.zeros(b.shape) for b in self.biases]

        # Propagate the inputs through the network
        activation = inputs
        activations = [inputs] # List to store all the activations, layer by layer
        zs = [] # List to store all the z vectors, layer by layer
        for w, b in zip(self.weights, self.biases):
            z = np.dot(activation, w) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # Compute the error at the output layer
        error = activations[-1] - targets

        # Compute the gradients for the output layer
        weight_grads[-1] = error * sigmoid_prime(zs[-1])
        bias_grads[-1] = error * sigmoid_prime(zs[-1])

        # Propagate the errors backwards
        for l in range(2, len(self.layers)):
            z = zs[-l]
            sp = sigmoid_prime(z)
            error = np.dot(error, self.weights[-l+1].transpose()) * sp
            weight_grads[-l] = error * sigmoid_prime(z)
            bias_grads[-l] = error * sigmoid_prime(z)

        # Update the weights and biases using the gradients
        self.weights = [w - learning_rate*dw for w, dw in zip(self.weights, weight_grads)]
        self.biases = [b - learning_rate*db for b, db in zip(self.biases, bias_grads)]

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))
