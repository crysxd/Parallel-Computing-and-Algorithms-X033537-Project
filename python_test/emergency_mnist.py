import numpy as np
import os
import sys
import struct
import cPickle as pickle
sys.path.append('../OpenCl_DNN/demo')
from mnist import load_mnist

# sys.path.append('../OpenCl_DNN/src/wrapper')
# from NeuralNetwork import NeuralNetwork

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1.0 - sigmoid(x))


def tanh(x):
    return np.tanh(x)


def tanh_prime(x):
    return 1.0 - x**2


class NeuralNetwork:

    def __init__(self, layers, activation='tanh'):
        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_prime = sigmoid_prime
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_prime = tanh_prime

        # Set weights
        self.weights = []
        # layers = [2,2,1]
        # range of weight values (-1,1)
        # input and hidden layers - random((2+1, 2+1)) : 3 x 3
        for i in range(1, len(layers) - 1):
            r = 2 * np.random.random((layers[i - 1] + 1, layers[i] + 1)) - 1
            self.weights.append(r)
        # output layer - random((2+1, 1)) : 3 x 1
        r = 2 * np.random.random((layers[i] + 1, layers[i + 1])) - 1
        self.weights.append(r)

    def fit(self, X, y, learning_rate=0.2, epochs=100000):
        # Add column of ones to X
        # This is to add the bias unit to the input layer
        ones = np.atleast_2d(np.ones(X.shape[0]))
        X = np.concatenate((ones.T, X), axis=1)

        for k in range(epochs):
            i = np.random.randint(X.shape[0])
            a = [X[i]]

            for l in range(len(self.weights)):
                dot_value = np.dot(a[l], self.weights[l])
                activation = self.activation(dot_value)
                a.append(activation)
            # output layer
            error = y[i] - a[-1]
            deltas = [error * self.activation_prime(a[-1])]

            # we need to begin at the second to last layer
            # (a layer before the output layer)
            for l in range(len(a) - 2, 0, -1):
                deltas.append(
                    deltas[-1].dot(self.weights[l].T) * self.activation_prime(a[l]))

            # reverse
            # [level3(output)->level2(hidden)]  => [level2(hidden)->level3(output)]
            deltas.reverse()

            # backpropagation
            # 1. Multiply its output delta and input activation
            #    to get the gradient of the weight.
            # 2. Subtract a ratio (percentage) of the gradient from the weight.
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)

            if k % 10000 == 0:
                print 'epochs:', k

    def predict(self, x):
        a = np.concatenate((np.ones(1).T, np.array(x)), axis=1)
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a



def loadData(dataset):
    img, numbers = load_mnist(dataset=dataset, path='../OpenCl_DNN/demo/mnist-data/')

    images = img.astype(np.float32).reshape(
        (img.shape[0], img.shape[1] * img.shape[2])).transpose()
    output = np.array([numbers == i for i in range(10)], dtype=np.float32)

    return images, output, numbers

def outputToNumber(output):
    return np.where(output == np.max(output))[0][0]

if __name__ == '__main__':
    trainImages, trainOutput, trainNumbers = loadData('training')
    # trainImages, trainOutput, trainNumbers = loadData('testing')
    testImages, testOutput, testNumbers = loadData('testing')
    NTrain = 10000

    layerSizes = [784, 784, 100, 10]
    nn = NeuralNetwork(layerSizes, 'sigmoid')
    # nn.fit(trainImages[:, 0:NTrain].T, trainOutput[:, 0:NTrain].T, learning_rate=0.2, epochs=10000000)
    nn.fit(trainImages.T, trainOutput.T, learning_rate=0.2, epochs=100000)
    NTest = testImages.shape[1]
    correct = 0
    for i in range(NTest):
        out = nn.predict(testImages[:, i].T)
        # print(testNumbers[i], outputToNumber(out), out)
        if outputToNumber(out) == testNumbers[i]:
            correct += 1
    print correct, 'of', NTest, 'correct'

    dumpFile = open('nnet.data', 'wb')
    dumpFile.write(struct.pack('Q', len(layerSizes))) # uint64 layer count
    # dumpFile.write(struct.pack('f', 0.2)) # float learning rate
    # dumpFile.write(struct.pack('f', 0.3)) # float momentum
    dumpFile.write(struct.pack('Q'*len(layerSizes), *layerSizes)) # uint64 layer sizes
    dumpFile.write(struct.pack('Q'*len(layerSizes), *([1]*len(layerSizes)))) # uint64 activation functions, sigmoid == 1

    pickleFile = open('nnet.pickle', 'wb')
    pickle.dump(nn.weights, pickleFile)

