from ctypes import cdll
import numpy as np
import ctypes
from ctypes import *
import os
import time
import pickle
import struct

lib = cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), '../../bin/libnn.so'))

class NeuralNetwork(object):
    def __init__(self, saveFile=None, layerCount=0, layerSize=None, actFunctions=None):
        self.buffer_from_memory = pythonapi.PyBuffer_FromMemory
        self.buffer_from_memory.restype = py_object

        if layerSize != None and actFunctions != None and layerCount != 0:
            layerCount = c_ulonglong(layerCount)
            layerSize = layerSize.ctypes.data_as(POINTER(c_ulonglong))
            actFunctions = actFunctions.ctypes.data_as(POINTER(c_ulonglong))

            self.obj = lib.NeuralNetwork_new(layerCount, layerSize, actFunctions)

        elif saveFile != None:
            lastLayerSize = self.obj = lib.NeuralNetwork_newLoad(saveFile)

        else:
            raise NameError('Insufficent arguments provided.')

    def save(self, saveFile):
        lib.NeuralNetwork_save(self.obj, saveFile)

    def trainBatch(self, inputValues, outputValues, learningRate=0.2, momentum=0.0, numEpochs=50):
        inputvaluespointer = inputValues.ctypes.data_as(POINTER(c_float))
        outputValuespointer = outputValues.ctypes.data_as(POINTER(c_float))
        errors = ctypes.POINTER(ctypes.c_float)()
        errorsLen = ctypes.c_int()
        print 'in', inputValues.shape, inputValues.strides
        print 'out', outputValues.shape, outputValues.strides
        learningRate = c_float(learningRate)
        momentum = c_float(momentum)
        numEpochs = c_ulonglong(numEpochs)
        lib.NeuralNetwork_train(self.obj, inputvaluespointer, inputValues.shape[0], inputValues.shape[1], inputValues.strides[0], inputValues.strides[1], learningRate, momentum, numEpochs, outputValuespointer, outputValues.shape[0],  outputValues.shape[1],  outputValues.strides[0], outputValues.strides[1], ctypes.byref(errors), ctypes.byref(errorsLen))

        return self._toNpArray(errors, (errorsLen.value, ))

    def trainStochastic(self, inputValues, outputValues, learningRate=0.2, momentum=0.0, numEpochs=50, miniBatchSize=10):
        inputvaluespointer = inputValues.ctypes.data_as(POINTER(c_float))
        outputValuespointer = outputValues.ctypes.data_as(POINTER(c_float))
        errors = ctypes.POINTER(ctypes.c_float)()
        errorsLen = ctypes.c_int()
        print 'in', inputValues.shape, inputValues.strides
        print 'out', outputValues.shape, outputValues.strides
        learningRate = c_float(learningRate)
        momentum = c_float(momentum)
        numEpochs = c_ulonglong(numEpochs)
        miniBatchSize = c_ulonglong(miniBatchSize)
        lib.NeuralNetwork_trainsgd(self.obj, inputvaluespointer, inputValues.shape[0], inputValues.shape[1], inputValues.strides[0], inputValues.strides[1], learningRate, momentum, numEpochs, miniBatchSize, outputValuespointer, outputValues.shape[0],  outputValues.shape[1],  outputValues.strides[0], outputValues.strides[1], ctypes.byref(errors), ctypes.byref(errorsLen))

        return self._toNpArray(errors, (errorsLen.value, ))

    def test(self, inputValues):
        inputvaluespointer = inputValues.ctypes.data_as(POINTER(c_float))
        rows = ctypes.c_int()
        cols = ctypes.c_int()
        result = ctypes.POINTER(ctypes.c_float)()
        lib.NeuralNetwork_test(self.obj, inputvaluespointer, inputValues.shape[0], inputValues.shape[1], inputValues.strides[0], inputValues.strides[1], ctypes.byref(result), ctypes.byref(rows), ctypes.byref(cols))
        return self._toNpArray(result, (rows.value, cols.value))

    def getResultNode(self, node):
        node = c_int(node)
        return lib.NeuralNetwork_getResultNode(self.obj, node)

    def getOutputSize(self):
        return lib.NeuralNetwork_getOutputSize(self.obj)

    def getInputSize(self):
        return lib.NeuralNetwork_getInputSize(self.obj)

    def __del__(self):
        lib.NeuralNetwork_free(self.obj)

    def readMatTest(self):
        rows = ctypes.c_int()
        cols = ctypes.c_int()
        data = ctypes.POINTER(ctypes.c_float)()
        lib.NeuralNetwork_readMatTest(self.obj, ctypes.byref(data), ctypes.byref(rows), ctypes.byref(cols))

        buffer = self.buffer_from_memory(data, 4*rows.value*cols.value)

        a = np.frombuffer(buffer, np.float32).reshape((rows.value, cols.value))
        return a

    def _toNpArray(self, data, shape):
        buffer = self.buffer_from_memory(data, 4*np.prod(shape))
        return np.frombuffer(buffer, np.float32).reshape(shape)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1.0 - sigmoid(x))


def tanh(x):
    return np.tanh(x)


def tanh_prime(x):
    return 1.0 - x**2

class NeuronalNetwork(object):
    def __init__(self, saveFile=None, layerCount=0, layerSize=None, actFunctions=None):
        if 1 in actFunctions:
            self.activation = sigmoid
            self.activation_prime = sigmoid_prime
        elif 0 in actFunctions:
            self.activation = tanh
            self.activation_prime = tanh_prime

        # Set weights
        self.weights = []
        self.layerSizes = layerSize
        # layers = [2,2,1]
        # range of weight values (-1,1)
        # input and hidden layers - random((2+1, 2+1)) : 3 x 3
        for i in range(1, len(layerSize) - 1):
            r = 2 * \
                np.random.random((layerSize[i - 1] + 1, layerSize[i] + 1)) - 1
            self.weights.append(r)
        # output layer - random((2+1, 1)) : 3 x 1
        r = 2 * np.random.random((layerSize[i] + 1, layerSize[i + 1])) - 1
        self.weights.append(r)

    def trainBatch(self):
        pass

    def trainStochastic(self, X, y, numEpochs, learningRate=0.2, momentum=0.001, minibatchsize=None):
        # Add column of ones to X
        # This is to add the bias unit to the input layer
        X = X.T
        y = y.T
        ones = np.atleast_2d(np.ones(X.shape[0]))
        X = np.concatenate((ones.T, X), axis=1)
        errors = []
        t = time.time()

        for k in range(numEpochs):
            i = np.random.randint(X.shape[0])
            a = [X[i]]
            for l in range(len(self.weights)):
                dot_value = np.dot(a[l], self.weights[l])
                activation = self.activation(dot_value)
                a.append(activation)
            # output layer
            error = y[i] - a[-1]
            errors.append(error)
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
                self.weights[i] += learningRate * layer.T.dot(delta)
            if k % 10000 == 0:
                print 'epochs:', k, round(time.time() - t, 2), 's'
        return errors

    def test(self, x):
        a = np.concatenate((np.ones(1).T, np.array(x)), axis=1)
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a

    def save(self,fileout):

        dumpFile = open(fileout, 'wb')
        dumpFile.write(struct.pack('Q', len(self.layerSizes))) # uint64 layer count
        # dumpFile.write(struct.pack('f', 0.2)) # float learning rate
        # dumpFile.write(struct.pack('f', 0.3)) # float momentum
        dumpFile.write(struct.pack('Q'*len(self.layerSizes), *self.layerSizes)) # uint64 layer sizes
        dumpFile.write(struct.pack('Q'*len(self.layerSizes), *([1]*len(self.layerSizes)))) # uint64 activation functions, sigmoid == 1
