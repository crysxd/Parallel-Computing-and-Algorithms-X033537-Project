from mnist import load_mnist
import numpy as np
import os
import sys

sys.path.append('../src/wrapper')
from NeuralNetwork import NeuralNetwork


def loadData(dataset):
    img, numbers = load_mnist(dataset=dataset, path='mnist-data/', selection=slice(0,100))

    images = img.astype(np.float32).reshape(
        (img.shape[0], img.shape[1] * img.shape[2])).transpose()
    output = np.array([numbers == i for i in range(10)], dtype=np.float32)

    return images, output, numbers


def outputToNumber(output):
    return np.where(output == np.max(output))[0][0]


def train(method, learningRate, momentum, numEpochs, miniBatchSize=None):
    trainImages, trainOutput, trainNumbers = loadData('training')
    nn = NeuralNetwork(layerCount=3,
                       layerSize=np.array([784, 100, 10]),
                       actFunctions=np.array([1, 1]))
    errors = nn.trainBatch(trainImages[:,0:20], trainOutput[:,0:20], learningRate=0.2, momentum=0.1, numEpochs=numEpochs)
    print 'Training done, errors:', errors
    test(nn)
    nn.save('nnet.dat')

def loadNetwork(filename):
    nn = NeuralNetwork(saveFile=filename)
    return nn

def test(nn):
    testImages, testOutput, testNumbers = loadData('testing')
    result = nn.test(testImages)
    correct = 0
    for i in range(len(testNumbers)):
        num = outputToNumber(result[:, i])
        print testNumbers[i], num
        if testNumbers[i] == num:
            correct += 1
    print correct, 'of', len(testNumbers), 'correct'

train('batch', 0.2, 0.1, 10000)

# having some argparse support here would be cool
# nnet.py --trainBatch -r 0.2 -m 0.0 -N 100000
# nnet.py --trainStochastic -r 0.2 -m 0.0 -N 100000 --miniBatchSize=20
# nnet.py --test filename.dat
