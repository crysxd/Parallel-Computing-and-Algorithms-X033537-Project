from mnist import load_mnist
import numpy as np
import os
import sys

sys.path.append('../src/wrapper')
from NeuralNetwork import NeuralNetwork

def loadData(dataset):
    img, numbers = load_mnist(dataset=dataset, path='mnist-data/')

    images = img.astype(np.float32).reshape((img.shape[0], img.shape[1]*img.shape[2])).transpose()
    output = np.array([numbers==i for i in range(10)], dtype=np.float32)

    return images, output, numbers

def outputToNumber(output):
    return np.where(output==np.max(output))[0][0]

trainImages, trainOutput, trainNumbers = loadData('training')
testImages, testOutput, testNumbers = loadData('testing')

nn = NeuralNetwork(layerCount=3,
                   layerSize=np.array([5,10,2]),
                   actFunctions=np.array([1,2,3]),
                   learningRate=1337.1,
                   momentum=1337.2)

errors = nn.train(trainImages, trainOutput)
print errors

result = nn.test(testImages)
print result

errors = sum([outputToNumber(result[:,i]) == testNumbers[i] for i in range(result.shape[1])])
print errors, 'errors, total:', result.shape[1]
