from mnist import load_mnist
import argparse

import numpy as np
import os
import sys

sys.path.append('../src/wrapper')

from NeuralNetwork import NeuralNetwork
# from NeuralNetwork import NeuronalNetwork as NeuralNetwork

TRAININGMETHODS = ['batch', 'sgd']


def loadData(dataset, N=100):
    img, numbers = load_mnist(
        dataset=dataset, path='mnist-data/', selection=slice(0, N))

    images = img.astype(np.float32).reshape(
        (img.shape[0], img.shape[1] * img.shape[2])).transpose()
    output = np.array([numbers == i for i in range(10)], dtype=np.float32)

    return images, output, numbers


def outputToNumber(output):
    return np.where(output == np.max(output))[0][0]


def train(method, learningRate, momentum, numEpochs, output, samples, layers=None, minibatchsize=None):
    trainImages, trainOutput, trainNumbers = loadData('training', samples)
    nn = NeuralNetwork(layerCount=len(layers),
                       layerSize=np.array(layers),
                       actFunctions=np.array([1, 1]))

    trainmethod = {
        TRAININGMETHODS[0]: nn.trainBatch,
        TRAININGMETHODS[1]: nn.trainStochastic
    }
    errors = trainmethod[method](
        trainImages, trainOutput, learningRate=learningRate, momentum=momentum, numEpochs=numEpochs)
    # Error did happen, we just return, error message would be printed by the
    # train method
    if not errors or len(errors) == 0:
        return

    # print 'Training done, errors:', errors
    print 'Saving output to', output
    nn.save(output)
    print 'Testing...'
    test(nn)


def loadNetwork(filename):
    nn = NeuralNetwork(saveFile=filename)
    return nn


def test(nn):
    testImages, testOutput, testNumbers = loadData('testing')
    result = nn.test(testImages)
    correct = 0
    print len(testNumbers)
    for i in range(len(testNumbers)):
        num = outputToNumber(result[:, i])
        # print testNumbers[i], num
        if testNumbers[i] == num:
            correct += 1
    print correct, 'of', len(testNumbers), 'correct'


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='Training-test args help')

    trainparser = subparsers.add_parser('train')
    trainparser.set_defaults(which='train')
    trainparser.add_argument('layers', nargs='+', type=int,
                             help='Layers (from input to output) of the neural network. Multiple arguments in a succession are possible')
    trainparser.add_argument(
        '-mom', '--momentum', type=float, help='Momentum of the neural network, default:%(default)s', default=0.001)
    trainparser.add_argument(
        '-ep', '--numEpochs', type=int, default=100, help="Number of epochs for the neural network, default: %(default)s")
    trainparser.add_argument(
        '-lr', '--learningRate', type=float, help='Learning rate of the neural network ,default:%(default)s', default=0.2)
    trainparser.add_argument(
        '-N', '--samples', type=int, help='Number of training samples to use, default:%(default)s',default=100)
    trainparser.add_argument(
        '-m', '--method', default=TRAININGMETHODS[1], help='Trainingsmethod, default : %(default)s')
    trainparser.add_argument(
        '-mb', '--minibatchsize', help='Size of the minibatch. If batch as method is used, this has no effect. Default :%(default)s')
    trainparser.add_argument(
        '-o', '--output', type=str, help='Output file of trained nnet, default:%(default)s', default='output.nnet')

    testparser = subparsers.add_parser('test')
    testparser.set_defaults(which='test')
    testparser.add_argument(
        'nn', metavar='nnet file', help='Trained neural network output file')

    return parser.parse_args()


def main():
    args = parse_args()

    # Testing
    if args.which == 'test':
        # Delete the attribute
        delattr(args, 'which')
        nn = loadNetwork(args.nn)
        test(nn)
    # Training
    else:
        delattr(args, 'which')
        train(**vars(args))

    # train('batch', 0.2, 0.1, 10000)


if __name__ == '__main__':
    main()

# having some argparse support here would be cool
# nnet.py --trainBatch -r 0.2 -m 0.0 -N 100000
# nnet.py --trainStochastic -r 0.2 -m 0.0 -N 100000 --miniBatchSize=20
# nnet.py --test filename.dat
