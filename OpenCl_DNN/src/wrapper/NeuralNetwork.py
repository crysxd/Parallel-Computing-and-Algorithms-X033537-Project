from ctypes import cdll
import numpy as np
from ctypes import *

lib = cdll.LoadLibrary('./libnn.so')

class NeuralNetwork(object):
    def __init__(self, saveFile=None, layerCount=0, layerSize=None, actFunctions=None, learningRate=0, momentum=0):
        if layerSize != None and actFunctions != None and layerCount != 0 and learningRate != 0 and momentum != 0:
            layerCount = c_ulonglong(layerCount)
            layerSize = layerSize.ctypes.data_as(POINTER(c_ulonglong))
            actFunctions = actFunctions.ctypes.data_as(POINTER(c_ulonglong))
            learningRate = c_float(learningRate)
            momentum = c_float(momentum)            

            self.obj = lib.NeuralNetwork_new(layerCount, layerSize, actFunctions, learningRate, momentum)

        elif saveFile != None:
            lastLayerSize = self.obj = lib.NeuralNetwork_newLoad(saveFile)

        else:
            raise NameError('Insufficent arguments provided.')

    def save(self, saveFile):
        lib.NeuralNetwork_save(self.obj, saveFile)

    def train(self, inputValues, outputValues):
        inputValues = inputValues.ctypes.data_as(POINTER(c_float))
        outputValues = outputValues.ctypes.data_as(POINTER(c_float))
        lib.NeuralNetwork_train(self.obj, inputValues, outputValues)

    def calc(self, inputValues):
        inputValues = inputValues.ctypes.data_as(POINTER(c_float))
        lib.NeuralNetwork_calc(self.obj, inputValues)

    def getResultNode(self, node):
        node = c_int(node)
        return lib.NeuralNetwork_getResultNode(self.obj, node)

    def getOutputSize(self):
        return lib.NeuralNetwork_getOutputSize(self.obj)

    def getInputSize(self):
        return lib.NeuralNetwork_getInputSize(self.obj)

    def __del__(self):
        lib.NeuralNetwork_free(self.obj)
