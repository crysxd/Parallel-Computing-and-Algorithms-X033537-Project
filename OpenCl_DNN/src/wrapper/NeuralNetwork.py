from ctypes import cdll
import numpy as np
import ctypes
from ctypes import *
import os

lib = cdll.LoadLibrary(os.path.join(os.path.dirname(__file__), '../../bin/libnn.so'))

class NeuralNetwork(object):
    def __init__(self, saveFile=None, layerCount=0, layerSize=None, actFunctions=None, learningRate=0, momentum=0):
        self.buffer_from_memory = pythonapi.PyBuffer_FromMemory
        self.buffer_from_memory.restype = py_object

        if layerSize != None and actFunctions != None and layerCount != 0 and learningRate != 0:
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
        inputvaluespointer = inputValues.ctypes.data_as(POINTER(c_float))
        outputValuespointer = outputValues.ctypes.data_as(POINTER(c_float))
        errors = ctypes.POINTER(ctypes.c_float)()
        errorsLen = ctypes.c_int()
        print 'in', inputValues.shape, inputValues.strides
        print 'out', outputValues.shape, outputValues.strides
        lib.NeuralNetwork_train(self.obj, inputvaluespointer, inputValues.shape[0], inputValues.shape[1], inputValues.strides[0], inputValues.strides[1], outputValuespointer, outputValues.shape[0],  outputValues.shape[1],  outputValues.strides[0], outputValues.strides[1], ctypes.byref(errors), ctypes.byref(errorsLen))

        return self._toNpArray(errors, (errorsLen.value, ))

    def trainsgd(self, inputValues, outputValues):
        inputvaluespointer = inputValues.ctypes.data_as(POINTER(c_float))
        outputValuespointer = outputValues.ctypes.data_as(POINTER(c_float))
        errors = ctypes.POINTER(ctypes.c_float)()
        errorsLen = ctypes.c_int()
        print 'in', inputValues.shape, inputValues.strides
        print 'out', outputValues.shape, outputValues.strides
        lib.NeuralNetwork_trainsgd(self.obj, inputvaluespointer, inputValues.shape[0], inputValues.shape[1], inputValues.strides[0], inputValues.strides[1], outputValuespointer, outputValues.shape[0],  outputValues.shape[1],  outputValues.strides[0], outputValues.strides[1], ctypes.byref(errors), ctypes.byref(errorsLen))

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
