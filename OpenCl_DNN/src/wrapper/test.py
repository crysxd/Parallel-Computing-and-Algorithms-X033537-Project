from NeuralNetwork import NeuralNetwork
import numpy as np

f = NeuralNetwork(None, 3, np.array([5,10,2]), np.array([1,2,3]), 1337.1, 1337.2)

print f.getInputSize()
print f.getOutputSize()
arr = f.calc(np.array([1,2,3,4,5]))
for i in range(0, f.getOutputSize()):
    print f.getResultNode(i)
f.save("savefile.bin")

print "Load...."
f = NeuralNetwork("savefile.bin")
print f.getInputSize()
print f.getOutputSize()
for i in range(0, f.getOutputSize()):
    print f.getResultNode(i)