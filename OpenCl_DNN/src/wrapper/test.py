from NeuralNetwork import NeuralNetwork
import numpy as np

arr = np.array([[1.2, 23.4, 23.4, 45.4],[1.2, 2.2, 3.3, 4.4]], dtype=np.float32)
f = NeuralNetwork(None, 3, np.array([5,10,2]), np.array([1,2,3]), 1337.1, 1337.2)

print f.getInputSize()
print f.getOutputSize()
# arr = f.test(arr)
# f.save("savefile.bin")

# print "Load...."
# f = NeuralNetwork("savefile.bin")
# print f.getInputSize()
# print f.getOutputSize()
# for i in range(0, f.getOutputSize()):
#     print f.getResultNode(i)

# print f.readMatTest()
# print type(f.readMatTest())
