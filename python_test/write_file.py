import struct
import pickle
import numpy as np

pickleFile = open('nnet_onelayer_quick.pickle', 'rb')
weights = pickle.load(pickleFile)

layerSizes = [784, 101, 10]


dumpFile = open('nnet_onelayer_quick2.data', 'wb')
dumpFile.write(struct.pack('Q', len(layerSizes))) # uint64 layer count
# dumpFile.write(struct.pack('f', 0.2)) # float learning rate
# dumpFile.write(struct.pack('f', 0.3)) # float momentum
dumpFile.write(struct.pack('Q'*len(layerSizes), *layerSizes)) # uint64 layer sizes
dumpFile.write(struct.pack('Q'*len(layerSizes), *([1]*len(layerSizes)))) # uint64 activation functions, sigmoid == 1
dumpFile.write(struct.pack('Q', len(weights)))

for i, weight in enumerate(weights):
    if i == 0:
        bias = weight[0,:]
        weight = weight[1:,:]
    else:
        bias = np.zeros(weight.shape[0])
    dumpFile.write(struct.pack('QQQ', bias.shape[0], bias.shape[0], 1))
    dumpFile.write(bias.astype(np.float32).tostring())
    dumpFile.write(struct.pack('QQQ', weight.shape[0]*weight.shape[1], weight.shape[0], weight.shape[1]))
    dumpFile.write(weight.astype(np.float32).tostring())