#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(uint64_t layerCount, uint64_t* layerSize, uint64_t* actFunctions, float learningRate, float momentum)
    : result(Matrix(1,1,0.0f)) {
	this->layerSize = (uint64_t*) std::malloc(sizeof(uint64_t) * layerCount);
	std::memcpy(this->layerSize, layerSize, sizeof(uint64_t) * layerCount);

	this->actFunctions = (uint64_t*) std::malloc(sizeof(uint64_t) * layerCount);
	std::memcpy(this->actFunctions, actFunctions, sizeof(uint64_t) * layerCount);

	this->layerCount = layerCount;
	this->learningRate = learningRate;
	this->momentum = momentum;

	this->initNetwork();

}

NeuralNetwork::NeuralNetwork(std::string saveFile)
    : result(Matrix(1,1,0.0f)) {
	std::ifstream file (saveFile, std::ios::in | std::ios::binary);
	file.read((char*) &(this->layerCount), sizeof(int64_t));
	file.read((char*) &(this->learningRate), sizeof(float));
	file.read((char*) &(this->momentum), sizeof(float));

	this->layerSize = (uint64_t*) std::malloc(sizeof(uint64_t) * this->layerCount);
	file.read((char*) this->layerSize, sizeof(uint64_t) * this->layerCount);

	this->actFunctions = (uint64_t*) std::malloc(sizeof(uint64_t) * this->layerCount);
	file.read((char*) this->actFunctions, sizeof(int64_t) * this->layerCount);

	/* Init network */
	this->initNetwork();

	/* Read size of weights in byte */
	size_t weightsSize;
	file.read((char*) &weightsSize, sizeof(size_t));

	/* Allocate buffer, read and assign */
	void* buf = malloc(weightsSize);
//	CHRIS: TODO: GIBT FEHLER
//	file.read((char*) buf, weightsSize);
//	this->network->getWeightBiases().assign(buf, buf + weightsSize);

	file.close();



}

void NeuralNetwork::initNetwork() {
	/* Create nn */
	this->network = new FeedForwardNN(uint32_t(this->getInputSize()), uint32_t(this->getOutputSize()), this->learningRate);
    
    /* Add layers */
    for(int i=1; i<this->layerCount-1; i++) {
    	this->network->addHiddenLayer(this->layerSize[i]);
    }

	/* Add actionations */
    for(int i=0; i<this->layerCount-1; i++) {
    	/* Add TanH function */
    	if(this->actFunctions[i] == ACTIVATION_TAN_H)  {
        	this->network->addActivation(&this->tanH);
    	}

        /* SIGMOID is default, if a int is not mapped */
        else {
        	this->network->addActivation(&this->sigmoid);
        }
    }
}

NeuralNetwork::~NeuralNetwork() {
	if(this->layerSize != 0) {
		free(this->layerSize);
	}

	if(this->actFunctions != 0) {
		free(this->actFunctions);
	}

	if(this->network != 0) {
		delete this->network;
	}

}

uint64_t NeuralNetwork::save(std::string saveFile) {
	char* weights = (char*) this->network->getWeightBiases().data();
	size_t weightsSize = this->network->getWeightBiases().size() * sizeof(std::pair<Matrix, Matrix>);

	std::ofstream file (saveFile, std::ios::out | std::ios::binary);
	file.write((char*) &(this->layerCount), sizeof(uint64_t));
	file.write((char*) &(this->learningRate), sizeof(float));
	file.write((char*) &(this->momentum), sizeof(float));
	file.write((char*) this->layerSize, sizeof(uint64_t) * this->layerCount);
	file.write((char*) this->actFunctions, sizeof(uint64_t) * this->layerCount);
//	CHRIS TODO: DUNNO WHAT TO DO
//	file.write((char*) &this->weightsSize, sizeof(size_t));
 	file.write(weights, weightsSize);
	file.close();

}

void NeuralNetwork::test(float* inputValues, int shape0, int shape1, int strides0, int strides1, float *resultOut[], int *resultRows, int *resultCols) {
	/* Create matrix */
	Matrix matrix(shape0, shape1);
	this->fillMatrixFromNumpy(matrix, inputValues, shape0, shape1, strides0, strides1);

	/* Run */
	this->result = this->network->test(matrix);

    *resultOut = this->result.data();
    *resultRows = this->result.getRows();
    *resultCols = this->result.getCols();
}

double NeuralNetwork::getResultNode(uint64_t node) {
	return this->result[node];

}


void NeuralNetwork::fillMatrixFromNumpy(Matrix &matrix, float* numpy, int shape0, int shape1, int strides0, int strides1) {
	for(int r=0; r < shape0; r++) {
		for(int c=0; c < shape1; c++) {
//             std::cout << r*strides1 + c*strides0 << ' ' << r << ' ' << c << '\n';
//             std::cout.flush();
			float* addr = numpy + r*strides0/4 + c*strides1/4;
			matrix.fillAt(r, c, *addr);
		}
	}
}

void NeuralNetwork::train(float* inputValues, int inShape0, int inShape1, int inStrides0, int inStrides1, float* outputValues, int outShape0, int outShape1, int outStrides0, int outStrides1, float *errorsOut[], int *errorsLen) {
//     std::cout << inputRowLength << ',' << outputRowLength << ',' << rowCount << '\n';
    /* Transform row length from the length in byte to the length in floats */
// 	inputRowLength = inputRowLength/ sizeof(float);
// 	outputRowLength = outputRowLength/ sizeof(float);
	/* Create matrix */
	Matrix matrixIn(inShape0, inShape1);
	Matrix matrixOut(outShape0, outShape1);
	this->fillMatrixFromNumpy(matrixIn, inputValues, inShape0, inShape1, inStrides0, inStrides1);
	this->fillMatrixFromNumpy(matrixOut, outputValues, outShape0, outShape1, outStrides0, outStrides1);
    std::cout << "in " << matrixIn.getRows() << 'x' << matrixIn.getCols() << '\n';
    std::cout << "out " << matrixOut.getRows() << 'x' << matrixOut.getCols() << '\n';

	/* Run */
	this->lastErrors = this->network->trainbatch(matrixIn, matrixOut);
    std::cout << "trained\n";

	*errorsOut = this->lastErrors.data();
    *errorsLen = this->lastErrors.size();
}


void NeuralNetwork::trainsgd(float* inputValues, int inShape0, int inShape1, int inStrides0, int inStrides1, float* outputValues, int outShape0, int outShape1, int outStrides0, int outStrides1, float *errorsOut[], int *errorsLen) {
//     std::cout << inputRowLength << ',' << outputRowLength << ',' << rowCount << '\n';
    /* Transform row length from the length in byte to the length in floats */
//  inputRowLength = inputRowLength/ sizeof(float);
//  outputRowLength = outputRowLength/ sizeof(float);
    /* Create matrix */
    Matrix matrixIn(inShape0, inShape1);
    Matrix matrixOut(outShape0, outShape1);
    this->fillMatrixFromNumpy(matrixIn, inputValues, inShape0, inShape1, inStrides0, inStrides1);
    this->fillMatrixFromNumpy(matrixOut, outputValues, outShape0, outShape1, outStrides0, outStrides1);
    std::cout << "in " << matrixIn.getRows() << 'x' << matrixIn.getCols() << '\n';
    std::cout << "out " << matrixOut.getRows() << 'x' << matrixOut.getCols() << '\n';

    /* Run */
    this->lastErrors = this->network->trainsgd(matrixIn, matrixOut);
    std::cout << "trained\n";

    *errorsOut = this->lastErrors.data();
    *errorsLen = this->lastErrors.size();
}

uint64_t NeuralNetwork::getOutputSize() {
	return this->layerSize[this->layerCount-1];

}

uint64_t NeuralNetwork::getInputSize() {
	return this->layerSize[0];

}

float testData[] = {1.1, 1.2, 1.3, 1.4, 1.5, 1.6};

void NeuralNetwork::readMatTest(float *out[], int *rows, int *cols) {
    *out = testData;
    *rows = 2;
    *cols = 3;
}

extern "C" {
	NeuralNetwork* NeuralNetwork_new(uint64_t layerCount, uint64_t* layerSize, uint64_t* actFunctions, float learningRate, float momentum) {
		return new NeuralNetwork(layerCount, layerSize, actFunctions, learningRate, momentum);

	}

    NeuralNetwork* NeuralNetwork_newLoad(char* saveFile) {
    	return new NeuralNetwork(saveFile);

    }

    uint64_t NeuralNetwork_save(NeuralNetwork* foo, char* saveFile) {
    	return foo->save(saveFile);

    }

    void NeuralNetwork_train(NeuralNetwork* foo, float* inputValues, int inShape0, int inShape1, int inStrides0, int inStrides1, float* outputValues, int outShape0, int outShape1, int outStrides0, int outStrides1, float *errorsOut[], int *errorsLen) {
    	foo->train(inputValues, inShape0, inShape1, inStrides0, inStrides1, outputValues, outShape0, outShape1, outStrides0, outStrides1, errorsOut, errorsLen);

    }

    void NeuralNetwork_trainsgd(NeuralNetwork* foo, float* inputValues, int inShape0, int inShape1, int inStrides0, int inStrides1, float* outputValues, int outShape0, int outShape1, int outStrides0, int outStrides1, float *errorsOut[], int *errorsLen) {
        foo->trainsgd(inputValues, inShape0, inShape1, inStrides0, inStrides1, outputValues, outShape0, outShape1, outStrides0, outStrides1, errorsOut, errorsLen);

    }

    void NeuralNetwork_test(NeuralNetwork* foo, float* inputValues, int shape0, int shape1, int strides0, int strides1, float *resultOut[], int *resultRows, int *resultCols) {
    	foo->test(inputValues, shape0, shape1, strides0, strides1, resultOut, resultRows, resultCols);

    }

    double NeuralNetwork_getResultNode(NeuralNetwork* foo, uint64_t node) {
    	return foo->getResultNode(node);

    }

    void NeuralNetwork_free(NeuralNetwork* foo) {
    	delete foo;

    }

    uint64_t NeuralNetwork_getOutputSize(NeuralNetwork* foo) {
    	return foo->getOutputSize();

    }

    uint64_t NeuralNetwork_getInputSize(NeuralNetwork* foo) {
    	return foo->getInputSize();

    }

    void NeuralNetwork_readMatTest(NeuralNetwork* foo, float *out[], int *rows, int *cols) {
        foo->readMatTest(out, rows, cols);
    }
}
