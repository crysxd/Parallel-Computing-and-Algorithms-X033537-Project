#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(uint64_t layerCount, uint64_t* layerSize, uint64_t* actFunctions)
    : result(Matrix(1,1,0.0f)) {
	this->layerSize = (uint64_t*) std::malloc(sizeof(uint64_t) * layerCount);
	std::memcpy(this->layerSize, layerSize, sizeof(uint64_t) * layerCount);

	this->actFunctions = (uint64_t*) std::malloc(sizeof(uint64_t) * layerCount);
	std::memcpy(this->actFunctions, actFunctions, sizeof(uint64_t) * layerCount);

    this->layerCount = layerCount;

	this->initNetwork();

}

NeuralNetwork::NeuralNetwork(std::string saveFile)
    : result(Matrix(1,1,0.0f)) {
	std::ifstream file (saveFile, std::ios::in | std::ios::binary);
    file.read((char*) &(this->layerCount), sizeof(int64_t));

	this->layerSize = (uint64_t*) std::malloc(sizeof(uint64_t) * this->layerCount);
	file.read((char*) this->layerSize, sizeof(uint64_t) * this->layerCount);

	this->actFunctions = (uint64_t*) std::malloc(sizeof(uint64_t) * this->layerCount);
	file.read((char*) this->actFunctions, sizeof(int64_t) * this->layerCount);

	/* Init network */
	this->initNetwork();

	/* Read the number of pairs */
	size_t weightsCount = 0;
	file.read((char*)&weightsCount, sizeof(size_t));

	std::vector<std::pair<Matrix,Matrix>> weights = this->network->getWeightBiases();

	/* For all pairs */
	for(auto i=0u; i<weightsCount; i++) {
		/* Read both vectors from pair */
		Matrix m1(1,1), m2(1, 1);
		m1 = readMatrix(file);
		m2 = readMatrix(file);

		/* Add to weights */
		weights.push_back(std::pair<Matrix,Matrix>(m1, m2));

	}

	file.close();

}

void NeuralNetwork::initNetwork() {
	/* Create nn */
    this->network = new FeedForwardNN(uint32_t(this->getInputSize()), uint32_t(this->getOutputSize()));

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
	std::vector<std::pair<Matrix,Matrix>> weights = this->network->getWeightBiases();
	size_t weightsCount = weights.size();

	std::ofstream file (saveFile, std::ios::out | std::ios::binary);
    file.write((char*) &(this->layerCount), sizeof(uint64_t));
	file.write((char*) this->layerSize, sizeof(uint64_t) * this->layerCount);
	file.write((char*) this->actFunctions, sizeof(uint64_t) * this->layerCount);
	/* Write the number of pairs */
	file.write((char*) &weightsCount, sizeof(size_t));
	/* For all pairs */
	for(int i=0; i<weightsCount; i++) {
		/* Write both vectors from pair */
		Matrix m1 = weights[i].first;
		Matrix m2 = weights[i].second;

		writeMatrix(m1, file);
		writeMatrix(m2, file);

	}

	file.close();

}

void writeMatrix(Matrix &m, std::ofstream &s) {
	std::vector<float> data = m.rawData();
	size_t size = data.size();
	size_t rows = m.getRows();
	size_t cols = m.getCols();

	/* Write length */
	s.write((char*) &size, sizeof(size_t));
	s.write((char*) &rows, sizeof(size_t));
	s.write((char*) &cols, sizeof(size_t));

	/* Write elements */
	for(int i=0; i<size; i++) {
		s.write((char*) &data[i], sizeof(float));
	}
}

Matrix& readMatrix(std::ifstream &s) {
	/* Read length */
	size_t size = 0;
	size_t rows = 0;
	size_t cols = 0;
	s.read((char*)&size, sizeof(size_t));
	s.read((char*)&rows, sizeof(size_t));
	s.read((char*)&cols, sizeof(size_t));

	/* Create matrix */
	Matrix* m = new Matrix(rows, cols);
	float* r =  m->data();

	/* Write elements */
	for(int i=0; i<size; i++) {
		float f;
		s.read((char*)&f, sizeof(float));
		r[i] = f;
	}

	return *m;
}


void test() {
	std::cout << "Hello World!" << std::endl;
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

void NeuralNetwork::train(float* inputValues, int inShape0, int inShape1, int inStrides0, int inStrides1, float learningRate, float momentum, int numEpochs, float* outputValues, int outShape0, int outShape1, int outStrides0, int outStrides1, float *errorsOut[], int *errorsLen) {
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
    this->lastErrors = this->network->trainbatch(matrixIn, matrixOut, learningRate, momentum, numEpochs);
    std::cout << "trained\n";

	*errorsOut = this->lastErrors.data();
    *errorsLen = this->lastErrors.size();
}


void NeuralNetwork::trainsgd(float* inputValues, int inShape0, int inShape1, int inStrides0, int inStrides1, float learningRate, float momentum, int numEpochs, int miniBatchSize, float* outputValues, int outShape0, int outShape1, int outStrides0, int outStrides1, float *errorsOut[], int *errorsLen) {
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
    this->lastErrors = this->network->trainsgd(matrixIn, matrixOut, learningRate, momentum, numEpochs, miniBatchSize);
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
    NeuralNetwork* NeuralNetwork_new(uint64_t layerCount, uint64_t* layerSize, uint64_t* actFunctions) {
        return new NeuralNetwork(layerCount, layerSize, actFunctions);

	}

    NeuralNetwork* NeuralNetwork_newLoad(char* saveFile) {
    	return new NeuralNetwork(saveFile);

    }

    uint64_t NeuralNetwork_save(NeuralNetwork* foo, char* saveFile) {
    	return foo->save(saveFile);

    }

    void NeuralNetwork_train(NeuralNetwork* foo, float* inputValues, int inShape0, int inShape1, int inStrides0, int inStrides1, float learningRate, float momentum, int numEpochs, float* outputValues, int outShape0, int outShape1, int outStrides0, int outStrides1, float *errorsOut[], int *errorsLen) {
        foo->train(inputValues, inShape0, inShape1, inStrides0, inStrides1, learningRate, momentum, numEpochs, outputValues, outShape0, outShape1, outStrides0, outStrides1, errorsOut, errorsLen);

    }

    void NeuralNetwork_trainsgd(NeuralNetwork* foo, float* inputValues, int inShape0, int inShape1, int inStrides0, int inStrides1, float learningRate, float momentum, int numEpochs, int miniBatchSize, float* outputValues, int outShape0, int outShape1, int outStrides0, int outStrides1, float *errorsOut[], int *errorsLen) {
        foo->trainsgd(inputValues, inShape0, inShape1, inStrides0, inStrides1, learningRate, momentum, numEpochs, miniBatchSize, outputValues, outShape0, outShape1, outStrides0, outStrides1, errorsOut, errorsLen);

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
