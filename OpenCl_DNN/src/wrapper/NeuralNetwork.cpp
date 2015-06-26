#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(uint64_t layerCount, uint64_t* layerSize, uint64_t* actFunctions, float learningRate, float momentum) {
	this->layerSize = (uint64_t*) std::malloc(sizeof(uint64_t) * layerCount);
	std::memcpy(this->layerSize, layerSize, sizeof(uint64_t) * layerCount);

	this->actFunctions = (uint64_t*) std::malloc(sizeof(uint64_t) * layerCount);
	std::memcpy(this->actFunctions, actFunctions, sizeof(uint64_t) * layerCount);

	this->layerCount = layerCount;
	this->learningRate = learningRate;
	this->momentum = momentum;

	this->result = (double*) std::malloc(sizeof(double) * this->getOutputSize());

	//vector<uint64_t> layerSizes(this->layerSizes, this->layerSizes + this->layerCount);
	//this->network = new FeedForwardNN(this->getInputSize(), this->getOutputSize(), layerSizes, this->learningRate);

}

NeuralNetwork::NeuralNetwork(std::string saveFile) {
	std::ifstream file (saveFile, std::ios::in | std::ios::binary);
	file.read((char*) &(this->layerCount), sizeof(int64_t));
	file.read((char*) &(this->learningRate), sizeof(float));
	file.read((char*) &(this->momentum), sizeof(float));

	this->layerSize = (uint64_t*) std::malloc(sizeof(uint64_t) * this->layerCount);
	file.read((char*) this->layerSize, sizeof(uint64_t) * this->layerCount);

	this->actFunctions = (uint64_t*) std::malloc(sizeof(uint64_t) * this->layerCount);
	file.read((char*) this->actFunctions, sizeof(int64_t) * this->layerCount);

	this->result = (double*) std::malloc(sizeof(double) * this->getOutputSize());
	file.read((char*) this->result, sizeof(double) * this->getOutputSize());

	file.close();

	//vector<uint64_t> layerSizes(this->layerSizes, this->layerSizes + this->layerCount);
	//this->network = new FeedForwardNN(this->getInputSize(), this->getOutputSize(), layerSizes, this->learningRate);

}

NeuralNetwork::~NeuralNetwork() {
	if(this->result != 0) {
		free(this->result);
	}
	if(this->layerSize != 0) {
		free(this->layerSize);
	}

	if(this->actFunctions != 0) {
		free(this->actFunctions);
	}

	/*if(this->network != 0) {
		delete this->network;
	}*/

}

uint64_t NeuralNetwork::save(std::string saveFile) {
	std::ofstream file (saveFile, std::ios::out | std::ios::binary);
	file.write((char*) &(this->layerCount), sizeof(uint64_t));
	file.write((char*) &(this->learningRate), sizeof(float));
	file.write((char*) &(this->momentum), sizeof(float));
	file.write((char*) this->layerSize, sizeof(uint64_t) * this->layerCount);
	file.write((char*) this->actFunctions, sizeof(uint64_t) * this->layerCount);
	file.write((char*) this->result, sizeof(uint64_t) * this->getOutputSize());
	file.close();

}

void NeuralNetwork::test(float* inputValues, int rowLength, int rowCount) {
    /* Transform row length from the length in byte to the length in floats */
	rowLength /= sizeof(float);

	/* Create matrix */
	CLMatrix matrix(rowCount, this->getInputSize());
	this->fillCLMatrixFromNumpy(matrix, inputValues, rowLength, rowCount);

	/* Run */
	CLMatrix result = this->network.test(matrix);

	/* FIXME: Give result back / output */
}

double NeuralNetwork::getResultNode(uint64_t node) {
	return this->result[node];

}


void NeuralNetwork::fillCLMatrixFromNumpy(CLMatrix &matrix, float* numpy, int rowLength, int rowCount) {
	for(int r=0; r<rowCount; r++) {
		for(int c=0; c<4; c++) {
			float* addr = numpy + rowLength * r + c;
			matrix.fillAt(r, c, *addr);

		}
	}
}

void NeuralNetwork::train(float* inputValues, float* outputValues, int inputRowLength, int outputRowLength, int rowCount) {
    /* Transform row length from the length in byte to the length in floats */
	rowLength /= sizeof(float);

	/* Create matrix */
	CLMatrix matrixIn(rowCount, this->getInputSize());
	CLMatrix matrixOut(rowCount, this->getInputSize());
	this->fillCLMatrixFromNumpy(matrixIn, inputValues, intputRowLength, rowCount);
	this->fillCLMatrixFromNumpy(matrixOut, inputValues, outtputRowLength, rowCount);

	/* Run */
	CLMatrix errors = this->network.trainbatch(matrixIn, matrixOut);

	/* FIXME: Give errors back / output */

}

uint64_t NeuralNetwork::getOutputSize() {
	return this->layerSize[this->layerCount-1];

}

uint64_t NeuralNetwork::getInputSize() {
	return this->layerSize[0];

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
    
    void NeuralNetwork_train(float* inputValues, float* outputValues, int inputRowLength, int outputRowLength, int rowCount) {
    	foo->train(inputValues, outPutValues, inputRowLength, outputRowLength, rowCount); 
    
    }

    void NeuralNetwork_test(NeuralNetwork* foo, float* inputValues, int rowLength, int rowCount) {
    	foo->test(inputValues, rowLength, rowCount); 

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
}