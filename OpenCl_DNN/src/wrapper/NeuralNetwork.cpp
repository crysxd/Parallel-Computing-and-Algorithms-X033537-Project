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

	vector<uint64_t> layerSizes(this->layerSizes, this->layerSizes + this->layerCount);
	this->network = new FeedForwardNN(this->getInputSize(), this->getOutputSize(), layerSizes, this->learningRate);

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

void NeuralNetwork::calc(double* inputValues) {
	std::cout << "In calc()\n";

	std::cout << "input values: \n";
	for(int i=0; i<this->getInputSize(); i++)
		std::cout << "\t" << inputValues[i] << "\n";

	for(int i=0; i<this->getOutputSize(); i++)
		this->result[i] = 1337+i;

}

double NeuralNetwork::getResultNode(uint64_t node) {
	return this->result[node];

}


void NeuralNetwork::train(double* inputValues, double* outputValues) {
	std::cout << "In train()\n";
		std::cout << "input values: \n";
	for(int i=0; i<this->getInputSize(); i++)
		std::cout << "\t" << inputValues[i] << "\n";


	std::cout << "output values: \n";
	for(int i=0; i<this->getOutputSize(); i++)
		std::cout << "\t" << outputValues[i] << "\n";

	//CL_Matrix<float> target;
	//CL_Matrix<float> input;
	//std::vector<float> errors = dnn.trainbatch(input,target);

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
    
    void NeuralNetwork_train(NeuralNetwork* foo, double* inputValues, double* outPutValues) {
    	foo->train(inputValues, outPutValues); 
    
    }

    void NeuralNetwork_calc(NeuralNetwork* foo, double* inputValues) {
    	foo->calc(inputValues); 

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