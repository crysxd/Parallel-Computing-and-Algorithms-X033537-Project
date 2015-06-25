#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(int64_t layerCount, int64_t* layerSize, int64_t* actFunctions, float learningRate, float momentum) {
	this->layerSize = (int64_t*) std::malloc(sizeof(uint64_t) * layerCount);
	std::memcpy(this->layerSize, layerSize, sizeof(uint64_t) * layerCount);

	this->actFunctions = (int64_t*) std::malloc(sizeof(uint64_t) * layerCount);
	std::memcpy(this->actFunctions, actFunctions, sizeof(uint64_t) * layerCount);

	this->layerCount = layerCount;
	this->learningRate = learningRate;
	this->momentum = momentum;

	this->result = (int64_t*) std::malloc(sizeof(uint64_t) * this->getOutputSize());

}

NeuralNetwork::NeuralNetwork(std::string saveFile) {
	std::ifstream file (saveFile, std::ios::in | std::ios::binary);
	file.read((char*) &(this->layerCount), sizeof(int64_t));
	file.read((char*) &(this->learningRate), sizeof(float));
	file.read((char*) &(this->momentum), sizeof(float));

	this->layerSize = (int64_t*) std::malloc(sizeof(uint64_t) * this->layerCount);
	file.read((char*) this->layerSize, sizeof(int64_t) * this->layerCount);

	this->actFunctions = (int64_t*) std::malloc(sizeof(uint64_t) * this->layerCount);
	file.read((char*) this->actFunctions, sizeof(int64_t) * this->layerCount);

	this->result = (int64_t*) std::malloc(sizeof(uint64_t) * this->getOutputSize());
	file.read((char*) this->result, sizeof(int64_t) * this->getOutputSize());

	file.close();

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

}

int64_t NeuralNetwork::save(std::string saveFile) {
	std::ofstream file (saveFile, std::ios::out | std::ios::binary);
	file.write((char*) &(this->layerCount), sizeof(int64_t));
	file.write((char*) &(this->learningRate), sizeof(float));
	file.write((char*) &(this->momentum), sizeof(float));
	file.write((char*) this->layerSize, sizeof(int64_t) * this->layerCount);
	file.write((char*) this->actFunctions, sizeof(int64_t) * this->layerCount);
	file.write((char*) this->result, sizeof(int64_t) * this->getOutputSize());
	file.close();

}

void NeuralNetwork::calc(int64_t* inputValues) {
	for(int i=0; i<this->getOutputSize(); i++)
		this->result[i] = 1337+i;

}

int64_t NeuralNetwork::getResultNode(int64_t node) {
	return this->result[node];

}


void NeuralNetwork::train(int64_t* inputValues, int64_t* outputValues) {
	std::cout << "In train()\n";

}

int64_t NeuralNetwork::getOutputSize() {
	return this->layerSize[this->layerCount-1];

}

int64_t NeuralNetwork::getInputSize() {
	return this->layerSize[0];

}

extern "C" {
	NeuralNetwork* NeuralNetwork_new(int64_t layerCount, int64_t* layerSize, int64_t* actFunctions, float learningRate, float momentum) {
		return new NeuralNetwork(layerCount, layerSize, actFunctions, learningRate, momentum); 
	
	}
    
    NeuralNetwork* NeuralNetwork_newLoad(char* saveFile) { 
    	return new NeuralNetwork(saveFile); 

    }

    int64_t NeuralNetwork_save(NeuralNetwork* foo, char* saveFile) { 
    	return foo->save(saveFile); 

    }
    
    void NeuralNetwork_train(NeuralNetwork* foo, int64_t* inputValues, int64_t* outPutValues) {
    	foo->train(inputValues, outPutValues); 
    
    }

    void NeuralNetwork_calc(NeuralNetwork* foo, int64_t* inputValues) {
    	foo->calc(inputValues); 

    }

    int64_t NeuralNetwork_getResultNode(NeuralNetwork* foo, int64_t node) {
    	return foo->getResultNode(node);

    }

    void NeuralNetwork_free(NeuralNetwork* foo) {
    	delete foo; 

    }

    int64_t NeuralNetwork_getOutputSize(NeuralNetwork* foo) {
    	return foo->getOutputSize(); 

    }

    int64_t NeuralNetwork_getInputSize(NeuralNetwork* foo) {
    	return foo->getInputSize(); 

    }
}