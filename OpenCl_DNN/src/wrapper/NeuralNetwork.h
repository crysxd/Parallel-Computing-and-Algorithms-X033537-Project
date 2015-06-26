#include <iostream>
#include <cstring>
#include <cstdint>
#include <fstream>
#include "CLMatrix.hpp"
#include "FeedForwardNN.h"

class NeuralNetwork {
	double* result = 0;
	uint64_t layerCount;
	uint64_t* layerSize = 0;
	uint64_t* actFunctions = 0;
	float learningRate;
	float momentum;
	FeedForwardNN* network = 0;

public:
	NeuralNetwork(uint64_t layerCount, uint64_t* layerSize, uint64_t* actFunctions, float learningRate, float momentum);
	NeuralNetwork(std::string saveFile);
	~NeuralNetwork();
	uint64_t save(std::string saveFile);
	double getResultNode(uint64_t node);
	void calc(double* inputValues);
	void train(double* inputValues, double* outputValues);
	uint64_t getOutputSize();
	uint64_t getInputSize();

};