#include <iostream>
#include <cstring>
#include <cstdint>
#include <fstream>

class NeuralNetwork {
	int64_t* result = 0;
	int64_t layerCount;
	int64_t* layerSize = 0;
	int64_t* actFunctions = 0;
	float learningRate;
	float momentum;

public:
	NeuralNetwork(int64_t layerCount, int64_t* layerSize, int64_t* actFunctions, float learningRate, float momentum);
	NeuralNetwork(std::string saveFile);
	~NeuralNetwork();
	int64_t save(std::string saveFile);
	int64_t getResultNode(int64_t node);
	void calc(int64_t* inputValues);
	void train(int64_t* inputValues, int64_t* outputValues);
	int64_t getOutputSize();
	int64_t getInputSize();

};