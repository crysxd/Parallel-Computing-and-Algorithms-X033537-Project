#include <iostream>
#include <cstring>
#include <cstdint>
#include <fstream>
#include "CLMatrix.hpp"
#include "FeedForwardNN.h"
#include "Sigmoid.h"

typedef CL_Matrix<float> Matrix;

class NeuralNetwork {
	Matrix result;
	uint64_t layerCount;
	uint64_t* layerSize = 0;
	uint64_t* actFunctions = 0;
	float learningRate;
	float momentum;
	FeedForwardNN* network = 0;
    std::vector<float> lastErrors;
	void fillMatrixFromNumpy(Matrix &matrix, float* inputValues, int rowLength, int rowCount);
    Sigmoid sigmoid;

public:
	NeuralNetwork(uint64_t layerCount, uint64_t* layerSize, uint64_t* actFunctions, float learningRate, float momentum);
	NeuralNetwork(std::string saveFile);
	~NeuralNetwork();
	uint64_t save(std::string saveFile);
	double getResultNode(uint64_t node);
	void test(float* inputValues, int rowLength, int rowCount, float *resultOut[], int *resultRows, int *resultCols);
	void train(float* inputValues, float* outputValues, int inputRowLength, int outputRowLength, int rowCount, float *errorsOut[], int *errorsLen);
	uint64_t getOutputSize();
	uint64_t getInputSize();
	void readMatTest(float *out[], int *rows, int *cols);

};
