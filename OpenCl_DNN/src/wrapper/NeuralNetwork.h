#include <iostream>
#include <cstring>
#include <cstdint>
#include <fstream>
#include "CLMatrix.hpp"
#include "FeedForwardNN.h"
#include "Sigmoid.h"
#include "TanH.h"

#define ACTIVATION_TAN_H 0
#define ACTIONATION_SIGMOID 1

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
	void fillMatrixFromNumpy(Matrix &matrix, float* numpy, int shape0, int shape1, int strides0, int strides1);
    Sigmoid sigmoid;
    TanH tanH;
    void initNetwork();

public:
	NeuralNetwork(uint64_t layerCount, uint64_t* layerSize, uint64_t* actFunctions, float learningRate, float momentum);
	NeuralNetwork(std::string saveFile);
	~NeuralNetwork();
	uint64_t save(std::string saveFile);
	double getResultNode(uint64_t node);
	void test(float* inputValues, int shape0, int shape1, int strides0, int strides1, float *resultOut[], int *resultRows, int *resultCols);
	void train(float* inputValues, int inShape0, int inShape1, int inStrides0, int inStrides1, float* outputValues, int outShape0, int outShape1, int outStrides0, int outStrides1, float *errorsOut[], int *errorsLen);
	void trainsgd(float* inputValues, int inShape0, int inShape1, int inStrides0, int inStrides1, float* outputValues, int outShape0, int outShape1, int outStrides0, int outStrides1, float *errorsOut[], int *errorsLen);
	uint64_t getOutputSize();
	uint64_t getInputSize();
	void readMatTest(float *out[], int *rows, int *cols);
	Matrix* NeuralNetwork::readMatrix(std:ifstream &s);
	void writeMatrix(Matrix &m, std::ofstream &s);

};
