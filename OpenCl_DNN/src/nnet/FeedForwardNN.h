/*
 * FeedForwardNN.h
 *
 *  Created on: Apr 30, 2015
 *      Author: richman
 */

#ifndef NNET_FEEDFORWARDNN_H_
#define NNET_FEEDFORWARDNN_H_

#include <string>
#include <vector>
#include "HiddenLayer.h"
#include <algorithm>
#include <random>
#include "CLMatrix.hpp"
#include <iostream>
#include "MSE.h"
#include <cassert>
#include <memory>

typedef CL_Matrix<float> Matrix;

class FeedForwardNN {
public:
	FeedForwardNN(u_int32_t indim, u_int32_t outdim,float lrate);
	FeedForwardNN(u_int32_t indim, u_int32_t outdim,std::vector<u_int32_t> hid_dims,float lrate);

	FeedForwardNN(u_int32_t indim, u_int32_t outdim,float lrate, std::vector<std::pair<Matrix,Matrix>> weight_biases);

	virtual ~FeedForwardNN();

	/** Adds a hidden layer with neuron- neurons to the network*/
	void addHiddenLayer(const u_int32_t neurons);
	/** Adds an activation function to the network */
	void addActivation(Activation* activation);

	Matrix feedforward(Matrix &in,bool learn);

	Matrix test(Matrix&in);

	std::vector<float> trainbatch(Matrix &in, Matrix &target);
// Helper functions to get the strides

	std::vector<float> trainsgd(Matrix &in, Matrix &target);

	std::vector<std::pair<Matrix,Matrix>> backpropagate(Matrix &error);

	void setCost(Cost *c){
		this->_costfunc = c;
	}

	const std::vector<std::pair<Matrix, Matrix> >& getWeightBiases() const {
		return _weight_biases;
	}


private:

	void init();



// Learning rate of the N
//	It is used for the update, where it is defined as:

	float _l_rate;
// Momentum
//	velocity = momentum * velocity - learning_rate * gradient
//	params = params + velocity
//	Initialize velocity as 0 and then store the last gradient in it
	float _momentum;
// Number of output dimension
	u_int32_t _out_dim;
//Number of input neurons
	u_int32_t _in_dim;
// Array indicating which activations for each layer we have
//	const std::vector<double (*)(double)> *activations;
	u_int32_t _net_size;
//	std::vector<HiddenLayer> hiddenlayers;

	std::vector<u_int32_t> _hid_dims;

	std::vector<std::reference_wrapper<Activation>> _activations;

	std::vector<std::pair<Matrix,Matrix>> _weight_biases;

//	Derivatives stored for the backpropagation
	std::vector<Matrix> _deriv;
// Buffer for
	std::vector<Matrix> _backprop_buf;

//	The costfunction used to calculate the target loss and backpropagation
	Cost* _costfunc;



//	unsigned int _seed;

//	Range of the weights for random init
//	int min_weight = -1;
////	Max weight when init randomly
//	int max_weight = 1;

};

#endif /* NNET_FEEDFORWARDNN_H_ */
