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
#include <iostream>
#include <cassert>

typedef std::vector<double> doubvec;
typedef std::vector<std::vector<double>> doubmat;

class FeedForwardNN {
public:
	FeedForwardNN(u_int32_t indim, u_int32_t outdim);
	virtual ~FeedForwardNN();

	void addHiddenLayer(const HiddenLayer layer);


	void feedforward(doubvec &in, doubvec *out);
	void backpropagate(doubvec *out_diff, doubvec *in_diff);

private:

	void init();

// Learning rate of the N
//	It is used for the update, where it is defined as:

	double _l_rate;
// Momentum
//	velocity = momentum * velocity - learning_rate * gradient
//	params = params + velocity
//	Initialize velocity as 0 and then store the last gradient in it
	double _momentum;
//	Hidden layer dimension
	u_int32_t _hid_dim;
// Number of hidden layers
	u_int32_t _hid_num;
// Number of output dimension
	u_int32_t _out_dim;
//Number of input neurons
	u_int32_t _in_dim;
// Array indicating which activations for each layer we have
	const std::vector<double (*)(double)> *activations;

	std::vector<HiddenLayer> hiddenlayers;

	std::vector<doubmat> weights;

	std::vector<float> biases;

	unsigned int _seed;

//	Range of the weights for random init
	int min_weight = -1;
//	Max weight when init randomly
	int max_weight = 1;

};

#endif /* NNET_FEEDFORWARDNN_H_ */
