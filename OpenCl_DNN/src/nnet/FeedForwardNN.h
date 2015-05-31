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
#include "../hiddenlayer/HiddenLayer.h"

typedef std::vector<double> doubvec;

class FeedForwardNN {
public:
	FeedForwardNN();
	virtual ~FeedForwardNN();

	void addHiddenLayer(HiddenLayer &layer);

	void feedforward(doubvec &in, doubvec *out);
	void backpropagate(doubvec *out_diff, doubvec *in_diff);

private:
// Learning rate of the N
//	It is used for the update, where it is defined as:

	double _l_rate;
// Momentum
//	velocity = momentum * velocity - learning_rate * gradient
//	params = params + velocity
//	Initialize velocity as 0 and then store the last gradient in it
	double _momentum;
//	Hidden layer dimension
	int _hid_dim;
// Number of hidden layers
	int _hid_num;
// Number of output dimension
	int _out_dim;
// Array indicating which activations for each layer we have
	const std::vector<double (*)(double)> *activations;

	const std::vector<HiddenLayer> *hiddenlayers;

};

#endif /* NNET_FEEDFORWARDNN_H_ */
