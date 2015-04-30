/*
 * FeedForwardNN.cpp
 *
 *  Created on: Apr 30, 2015
 *      Author: richman
 */

#include "FeedForwardNN.h"
#include <functional>

FeedForwardNN::FeedForwardNN(int hid):_hid_dim(hid) {
}

FeedForwardNN::~FeedForwardNN() {
	// TODO Auto-generated destructor stub
}

void FeedForwardNN::feedforward(doubvec& in, doubvec* out) {
}

void FeedForwardNN::backpropagate(doubvec* out_diff, doubvec* in_diff) {
}
