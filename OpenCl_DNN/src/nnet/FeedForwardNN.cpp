/*
 * FeedForwardNN.cpp
 *
 *  Created on: Apr 30, 2015
 *      Author: richman
 */

#include "FeedForwardNN.h"
#include <functional>

FeedForwardNN::FeedForwardNN(){
}

FeedForwardNN::~FeedForwardNN() {
	// TODO Auto-generated destructor stub
}

void FeedForwardNN::feedforward(doubvec& in, doubvec* out) {
	for(int i = 0; i < this->hiddenlayers->size();i++ ){
		this->hiddenlayers[i].propagate()
	}
}

void FeedForwardNN::addHiddenLayer(HiddenLayer& layer) {
	this->hiddenlayers->push_back(layer);

}

void FeedForwardNN::backpropagate(doubvec* out_diff, doubvec* in_diff) {
}
