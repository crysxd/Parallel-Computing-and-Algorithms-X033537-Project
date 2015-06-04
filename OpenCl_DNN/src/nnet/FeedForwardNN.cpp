/*
 * FeedForwardNN.cpp
 *
 *  Created on: Apr 30, 2015
 *      Author: richman
 */

#include "FeedForwardNN.h"


void FeedForwardNN::feedforward(doubvec& in, doubvec* out) {
//	Init weights and biases
	this->init();


	for(int i = 0; i < this->hiddenlayers.size();i++ ){
//		this->hiddenlayers[i].propagate(in);
	}

}

void FeedForwardNN::addHiddenLayer(const HiddenLayer layer) {
	this->hiddenlayers.push_back(layer);
}

void FeedForwardNN::backpropagate(doubvec* out_diff, doubvec* in_diff) {
}


FeedForwardNN::FeedForwardNN(u_int32_t indim, u_int32_t outdim) {
}

void FeedForwardNN::init() {

	assert(this->hiddenlayers.size() > 0);
	int i = 1;

	double bias = util::randfloat(this->min_weight,this->max_weight);
//	util::randinit(this->min_weight,this->max_weight,v);
	this->biases.push_back(bias);
//	this->weights.push_back(v);
//	for(; i < this->hiddenlayers.size();i++ ){
//		std::vector<float> v(this->hiddenlayers[i-1].getDim()*this->hiddenlayers[i].getDim());
//		this->weights.push_back(v);
//
//	}
//	std::vector<float> v(this->hiddenlayers[i].getDim() * this->_out_dim);
//	this->weights.push_back(v);
}
