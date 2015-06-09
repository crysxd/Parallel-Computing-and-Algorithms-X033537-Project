/*
 * FeedForwardNN.cpp
 *
 *  Created on: Apr 30, 2015
 *      Author: richman
 */

#include "FeedForwardNN.h"


void FeedForwardNN::feedforward(Matrix& in, Matrix* out) {
//	Init weights and biases

	this->init(in);
	Matrix tmpin = in;

	for(int i = 0; i < this->hiddenlayers.size();i++ ){
		this->hiddenlayers[i].propagate(
				&tmpin,
				this->_weight_biases[i].first,
				this->_weight_biases[i].second);
	}

}

void FeedForwardNN::addHiddenLayer(const HiddenLayer layer) {
	this->hiddenlayers.push_back(layer);
}

void FeedForwardNN::backpropagate(Matrix* out_diff, Matrix* in_diff) {
}


FeedForwardNN::FeedForwardNN(u_int32_t indim, u_int32_t outdim, double lrate):_in_dim(indim),_out_dim(outdim),_l_rate(lrate) {
}

void FeedForwardNN::init(Matrix &inputdata) {

	assert(this->hiddenlayers.size() > 0);

//	First layer is initialized independently
//	Matrix firstlayer(this->hiddenlayers[0].getDim(),this->_in_dim);
//	Matrix firstbias(this->hiddenlayers[0].getDim(),1);
//	Using rvalue references
	this->_weight_biases.push_back(std::make_pair(Matrix(this->hiddenlayers[0].getDim(),this->_in_dim),
			Matrix(this->hiddenlayers[0].getDim(),1)));
	for(auto i=0u; i < this->hiddenlayers.size();i++){
		this->_weight_biases.push_back(std::make_pair(
			Matrix(this->hiddenlayers[i+1].getDim(),this->hiddenlayers[i].getDim())
			,
			Matrix(this->hiddenlayers[i+1].getDim(),1)
		));
	}

}
