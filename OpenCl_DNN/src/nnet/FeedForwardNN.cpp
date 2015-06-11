/*
 * FeedForwardNN.cpp
 *
 *  Created on: Apr 30, 2015
 *      Author: richman
 */

#include "FeedForwardNN.h"


Matrix FeedForwardNN::feedforward(Matrix& in) {
//	Init weights and biases

	this->init();
	Matrix &tmpin = in;

	auto i=0u;

	for(; i < this->_activations.size();i++ ){
		Activation& activ = _activations[i];
		Matrix const &weights = this->_weight_biases[i].first;
		Matrix const &bias = this->_weight_biases[i].second;
//		Calculate output of the current layer
//		Apply activation
		tmpin = activ.propagate(weights.dot(tmpin) + bias);
//		Store the derivatives of the layers, for backprop
		this->_deriv.push_back(activ.grad(tmpin));
//		Store the output of every later for the backpropagation
		this->_backprop_buf.push_back(tmpin);
	}

//	Return the output of the network
	return tmpin;
}

void FeedForwardNN::backpropagate(Matrix &error) {

	std::vector<Matrix> prop_erros;
	Matrix &tmpin = error;
//	Last layer
	Matrix nabla_l = this->_backprop_buf[this->_backprop_buf.size()-2].dot((error * this->_deriv.back()).transpose());
//	std::cout << this->_weight_biases[this->_backprop_buf.size()-1].first;
//	std::cout << this->_backprop_buf[this->_backprop_buf.size()-2];
//	std::cout << std::endl;
//	std::cout << (error * this->_deriv.back());
//	for(auto i=this->_activations.size()-2;i >= 0;i--){
//		this->_backprop_buf[i].printDimension();
//		tmpin.printDimension();
//		this->_deriv[i].printDimension();
//		tmpin = this->_deriv[i]*this->_backprop_buf[i].dot(tmpin);
//		prop_erros.push_back(tmpin);
//	}

//	Update weights
//	for(auto i=0u,i<this->_weight_biases.size();i++){
//
//	}

}


FeedForwardNN::FeedForwardNN(u_int32_t indim, u_int32_t outdim, double lrate):
		_in_dim(indim),_out_dim(outdim),_l_rate(lrate),_costfunc(new MSE()) {
}

FeedForwardNN::~FeedForwardNN() {
}

void FeedForwardNN::trainbatch(Matrix &in, Matrix &target) {
	Matrix const &predict = this->feedforward(in);
	Matrix error = (target - predict);

	this->backpropagate(error);

}

void FeedForwardNN::addHiddenLayer(const u_int32_t neurons) {
	this->_hid_dims.push_back(neurons);
}

void FeedForwardNN::addActivation(Activation* activation) {
	this->_activations.push_back(*activation);
}

FeedForwardNN::FeedForwardNN(u_int32_t indim, u_int32_t outdim,
		std::vector<u_int32_t> hid_dims, double lrate):FeedForwardNN(indim,outdim,lrate) {
	_hid_dims = hid_dims;
}

void FeedForwardNN::init() {

	assert(this->_hid_dims.size() > 0);
	assert(this->_activations.size()>0);
	assert(this->_activations.size() + this->_hid_dims.size()+ 1);

//	First layer is initialized independently
//	Using rvalue references
	auto i = 0u;
	this->_weight_biases.push_back(std::make_pair(
			Matrix(this->_hid_dims[0],this->_in_dim,true),
			Matrix(this->_hid_dims[0],1,true)));
	for(;i < this->_hid_dims.size()-1;i++){
		this->_weight_biases.push_back(std::make_pair(
			Matrix(this->_hid_dims[i+1],this->_hid_dims[i],true)
			,
			Matrix(this->_hid_dims[i+1],1,true)
		));
	}
//	Add for the last layer the output layer
	this->_weight_biases.push_back(std::make_pair(
			Matrix(this->_out_dim,this->_hid_dims[i],true),
			Matrix(this->_out_dim,1,true)
			)
	);




}
