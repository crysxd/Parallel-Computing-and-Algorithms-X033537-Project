/*
 * FeedForwardNN.cpp
 *
 *  Created on: Apr 30, 2015
 *      Author: richman
 */

#include "FeedForwardNN.h"

#ifndef DEBUG
    #define DEBUG 1
#endif

Matrix FeedForwardNN::feedforward(Matrix& in) {
//	Init weights and biases

	Matrix &tmpin = in;
// Plain forward propagate the weights using the given activation function
// Store the activations and the gradients for the backpropagation later
	auto i=0u;
//Append the input layer as the first layer into the buffer;
	this->_backprop_buf.push_back(tmpin);

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
	// For backpropagation we use the following rule:
	// first estimate the output error, by calculating the derivative D ( aka _deriv) times the error
	// Which is equal to out * (1-out) * (y-t)
	//
	// Then we backpropagate with:
	// \delta_i = D_i * W_{i+1}.\delta_{i+1}
	// Where . is the dot product and * is the element wise multiplication
	// Finally we update by using gradient descrent:
	//	W_i = W_i - l_rate * \delta_i.
//	Last layer
	Matrix delta_l = (error * this->_deriv.back());
#if !DEBUG
	std::cout << "Updating layer " << this->_weight_biases.size()-1 << " with dimensions : ";
	this->_weight_biases.back().first.printDimension();
	std::cout << endl;
#endif
	Matrix const &nabla =  this->_backprop_buf[this->_backprop_buf.size()-2].dot(delta_l.transpose());
//	Update last layer weights
	this->_weight_biases.back().first -= this->_l_rate * nabla.transpose();
//	Update last bias
	this->_weight_biases.back().second -= this->_l_rate * delta_l;
	for(int i=this->_activations.size()-2;i >= 0;i--){
#if !DEBUG
		std::cout << "Updating layer" << i << " with dimensions : ";
		this->_weight_biases[i+1].first.printDimension();
		std::cout << std::endl;
#endif
		delta_l = this->_deriv[i]*(this->_weight_biases[i+1].first.transpose().dot(delta_l));
		// Note that backprop_buf has size of L = number of layers, whereas activations are L-1!
		// Therefore _backprop_buf[i] refers to the last layer not the current one, thought weight_biases[i]
		// refers to the current layer weight!
		Matrix const &nabla_back = this->_backprop_buf[i].dot(delta_l.transpose());
		//	Update the layer weights
		this->_weight_biases[i].first -= this->_l_rate * nabla_back.transpose();
		// Update the bias
		this->_weight_biases[i].second -= this->_l_rate * delta_l;
	}


}


FeedForwardNN::FeedForwardNN(u_int32_t indim, u_int32_t outdim, float lrate):
		_in_dim(indim),_out_dim(outdim),_l_rate(lrate),_costfunc(new MSE()) {

}

FeedForwardNN::~FeedForwardNN() {
}

void FeedForwardNN::trainbatch(Matrix &in, Matrix &target) {
	if(in.getRows() != this->_in_dim){
		std::cerr<< "Input dimensions and datainput dimensions do not match. Expected : " << this->_in_dim
				<< " , but got " << in.getRows() << " in the matrix";
		return;
	}
	if(target.getRows() != this->_out_dim){
		std::cerr<< "Input dimensions and datainput dimensions do not match. Expected : " << this->_out_dim
						<< " , but got " << target.getRows() << " in the matrix";
		return;
	}
// Init the weights and other variables
	this->init();
	for(auto i=0u; i < 5;i++){
//	Using We assume the the input has N independent column vectors
		for(auto i=0u; i < in.getCols();i++){
			Matrix inputvector = in.subMatCol(i);
			Matrix const &predict = this->feedforward(inputvector);
			Matrix error = (target - predict);
			std::cout << "Training Error " << error;
			std::cout << "Output "<< predict;
			this->backpropagate(error);
		}

	}


}

void FeedForwardNN::addHiddenLayer(const u_int32_t neurons) {
	this->_hid_dims.push_back(neurons);
}

void FeedForwardNN::addActivation(Activation* activation) {
	this->_activations.push_back(*activation);
}

FeedForwardNN::FeedForwardNN(u_int32_t indim, u_int32_t outdim,
		std::vector<u_int32_t> hid_dims, float lrate):FeedForwardNN(indim,outdim,lrate) {
	_hid_dims = hid_dims;

}

void FeedForwardNN::init() {

	assert(this->_hid_dims.size() > 0);
	assert(this->_activations.size()>0);
	assert(this->_activations.size() == this->_hid_dims.size()+ 1);
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

