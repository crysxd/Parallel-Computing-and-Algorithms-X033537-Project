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


#define NUM_EPOCHS 100

#define MINI_BATCH_SIZE 10
Matrix FeedForwardNN::feedforward(Matrix& in,bool learn) {

//	Init weights and biases

	Matrix &tmpin = in;
// Plain forward propagate the weights using the given activation function
// Store the activations and the gradients for the backpropagation later
	auto i=0u;
//Append the input layer as the first layer into the buffer;
	this->_backprop_buf.at(0)=tmpin;
	Matrix &netout= tmpin;
	for(; i < this->_activations.size();i++ ){
		Activation& activ = _activations[i];
		Matrix const &weights = this->_weight_biases[i].first;
		Matrix const &bias = this->_weight_biases[i].second;
//		Calculate output of the current layer
//		Apply activation
		netout = weights.dot(tmpin) + bias;
		tmpin = activ.propagate(netout);
//		If non learning mode, we dont need to store anything, just propagate through
		if (learn){
	//		Store the derivatives of the layers, for backprop
			this->_deriv.at(i) = activ.grad(netout);
	//		Store the output of every layer for the backpropagation, except the last one
			if (i+1 < this->_activations.size()){
				this->_backprop_buf.at(i+1) = tmpin;
			}

		}
	}
//	Return the output of the network
	return tmpin;
}

std::vector<std::pair<Matrix,Matrix>> FeedForwardNN::backpropagate(Matrix &error) {
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
	std::vector<std::pair<Matrix,Matrix>> nablas;

	Matrix delta_l = (error * this->_deriv.back());

#if !DEBUG
	std::cout << "Updating layer " << this->_weight_biases.size()-1 << " with dimensions : ";
	this->_weight_biases.back().first.printDimension();
	std::cout << endl;
#endif
// Get the last hidden layer from the back_propbuf
	Matrix const &nabla =  this->_backprop_buf.back().dot(delta_l.transpose());

	nablas.push_back(std::make_pair(nabla.transpose(),delta_l));
	for(int i=this->_net_size-2;i >= 0;i--){
#if !DEBUG
		std::cout << "Updating layer " << i << " with dimensions : ";
		this->_weight_biases[i+1].first.printDimension();
		std::cout << std::endl;
#endif
		delta_l = this->_deriv.at(i)*(this->_weight_biases[i+1].first.transpose().dot(delta_l));
		// Note that backprop_buf has size of L = number of layers, whereas activations are L-1!
		// Therefore _backprop_buf[i] refers to the last layer not the current one, thought weight_biases[i]
		// refers to the current layer weight!
		Matrix const &nabla_back = this->_backprop_buf.at(i).dot(delta_l.transpose());
		//	Update the layer weights

		nablas.push_back(std::make_pair(nabla_back.transpose(),delta_l));

	}
	return nablas;


}


FeedForwardNN::FeedForwardNN(u_int32_t indim, u_int32_t outdim, float lrate):
		_in_dim(indim),_out_dim(outdim),_l_rate(lrate),_costfunc(new MSE()),_momentum(0.) {

}

FeedForwardNN::~FeedForwardNN() {
}



std::vector<float> FeedForwardNN::trainbatch(Matrix &in, Matrix &target) {
    std::cout << "train " << in.getRows() << 'x' << in.getCols() << " -> " << target.getRows() << 'x' << target.getCols() << '\n';
	// trains in batch gradient descent.
	// Input is a N x M matrix, where the rows represent the size of the input layer and the cols the amount
	// of data we have.
	// E.g the matrix (5,1000), has a 5 dimensional input and 1000 samples.
	std::vector<float> errors;


	if(in.getRows() != this->_in_dim){
		std::cerr<< "Input dimensions and datainput dimensions do not match. Expected : " << this->_in_dim
				<< " , but got " << in.getRows() << " in the matrix";
		return errors;
	}
	if(target.getRows() != this->_out_dim){
		std::cerr<< "Target dimensions and output dimensions do not match. Expected : " << this->_out_dim
						<< " , but got " << target.getRows() << " in the matrix";
		return errors;
	}

	float l_rate = 1./in.getCols() * this->_l_rate;

	// Init the weights and other variables
	this->init();
	// Begin running the neural network for NUM_EPOCHS iterations
	for(auto epoch=0u; epoch < NUM_EPOCHS ;epoch++){
		double epoch_error = 0;
		std::vector<std::pair<Matrix,Matrix>> w_b;
		std::vector<Matrix> momentumbuf;
		for(int i=this->_weight_biases.size()-1; i>=0;i--){
			//Init the weights and biases for this epoch with zero
//			Accumulators should be zero
			Matrix weight(this->_weight_biases[i].first.getRows(),this->_weight_biases[i].first.getCols(),0.f);

			Matrix bias (this->_weight_biases[i].second.getRows(),this->_weight_biases[i].second.getCols(),0.f);

			Matrix mombuf(this->_weight_biases[i].first.getRows(),this->_weight_biases[i].first.getCols(),0.f);
			w_b.push_back(
					std::make_pair(
							weight,bias
					));

			momentumbuf.push_back(mombuf);
		}

//	Using We assume the the input has N independent column vectors
		for(auto i=0u; i < in.getCols();i++){
            std::cout << "epoch: " << epoch << ", i: " << i << "\n";
			// Get the column of the input and use it as input
			Matrix inputvector = in.subMatCol(i);
			////////////////////////////////////////////////////////////
			// Feed forward step, returns the predictions of the nnet //
			////////////////////////////////////////////////////////////
			Matrix const &predict = this->feedforward(inputvector,true);
			Matrix error = (predict - target.subMatCol(i));
			epoch_error+= error.transpose().dot(error);
			std::cout << " Predict \n" << predict << "\nTarget: \n"<< target.subMatCol(i) << std::endl;
			///////////////////////////////
			// Backpropagate the errors  //
			///////////////////////////////
			std::vector<std::pair<Matrix,Matrix>> const &delta_w_b = this->backpropagate(error);
			//We got the weights, so just update the non accumulated ones
			for(int wi=this->_weight_biases.size()-1; wi>=0;wi--){
				w_b[wi].first += delta_w_b[wi].first;
				w_b[wi].second += delta_w_b[wi].second;
			}
		}
		// Print out the result
// #if !DEBUG
		std::cout << "Epoch " << epoch +1 << " Error " << epoch_error << '\n';
// #endif
		errors.push_back(epoch_error);

		/////////////////////////
		// Update the weights //
		/////////////////////////
		for(int i=this->_weight_biases.size()-1; i>=0;i--){
			this->_weight_biases.at(i).first -= l_rate * w_b[w_b.size()-i-1].first + this->_momentum * momentumbuf.at(w_b.size()-i-1);
//			this->_weight_biases.at(i).first -= this->_l_rate * w_b[w_b.size()-i-1].first;
			this->_weight_biases.at(i).second -= l_rate * w_b[w_b.size()-i-1].second;
//		Momentum is defined as delta w_i+1 = w_i - lrate*nabla_w + momentum * delta w_i(t)
//			momentumbuf.at(w_b.size()-i-1) = this->_weight_biases[i].first;
		}


	}

	return errors;


}

void FeedForwardNN::addHiddenLayer(const u_int32_t neurons) {
	this->_hid_dims.push_back(neurons);
}

void FeedForwardNN::addActivation(Activation* activation) {
	this->_activations.push_back(*activation);
}


Matrix FeedForwardNN::test(Matrix& in) {
//	Just run a ffwd and return result
	assert(in.getRows() == _in_dim);
	Matrix predictions(this->_out_dim, in.getCols());
    for(auto i=0u; i < in.getCols();i++){
        // Get the column of the input and use it as input
        Matrix inputvector = in.subMatCol(i);
        std::cout << "test feedforward " << inputvector.getRows() << 'x' << inputvector.getCols() << '\n';
//		Do not train the network
        Matrix const &predict = this->feedforward(inputvector, false);
        for (int j =0; j < this->_out_dim; j++)
            predictions.fillAt(j, i, predict(j, 0));
	}
	return predictions;

}

FeedForwardNN::FeedForwardNN(u_int32_t indim, u_int32_t outdim, float lrate,
		std::vector<std::pair<Matrix, Matrix> > weight_biases):FeedForwardNN(indim,outdim,lrate) {
	this->_weight_biases = weight_biases;
}

std::vector<float> FeedForwardNN::trainsgd(Matrix& in, Matrix& target) {
	std::cout << "train " << in.getRows() << 'x' << in.getCols() << " -> " << target.getRows() << 'x' << target.getCols() << '\n';
	// trains in batch gradient descent.
	// Input is a N x M matrix, where the rows represent the size of the input layer and the cols the amount
	// of data we have.
	// E.g the matrix (5,1000), has a 5 dimensional input and 1000 samples.
	std::vector<float> errors;
	if(in.getRows() != this->_in_dim){
		std::cerr<< "Input dimensions and datainput dimensions do not match. Expected : " << this->_in_dim
				<< " , but got " << in.getRows() << " in the matrix";
		return errors;
	}
	if(target.getRows() != this->_out_dim){
		std::cerr<< "Target dimensions and output dimensions do not match. Expected : " << this->_out_dim
						<< " , but got " << target.getRows() << " in the matrix";
		return errors;
	}
	// Init the weights and other variables
	this->init();

	std::vector<std::pair<Matrix,Matrix>> w_b;
	std::vector<Matrix> momentumbuf;
	for(int i=this->_weight_biases.size()-1; i>=0;i--){
		//Init the weights and biases for this epoch with zero
		Matrix weight(this->_weight_biases[i].first.getRows(),this->_weight_biases[i].first.getCols(),0.f);

		Matrix bias (this->_weight_biases[i].second.getRows(),this->_weight_biases[i].second.getCols(),0.f);

		Matrix mombuf(this->_weight_biases[i].first.getRows(),this->_weight_biases[i].first.getCols(),0.f);
		w_b.push_back(
				std::make_pair(
						weight,bias
				));

		momentumbuf.push_back(mombuf);
	}

	// Begin running the neural network for NUM_EPOCHS iterations
	for(auto epoch=0u; epoch < NUM_EPOCHS ;epoch++){
		double epoch_error = 0;
//		Reset the buffers
		for(auto i = 0u ; i < momentumbuf.size();i++){
			momentumbuf.at(i).zeros();
			w_b.at(i).first.zeros();
			w_b.at(i).second.zeros();
		}

//	Using We assume the the input has N independent column vectors
		for(auto i=0u; i < in.getCols();i+= MINI_BATCH_SIZE){
			std::cout << "epoch: " << epoch << ", i: " << i << "\n";
			// Get the column of the input and use it as input

			for (auto j=i; j < min(i+MINI_BATCH_SIZE,in.getCols()); j++){
				Matrix inputvector = in.subMatCol(i);
				////////////////////////////////////////////////////////////
				// Feed forward step, returns the predictions of the nnet //
				////////////////////////////////////////////////////////////
				Matrix const &predict = this->feedforward(inputvector,true);
				Matrix error = predict - target.subMatCol(i);


				epoch_error+= error.transpose().dot(error);
				std::cout << predict << "Error "<< epoch_error <<  std::endl;
				///////////////////////////////
				// Backpropagate the errors  //
				///////////////////////////////
				std::vector<std::pair<Matrix,Matrix>> const &delta_w_b = this->backpropagate(error);
				//We got the weights, so just update the non accumulated ones
				for(int i=this->_weight_biases.size()-1; i>=0;i--){
					w_b[i].first += delta_w_b[i].first;
					w_b[i].second += delta_w_b[i].second;
				}
			}
			// Print out the result
	// #if !DEBUG
			std::cout << "Epoch " << epoch +1 << " Error " << epoch_error << '\n';
	// #endif
			errors.push_back(epoch_error);

			/////////////////////////
			// Update the weights //
			/////////////////////////
			for(int i=this->_weight_biases.size()-1; i>=0;i--){
				this->_weight_biases[i].first -= this->_l_rate * 1.f/MINI_BATCH_SIZE * w_b[w_b.size()-i-1].first + this->_momentum * momentumbuf.at(w_b.size()-i-1);
				this->_weight_biases[i].second -= this->_l_rate * 1.f/MINI_BATCH_SIZE * w_b[w_b.size()-i-1].second;

	//		Momentum is defined as delta w_i+1 = w_i - lrate*nabla_w + momentum * delta w_i(t)
				momentumbuf.at(w_b.size()-i-1) = this->_weight_biases[i].first;
			}
		}
	}
	return errors;
}

FeedForwardNN::FeedForwardNN(u_int32_t indim, u_int32_t outdim, float lrate,
		float momentum):FeedForwardNN(indim,outdim,lrate) {
	_momentum = momentum;
}

void FeedForwardNN::init() {

	assert(this->_hid_dims.size() > 0);
	assert(this->_activations.size()>0);
//    std::cout << this->_activations.size() << ' ' << this->_hid_dims.size() << '\n';
	assert(this->_activations.size() == this->_hid_dims.size()+ 1);

	this->_net_size = this->_activations.size();
//	First layer is initialized independently
//	Using rvalue references
	auto i = 0u;
	this->_weight_biases.push_back(std::make_pair(
			Matrix(this->_hid_dims[0],this->_in_dim,true),
			Matrix(this->_hid_dims[0],1,true)));
//	Initialize the buffers for the derivaties and the output of each layer
	this->_deriv.push_back(Matrix(this->_hid_dims[0],1));
	this->_backprop_buf.push_back(Matrix(this->_hid_dims[0],1));

	for(;i < this->_hid_dims.size()-1;i++){
		this->_weight_biases.push_back(std::make_pair(
			Matrix(this->_hid_dims[i+1],this->_hid_dims[i],true)
			,
			Matrix(this->_hid_dims[i+1],1,true)
		));
		this->_deriv.push_back(Matrix(this->_hid_dims[i+1],1));
		this->_backprop_buf.push_back(Matrix(this->_hid_dims[i+1],1));
	}

	this->_backprop_buf.push_back(Matrix(this->_out_dim,1));
	this->_deriv.push_back(Matrix(this->_out_dim,1));
//	Add for the last layer the output layer
	this->_weight_biases.push_back(std::make_pair(
			Matrix(this->_out_dim,this->_hid_dims[i],true),
			Matrix(this->_out_dim,1,true)
			)
	);

}
