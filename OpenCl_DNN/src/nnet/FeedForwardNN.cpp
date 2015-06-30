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

Matrix FeedForwardNN::feedforward(Matrix& in,bool learn) {

//	Init weights and biases

	Matrix &tmpin = in;
// Plain forward propagate the weights using the given activation function
// Store the activations and the gradients for the backpropagation later
	auto i=0u;
//Append the input layer as the first layer into the buffer;
	if (learn){
// Only in case of learning, we need to consider the buffer sizes
		assert(this->_backprop_buf.size() > 0);
		assert(this->_activations.size()>0);
		assert(this->_weight_biases.size() == this->_activations.size() );
		this->_backprop_buf.at(i)=tmpin;
	}
	Matrix &netout= tmpin;
	assert(this->_weight_biases.size()>0);
	for(; i < this->_weight_biases.size();i++ ){
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

	assert(this->_deriv.size() == this->_net_size);
	assert(error.getRows() == this->_deriv.back().getRows());

	Matrix delta_l = error * (this->_deriv.back());
#if !DEBUG
	std::cout << "Updating layer " << this->_weight_biases.size()-1 << " with dimensions : ";
	this->_weight_biases.back().first.printDimension();
	std::cout << endl;
#endif
// Get the last hidden layer from the back_propbuf
	Matrix const &nabla =  this->_backprop_buf.back().dot(delta_l.transpose());

	nablas.push_back(std::make_pair(nabla.transpose(),delta_l));
	for(auto i=2u;i <= this->_net_size;i++){

		auto idx = this->_net_size-i;
#if !DEBUG
		std::cout << "Updating layer " << idx << " with dimensions : ";
		this->_weight_biases[idx+1].first.printDimension();
		std::cout << std::endl;
#endif
		delta_l = (this->_weight_biases[idx+1].first.transpose().dot(delta_l))*this->_deriv.at(idx);
		// Note that backprop_buf has size of L = number of layers, whereas activations are L-1!
		// Therefore _backprop_buf[i] refers to the last layer not the current one, thought weight_biases[i]
		// refers to the current layer weight!
		Matrix const &nabla_back = this->_backprop_buf.at(idx).dot(delta_l.transpose());
		//	Update the layer weights
//		this->_weight_biases[i].first -= this->_l_rate * nabla_back.transpose();
		// Update the bias
//		this->_weight_biases[i].second -= this->_l_rate * delta_l;

		nablas.push_back(std::make_pair(nabla_back.transpose(),delta_l));

	}
	return nablas;


}


FeedForwardNN::FeedForwardNN(u_int32_t indim, u_int32_t outdim):
        _in_dim(indim),_out_dim(outdim),_costfunc(new MSE()) {

}

FeedForwardNN::~FeedForwardNN() {
}



std::vector<float> FeedForwardNN::trainbatch(Matrix &in, Matrix &target, float l_rate, float momentum, unsigned int numEpochs) {
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

    l_rate =  l_rate/in.getCols() ;

	// Init the weights and other variables
	this->init();
	// Begin running the neural network for NUM_EPOCHS iterations
	std::vector<std::pair<Matrix,Matrix>> w_b;
	std::vector<Matrix> momentumbuf;
	for(int i=this->_weight_biases.size()-1; i>=0;i--){
		//Init the weights and biases for this epoch with zero
//			Accumulators should be zero
		Matrix weight(this->_weight_biases[i].first.getRows(),this->_weight_biases[i].first.getCols(),0.f);

		Matrix bias (this->_weight_biases[i].second.getRows(),this->_weight_biases[i].second.getCols(),0.f);

		Matrix momentum_buf(this->_weight_biases[i].first.getRows(),this->_weight_biases[i].first.getCols(),0.f);
		w_b.push_back(
				std::make_pair(
						weight,bias
				));

		momentumbuf.push_back(momentum_buf);
    }

    for(auto epoch=0u; epoch < numEpochs ;epoch++){
		double epoch_error = 0;

//		Reset all the buffers
		for(auto i =0u ; i < w_b.size();i++){
			w_b.at(i).first.zeros();
			w_b.at(i).second.zeros();
			momentumbuf.at(i).zeros();
		}

//	Using We assume the the input has N independent column vectors
		for(auto i=0u; i < in.getCols();i++){

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
			assert(this->_weight_biases.size() == w_b.size());
			assert(this->_weight_biases.size() == delta_w_b.size());
			assert(w_b.size() == delta_w_b.size());
//			Weights are in reversed order since there was some trouble using the insert() or any reserve()
			for(int wi=this->_weight_biases.size()-1; wi>=0;wi--){
				assert(wi >= 0);
//				std::cout << " Weights for Layer" << wi << "\n" << this->_weight_biases.at(wi).first << std::endl;
				w_b.at(wi).first += delta_w_b.at(wi).first;
				w_b.at(wi).second += delta_w_b.at(wi).second;
			}

		}

		// Print out the result
// #if !DEBUG
		std::cout << "Epoch " << epoch +1 << " Error " << epoch_error << std::endl;
// #endif
		errors.push_back(epoch_error);


		/////////////////////////
		// Update the weights //
		/////////////////////////
		for(int i=this->_weight_biases.size()-1,j=0; i>=0, j<w_b.size();i--,j++){
			std::cout << " update weight  " << i << " j " <<j << std::endl;
            this->_weight_biases.at(i).first -= l_rate * w_b.at(j).first + momentum * momentumbuf.at(j);
//			this->_weight_biases.at(i).first -= this->_l_rate * w_b[w_b.size()-i-1].first;
			this->_weight_biases.at(i).second -= l_rate * w_b.at(j).second;
//		Momentum is defined as delta w_i+1 = w_i - lrate*nabla_w + momentum * delta w_i(t)
			momentumbuf.at(j) = this->_weight_biases.at(i).first;
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
//        std::cout << "test feedforward " << inputvector.getRows() << 'x' << inputvector.getCols() << '\n';
//		Do not train the network
        Matrix const &predict = this->feedforward(inputvector, false);
//        std::cout << predict.getRows() << " "  << this->_out_dim <<std::endl;
        assert(predict.getRows() == this->_out_dim);
        for (auto j =0u; j < this->_out_dim; j++){
            predictions.fillAt(j, i, predict(j, 0));
        }
	}
	return predictions;

}

FeedForwardNN::FeedForwardNN(u_int32_t indim, u_int32_t outdim,
        std::vector<std::pair<Matrix, Matrix> > weight_biases):FeedForwardNN(indim,outdim) {
	this->_weight_biases = weight_biases;
}

std::vector<float> FeedForwardNN::trainsgd(Matrix& in, Matrix& target, float l_rate, float momentum, int numEpochs, int miniBatchSize) {
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
	// Begin running the neural network for NUM_EPOCHS iterations
	std::vector<std::pair<Matrix,Matrix>> w_b;
	std::vector<Matrix> momentumbuf;

    l_rate = min(l_rate / miniBatchSize, l_rate / in.getCols());
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
    for(auto epoch=0u; epoch < numEpochs ;epoch++){
		double epoch_error = 0;
//		Reset the buffers
		for(auto i = 0u ; i < momentumbuf.size();i++){
			momentumbuf.at(i).zeros();
			w_b.at(i).first.zeros();
			w_b.at(i).second.zeros();
		}

//	Using We assume the the input has N independent column vectors
        for(auto i=0u; i < in.getCols();i+= miniBatchSize){
			// Get the column of the input and use it as input

            for (auto j=i; j < min(i+miniBatchSize,in.getCols()); j++){
				Matrix inputvector = in.subMatCol(j);
				////////////////////////////////////////////////////////////
				// Feed forward step, returns the predictions of the nnet //
				////////////////////////////////////////////////////////////
				Matrix const &predict = this->feedforward(inputvector,true);
				Matrix error = predict - target.subMatCol(j);


				epoch_error += error.transpose().dot(error);
				///////////////////////////////
				// Backpropagate the errors  //
				///////////////////////////////
				std::vector<std::pair<Matrix,Matrix>> const &delta_w_b = this->backpropagate(error);
				//We got the weights, so just update the non accumulated ones
				for(int wi=this->_weight_biases.size()-1; wi>=0;wi--){
	//				std::cout << " Weights for Layer" << wi << "\n" << this->_weight_biases.at(wi).first << std::endl;
					w_b[wi].first += delta_w_b[wi].first;
					w_b[wi].second += delta_w_b[wi].second;
				}
			}
			// Print out the result
	// #if !DEBUG
	// #endif
			errors.push_back(epoch_error);

			/////////////////////////
			// Update the weights //
			/////////////////////////
			for(int i=this->_weight_biases.size()-1,j=0; i>=0, j<w_b.size();i--,j++){
                this->_weight_biases.at(i).first -= l_rate * w_b[j].first + momentum * momentumbuf.at(j);
				this->_weight_biases.at(i).second -= l_rate * w_b[j].second;
	//		Momentum is defined as delta w_i+1 = w_i - lrate*nabla_w + momentum * delta w_i(t)
				momentumbuf.at(j) = this->_weight_biases[i].first;
			}
		}
		std::cout << "Epoch " << epoch +1 << " Error " << epoch_error << '\n';
	}
	return errors;
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

