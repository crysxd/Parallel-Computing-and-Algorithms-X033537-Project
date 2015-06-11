/*
 * HiddenLayer.cpp
 *
 *  Created on: Apr 30, 2015
 *      Author: richman
 */

#include "HiddenLayer.h"


HiddenLayer::~HiddenLayer() {
	// TODO Auto-generated destructor stub
}

HiddenLayer::HiddenLayer(Activation& activation, u_int32_t dim):activation(&activation),dim(dim) {
}


Matrix HiddenLayer::propagate(const Matrix &input,const Matrix &weight,const Matrix &bias){
	Matrix dot = weight.dot(input) + bias;
	return activation->propagate(dot);
}

Matrix HiddenLayer::grad(const Matrix& layer) {
	return layer;
}
