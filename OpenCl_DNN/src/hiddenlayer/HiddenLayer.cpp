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

HiddenLayer::HiddenLayer(Activation& activation, u_int32_t dim):dim(dim),activation(&activation) {
}


void HiddenLayer::propagate(Matrix *input,Matrix &weight, Matrix &bias){

}
