/*
 * HiddenLayer.h
 *
 *  Created on: Apr 30, 2015
 *      Author: richman
 */

#ifndef HIDDENLAYER_HIDDENLAYER_H_
#define HIDDENLAYER_HIDDENLAYER_H_

#include "../activations/Activation.h"
#include "../cost/Cost.h"

class HiddenLayer {
public:
	HiddenLayer(Cost *cost, Activation *activation);
	virtual ~HiddenLayer();

private:
	Activation *activation;
	Cost *cost;

};

#endif /* HIDDENLAYER_HIDDENLAYER_H_ */
