/*
 * TanH.h
 *
 *  Created on: Apr 30, 2015
 *      Author: richman
 */

#ifndef ACTIVATIONS_TANH_H_
#define ACTIVATIONS_TANH_H_

#include "Activation.h"

class TanH: public Activation {
public:
	TanH();
	virtual ~TanH();
	virtual std::vector<float> activate(std::vector<float>& f);
	virtual cl::Buffer activate(cl::Buffer& buf);
};

#endif /* ACTIVATIONS_TANH_H_ */
