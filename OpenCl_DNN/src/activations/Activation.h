/*
 * Activation.h
 *
 *  Created on: Apr 30, 2015
 *      Author: richman
 */

#ifndef ACTIVATIONS_ACTIVATION_H_
#define ACTIVATIONS_ACTIVATION_H_

#include <vector>
#include "ClInterface.hpp"

class Activation {
public:
	Activation();
	virtual ~Activation();
//	Activates the function and return
	virtual std::vector<float> activate(std::vector<float>& f) = 0;
//	virtual cl::Buffer activate(cl::Buffer& buf) = 0;
};

#endif /* ACTIVATIONS_ACTIVATION_H_ */
