/*
 * Activation.h
 *
 *  Created on: Apr 30, 2015
 *      Author: richman
 */

#ifndef ACTIVATIONS_ACTIVATION_H_
#define ACTIVATIONS_ACTIVATION_H_

#include <vector>
#include "CLMatrix.hpp"

class Activation {
public:
	Activation();
	virtual ~Activation();
// Uses activation function of the input arguments src and returns the result
	virtual CL_Matrix<float> propagate(const CL_Matrix<float>& src) = 0;

	virtual CL_Matrix<float> grad(const CL_Matrix<float>& src) = 0;
//	virtual cl::Buffer activate(cl::Buffer& buf) = 0;
};

#endif /* ACTIVATIONS_ACTIVATION_H_ */
