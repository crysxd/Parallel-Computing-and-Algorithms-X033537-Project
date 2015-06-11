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
	virtual CL_Matrix<float> propagate(const CL_Matrix<float>& f);
	virtual CL_Matrix<float> grad(const CL_Matrix<float>& src);

};

#endif /* ACTIVATIONS_TANH_H_ */
