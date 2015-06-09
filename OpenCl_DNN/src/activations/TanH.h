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
	virtual CL_Matrix<float> activate(CL_Matrix<float>& f);
};

#endif /* ACTIVATIONS_TANH_H_ */
