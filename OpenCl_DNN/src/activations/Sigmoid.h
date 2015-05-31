/*
 * Sigmoid.h
 *
 *  Created on: Apr 30, 2015
 *      Author: richman
 */

#ifndef ACTIVATIONS_SIGMOID_H_
#define ACTIVATIONS_SIGMOID_H_

#include "Activation.h"

class Sigmoid: public Activation {
public:
	Sigmoid();
	virtual ~Sigmoid();

	double activate();
};

#endif /* ACTIVATIONS_SIGMOID_H_ */
