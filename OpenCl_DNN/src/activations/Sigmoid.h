/*
 * Sigmoid.h
 *
 *  Created on: Apr 30, 2015
 *      Author: richman
 */

#ifndef ACTIVATIONS_SIGMOID_H_
#define ACTIVATIONS_SIGMOID_H_

#include "Activation.h"
#include "../cl_intf/ClInterface.hpp"

class Sigmoid: public Activation {
public:
	Sigmoid();
	virtual ~Sigmoid();
	virtual double activate();

private:
	Cl_Interface<double,double> _cl_intf;
};

#endif /* ACTIVATIONS_SIGMOID_H_ */
