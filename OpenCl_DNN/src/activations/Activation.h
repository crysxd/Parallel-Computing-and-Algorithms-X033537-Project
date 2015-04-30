/*
 * Activation.h
 *
 *  Created on: Apr 30, 2015
 *      Author: richman
 */

#ifndef ACTIVATIONS_ACTIVATION_H_
#define ACTIVATIONS_ACTIVATION_H_

class Activation {
public:
	Activation();
	virtual ~Activation();
//	Activates the function and return
	virtual double activate() = 0;
};

#endif /* ACTIVATIONS_ACTIVATION_H_ */
