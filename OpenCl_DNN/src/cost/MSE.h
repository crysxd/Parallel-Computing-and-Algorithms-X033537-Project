/*
 * MSE.h
 *
 *  Created on: Apr 30, 2015
 *      Author: richman
 */

#ifndef COST_MSE_H_
#define COST_MSE_H_

#include "Cost.h"

class MSE : public Cost{
public:
	MSE();
	virtual ~MSE();
	float cost(float f);
	float grad(float f);
};

#endif /* COST_MSE_H_ */
