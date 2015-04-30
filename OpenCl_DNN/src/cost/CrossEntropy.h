/*
 * CrossEntropy.h
 *
 *  Created on: Apr 30, 2015
 *      Author: richman
 */

#ifndef COST_CROSSENTROPY_H_
#define COST_CROSSENTROPY_H_

#include "Cost.h"

class CrossEntropy: public Cost {
public:
	CrossEntropy();
	virtual ~CrossEntropy();
	float cost(float f);
	float grad(float f);
};

#endif /* COST_CROSSENTROPY_H_ */
