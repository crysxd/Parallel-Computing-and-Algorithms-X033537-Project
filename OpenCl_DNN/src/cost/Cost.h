/*
 * Cost.h
 *
 *  Created on: Apr 30, 2015
 *      Author: richman
 */

#ifndef COST_COST_H_
#define COST_COST_H_

class Cost {
public:
	Cost();
	virtual ~Cost();
	virtual float cost(float f) = 0;

	virtual float grad(float f) = 0;
};

#endif /* COST_COST_H_ */
