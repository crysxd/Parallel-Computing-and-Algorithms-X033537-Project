/*
 * Cost.h
 *
 *  Created on: Apr 30, 2015
 *      Author: richman
 */

#ifndef COST_COST_H_
#define COST_COST_H_

#include "CLMatrix.hpp"

typedef CL_Matrix<float> Matrix;

class Cost {
public:
	Cost();
	virtual ~Cost();
	virtual Matrix cost(Matrix& f) = 0;
};

#endif /* COST_COST_H_ */
