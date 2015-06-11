/*
 * HiddenLayer.h
 *
 *  Created on: Apr 30, 2015
 *      Author: richman
 */

#ifndef HIDDENLAYER_HIDDENLAYER_H_
#define HIDDENLAYER_HIDDENLAYER_H_

#include "Activation.h"
#include "Cost.h"
#include <vector>
#include "CLMatrix.hpp"

typedef CL_Matrix<float> Matrix;

class HiddenLayer {
public:
	HiddenLayer(Activation &activation,u_int32_t dim);
	virtual ~HiddenLayer();

	Matrix propagate(const Matrix &input,const Matrix &weight,const Matrix &bias);

	Matrix grad(const Matrix &layer);

//	Getters and setters
	u_int32_t getDim() const {
		return dim;
	}


private:
	Activation *activation;
	u_int32_t dim;

};

#endif /* HIDDENLAYER_HIDDENLAYER_H_ */
