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
    Matrix cost(const Matrix& target,const Matrix &output) const;
};

#endif /* COST_MSE_H_ */
