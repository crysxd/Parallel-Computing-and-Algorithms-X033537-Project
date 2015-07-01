/*
 * MSE.cpp
 *
 *  Created on: Apr 30, 2015
 *      Author: richman
 */

#include "MSE.h"

MSE::MSE() {
    // TODO Auto-generated constructor stub

}

MSE::~MSE() {
    // TODO Auto-generated destructor stub
}

Matrix MSE::cost(const Matrix& target,const Matrix& output)const {
    return 0.5f*((target - output)*(target-output));
}
