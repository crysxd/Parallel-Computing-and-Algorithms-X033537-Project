/*
 * TanH.cpp
 *
 *  Created on: Apr 30, 2015
 *      Author: richman
 */

#include "TanH.h"

TanH::TanH() {
    // TODO Auto-generated constructor stub

}

TanH::~TanH() {
    // TODO Auto-generated destructor stub
}

CL_Matrix<float> TanH::propagate(const CL_Matrix<float>& src) {
    return src.tanh();
}

CL_Matrix<float> TanH::grad(const CL_Matrix<float>& src) {
//  return src.tanhgrad();
}
