/*
 * Sigmoid.cpp
 *
 *  Created on: Apr 30, 2015
 *      Author: richman
 */

#include "Sigmoid.h"

Sigmoid::Sigmoid(){


}

Sigmoid::~Sigmoid() {
    // TODO Auto-generated destructor stub
}




CL_Matrix<float> Sigmoid::propagate(const CL_Matrix<float>& src) {
    return src.sigmoid();
}

CL_Matrix<float> Sigmoid::grad(const CL_Matrix<float>& src) {
    return src.sigmoidgrad();
}

//std::vector<util::GPU_Buffer> Sigmoid::activateKeep(std::vector<float>& f) {
//  std::vector<std::vector<float>> input;
//  input.push_back(f);
//  std::vector<std::vector<float>> output;
//  std::vector<float> a(f.size());
//  output.push_back(a);
//  return this->_cl_intf.runKernelBuffer(this->_kernelname,1,input,&output);
//}
//
//void Sigmoid::chainActivate(std::vector<util::GPU_Buffer>* in) {
//  this->_cl_intf.chainKernel(this->_kernelname,1,*in,*in);
//}
//
//std::vector<float> Sigmoid::activate(std::vector<util::GPU_Buffer>& f) {
//  std::vector<std::vector<float>> output;
//  std::vector<float> ret(f[0].datalength);
//  output.push_back(ret);
//  this->_cl_intf.chainKernel(this->_kernelname,1,f,f);
//  this->_cl_intf.readResult(f,&output);
//  return output[0];
//
//
//}

Sigmoid& Sigmoid::operator =(const Sigmoid &other) {
    return *this;
}
