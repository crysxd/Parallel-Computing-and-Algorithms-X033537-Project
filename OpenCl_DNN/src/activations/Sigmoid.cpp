/*
 * Sigmoid.cpp
 *
 *  Created on: Apr 30, 2015
 *      Author: richman
 */

#include "Sigmoid.h"

Sigmoid::Sigmoid():_cl_intf("sigmoid.cl") {
	// TODO Auto-generated constructor stub
	std::vector<std::vector<double>> a;
	std::vector<double> b;
	b.push_back(1);
	b.push_back(20);
	b.push_back(30);
	a.push_back(b);
	std::vector<double> bb;
	bb.push_back(20);
	bb.push_back(10);
	bb.push_back(4);
	a.push_back(bb);
	std::vector<std::vector<double>> c;
	std::vector<double> d(3);
	c.push_back(d);
	this->_cl_intf.runKernel("sigmoid",1,a,&c);
}

Sigmoid::~Sigmoid() {
	// TODO Auto-generated destructor stub
}

double Sigmoid::activate() {
	return 0;
}

