//============================================================================
// Name        : OpenCl_DNN.cpp
// Author      :
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
using namespace std;

#include <vector>
#include "cl_intf/ClInterface.hpp"

int main() {
    std::vector<std::vector<float>> a;
    std::vector<float> b;
    b.push_back(1);
    b.push_back(20);
    b.push_back(30);
    a.push_back(b);
    std::vector<float> bb;
    bb.push_back(20);
    bb.push_back(10);
    bb.push_back(4);
    a.push_back(bb);
	Cl_Interface<float,float> clinterface("vectoradd.cl");
    std::vector<std::vector<float>> c;
    std::vector<float> d(3);
    c.push_back(d);
    clinterface.runKernel("vector_add_gpu",a,&c);
	return 0;
}

