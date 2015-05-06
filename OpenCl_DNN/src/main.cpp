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
    std::vector<std::vector<int>> a;
    std::vector<int> b;
    b.push_back(1);
    a.push_back(b);
    std::vector<int> bb;
    bb.push_back(20);
    a.push_back(bb);
	Cl_Interface<int,int> clinterface("vectoradd.cl");
    std::vector<std::vector<int>> c;
    std::vector<int> d;
    d.push_back(0);
    c.push_back(d);
    clinterface.runKernel("vector_add_gpu",a,&c);
	return 0;
}

