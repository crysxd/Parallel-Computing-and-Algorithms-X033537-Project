//============================================================================
// Name        : OpenCl_DNN.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
using namespace std;

#include "cl_intf/ClInterface.hpp"

int main() {
	cout << "!!!Hello World!!!" << endl; // prints !!!Hello World!!!
	Cl_Interface<int,int> clinterface("cl_prog/vectoradd");
	return 0;
}

