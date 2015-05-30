/*
 * ClInterface.hpp
 *
 *  Created on: May 3, 2015
 *      Author: hedi7
 */

#ifndef SRC_CL_INTF_CLINTERFACE_HPP_
#define SRC_CL_INTF_CLINTERFACE_HPP_

#include <iostream>
#include <vector>
#include "util.hpp"

#include <CL/cl.hpp>

template<typename I, typename O> class Cl_Interface {
public:
	Cl_Interface(const char* path);
	virtual ~Cl_Interface();
	void loadProgram(const char* path);

	void runKernel(const char* kernelname,const int blocksize,const std::vector<std::vector<I>> &input,std::vector<std::vector<O>> *output);


private:
	//handles for creating an opencl context
	cl::Platform platform;

	//buildExecutable is called by loadProgram
	//build runtime executable from a program
	void buildExecutable();
	cl::Device device;
	cl::Context *context;

	char *contents;
};
// Template inclusion needs to be done after the header
// Otherwise it will result in an error.
#include "ClInterface.inl"

#endif /* SRC_CL_INTF_CLINTERFACE_HPP_ */
