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
#include <tuple>
#include <CL/cl.hpp>
#include <type_traits>


template<typename I, typename O> class Cl_Interface {
public:
	Cl_Interface(const char* path);
	Cl_Interface(const Cl_Interface&);

	virtual ~Cl_Interface();
	void loadProgram(const char* path);


	void runKernel(const char* kernelname,const int blocksize,const std::vector<std::vector<I>> &input,std::vector<std::vector<O>> *output);

//	Does the same as runKernel, except that it doesnt read out the result and keeps it on the GPU
	std::vector<util::GPU_Buffer> runKernelBuffer(const char* kernelname,const int blocksize,const std::vector<std::vector<I>> &input,std::vector<std::vector<O>> *output);


	void chainKernel(const char* kernelname,const int blocksize,const std::vector<util::GPU_Buffer> &input,const std::vector<util::GPU_Buffer> &output);
//	Reads buffer outputbuffer into output output
	void readResult(std::vector<util::GPU_Buffer> outputbuffer,std::vector<std::vector<O>> *output);


private:
	//handles for creating an opencl context
	cl::Platform platform;

	//buildExecutable is called by loadProgram
	//build runtime executable from a program
	void buildExecutable();
	cl::Device device;
	cl::Context context;

	char *contents;
};
// Template inclusion needs to be done after the header
// Otherwise it will result in an error.
#include "ClInterface.inl"

#endif /* SRC_CL_INTF_CLINTERFACE_HPP_ */
