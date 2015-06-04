/*
 * OpenCL.h
 *
 *  Created on: Jun 4, 2015
 *      Author: hedi7
 */

#ifndef SRC_CL_INTF_OPENCL_HPP_
#define SRC_CL_INTF_OPENCL_HPP_

#include <iostream>
#include <vector>
#include "util.hpp"
#include <tuple>
#include <CL/cl.hpp>
#include <type_traits>


class OpenCL {
public:
public:
	OpenCL(const char *p);

	OpenCL(OpenCL &other);
	virtual ~OpenCL();

	template<typename... Tp>
	void runKernel(const char* kernelname,const int blocksize,Tp && ...args);

	void loadProgram(const char *path);
private:
	//handles for creating an opencl context
	cl::Platform platform;

	//buildExecutable is called by loadProgram
	//build runtime executable from a program
	cl::Device device;
	cl::Context context;

//	Hook for the iteration
	template<std::size_t P=0,typename... Tp>
	typename std::enable_if<P == sizeof...(Tp), void>::type addkernelargs(std::tuple<Tp ...>&& t,cl::Kernel &k);

//  Start of the iteration
	template<std::size_t P = 0, typename... Tp>
	typename std::enable_if< P < sizeof...(Tp), void>::type addkernelargs(std::tuple<Tp...> && t,cl::Kernel &kernel);

//	Adding Std::vector as type to the kernel args list
	template<typename T>
	void addkernelarg(std::size_t i, std::vector<T> const & arg, cl::Kernel & kernel);

//	Adding any array into the kernel args
	template<typename T,std::size_t N>
	void addkernelarg(std::size_t i, T const (& arg)[N], cl::Kernel & kernel);

// Adding any constant to the kernel
	template<typename T>
	void addkernelarg(std::size_t i, T const & arg, cl::Kernel & kernel);


	char *contents;
};

#include "OpenCL.cpp"

#endif /* SRC_CL_INTF_OPENCL_HPP_ */
