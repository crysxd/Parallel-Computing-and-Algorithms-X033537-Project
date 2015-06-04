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
#include <cassert>
#include <CL/cl.hpp>
#include <type_traits>


class OpenCL {
public:
	OpenCL(const char *);

	OpenCL(OpenCL &other);
	virtual ~OpenCL();

//	Runs a given kernel. Not that u need to allocate at least the space required for any array
//	That might be filled with data, we only return and read out the last arg
	template<typename... Tp>
	void runKernel(const char* kernelname,std::vector<std::size_t> const & outputargs,const u_int32_t globalsize,const u_int32_t blocksize,Tp && ...args);

	void loadProgram(const char *path);
private:
	//handles for creating an opencl context
	cl::Platform platform;

	//buildExecutable is called by loadProgram
	//build runtime executable from a program
	cl::Device device;
	cl::Context context;


	/////////////////////////////////////////////////////////////////////
	// Reading in the variables from the device at a given constant E //
	/////////////////////////////////////////////////////////////////////

//	Hook for the iteration
	template<std::size_t P,typename ...Tp>
	typename std::enable_if<P == sizeof ...(Tp), void>::type readargs(std::tuple<Tp ...>&& t,std::size_t outputarg,cl::CommandQueue &,cl::Buffer &outbuf);

//  Start of the iteration
	template<std::size_t P,typename ...Tp>
	typename std::enable_if< P < sizeof...(Tp), void>::type readargs(std::tuple<Tp...> && t,std::size_t outputarg,cl::CommandQueue &,cl::Buffer &outbuf);



//	Read in the varaibles
	template<typename T>
	void readarg(std::vector<T> & arg, cl::CommandQueue &,cl::Buffer &buf);

	template <typename T, std::size_t N>
	void readarg(T (&arg)[N], cl::CommandQueue &,cl::Buffer &buf);

	template <typename T>
	void readarg(T &arg, cl::CommandQueue &,cl::Buffer &buf);



	/////////////////////////////////////////////
	// Adding the input parameters as buffers //
	/////////////////////////////////////////////

//	Hook for the iteration
	template<std::size_t P=0,typename... Tp>
	typename std::enable_if<P == sizeof...(Tp), void>::type addkernelargs(std::tuple<Tp ...>&& t,cl::Kernel &k,cl::CommandQueue &,std::vector<cl::Buffer> &outputbuffers);

//  Start of the iteration
	template<std::size_t P = 0, typename... Tp>
	typename std::enable_if< P < sizeof...(Tp), void>::type addkernelargs(std::tuple<Tp...> && t,cl::Kernel &kernel,cl::CommandQueue &,std::vector<cl::Buffer> &outputbuffers);

//	Adding Std::vector as type to the kernel args list
	template<typename T>
	void addkernelarg(std::size_t i, std::vector<T> const & arg, cl::Kernel & kernel,cl::CommandQueue &);
	//In case of an outputargument, this overload is called and the buffer appended to the list
	template<typename T>
	void addkernelarg(std::size_t i, std::vector<T> const & arg, cl::Kernel & kernel,cl::CommandQueue &,std::vector<cl::Buffer> &);

//	Adding any array into the kernel args
	template<typename T,std::size_t N>
	void addkernelarg(std::size_t i, T const (& arg)[N], cl::Kernel & kernel,cl::CommandQueue &);
	//In case of an outputargument, this overload is called and the buffer appended to the list
	template<typename T,std::size_t N>
	void addkernelarg(std::size_t i, T const (& arg)[N], cl::Kernel & kernel,cl::CommandQueue &,std::vector<cl::Buffer> &);

// Adding any constant to the kernel
	template<typename T>
	void addkernelarg(std::size_t i, T const & arg, cl::Kernel & kernel,cl::CommandQueue &);
//In case of an outputargument, this overload is called and the buffer appended to the list
	template<typename T>
	void addkernelarg(std::size_t i, T const & arg, cl::Kernel & kernel,cl::CommandQueue &,std::vector<cl::Buffer> &);
//

	char *contents;
};

#include "OpenCL.cpp"

#endif /* SRC_CL_INTF_OPENCL_HPP_ */
