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

    OpenCL(const OpenCL &other);

    OpenCL(OpenCL &&other);

    OpenCL& operator=(OpenCL other);

    virtual ~OpenCL();

    template<typename T>
    cl::Buffer putDataOnGPU(std::vector<T> const & arg);

//  Put an array on the GPU
    template<typename T,std::size_t N>
    cl::Buffer putDataOnGPU(T const (& arg)[N]);

    template<typename T>
    void readBuffer(cl::Buffer& ,T* data);

    template<typename T>
    void readBuffer(cl::Buffer& ,std::vector<T>& data);

//  Runs a given kernel. Not that u need to allocate at least the space required for any array
//  That might be filled with data, we only return and read out the last arg
    template<typename... Tp>
    void runKernel(const char* kernelname,std::vector<std::size_t> const & outputargs,std::vector<size_t> &globalsize,std::vector<size_t> &blocksize,Tp && ...args) const;
//Runs kernel, but does not read any output arguments
    template<typename... Tp>
    void runKernelnoOut(const char* kernelname,std::vector<std::size_t> &globalsize,std::vector<size_t> &blocksize,Tp && ...args) const;//  template<typename... Tp>
//  void runKernel(const char* kernelname,std::vector<std::size_t> const & outputargs,std::vector<size_t> &globalsize,std::vector<size_t> &blocksize,Tp && ...args) const;

    void loadProgram(const char *path);

    std::vector<std::size_t> getMaxWorkItemSize() const;

    friend void swap(OpenCL &,OpenCL &);
private:
//  //handles for creating an opencl context
//  cl::Platform platform;

    //buildExecutable is called by loadProgram
    //build runtime executable from a program
    cl::Context context;
    cl::Device device;
    cl::Program program;

    void initProgramQuene(cl::Program*,cl::CommandQueue*);


    /////////////////////////////////////////////////////////////////////
    // Reading in the variables from the device at a given constant E //
    /////////////////////////////////////////////////////////////////////

//  Hook for the iteration
    template<std::size_t P,typename ...Tp>
    typename std::enable_if<P == sizeof ...(Tp), void>::type readargs(
            std::tuple<Tp ...>&& t,std::size_t outputarg,cl::Buffer &outbuf,cl::CommandQueue &quene) const;

//  Start of the iteration
    template<std::size_t P,typename ...Tp>
    typename std::enable_if< P < sizeof...(Tp), void>::type readargs(
            std::tuple<Tp...> && t,std::size_t outputarg,cl::Buffer &outbuf,cl::CommandQueue &quene) const;



//  Read in the variables
    template<typename T>
    void readarg(std::vector<T> & arg, cl::Buffer &buf,cl::CommandQueue &quene) const;

    template <typename T, std::size_t N>
    void readarg(T (&arg)[N], cl::Buffer &buf,cl::CommandQueue &quene) const;

    template <typename T>
    void readarg( T &arg, cl::Buffer &buf,cl::CommandQueue &quene) const;



    /////////////////////////////////////////////
    // Adding the input parameters as buffers //
    /////////////////////////////////////////////

//  Hook for the iteration
    template<std::size_t P=0,typename... Tp>
    typename std::enable_if<P == sizeof...(Tp), void>::type addkernelargs(std::tuple<Tp ...>&& t,cl::Kernel &k,std::vector<cl::Buffer> &outputbuffers,cl::CommandQueue &quene) const;

//  Start of the iteration
    template<std::size_t P = 0, typename... Tp>
    typename std::enable_if< P < sizeof...(Tp), void>::type addkernelargs(std::tuple<Tp...> && t,cl::Kernel &kernel,std::vector<cl::Buffer> &outputbuffers,cl::CommandQueue &quene) const;

//  Adding Std::vector as type to the kernel args list
    template<typename T>
    void addkernelarg(std::size_t i, std::vector<T> const & arg, cl::Kernel & kernel,cl::CommandQueue &quene) const;

    //In case of an outputargument, this overload is called and the buffer appended to the list
    template<typename T>
    void addkernelarg(std::size_t i, std::vector<T> const & arg, cl::Kernel & kernel,std::vector<cl::Buffer> &,cl::CommandQueue &quene) const;

//  Adding any array into the kernel args
    template<typename T,std::size_t N>
    void addkernelarg(std::size_t i, T const (& arg)[N], cl::Kernel & kernel,cl::CommandQueue &quene) const;
    //In case of an outputargument, this overload is called and the buffer appended to the list
    template<typename T,std::size_t N>
    void addkernelarg(std::size_t i, T const (& arg)[N], cl::Kernel & kernel,std::vector<cl::Buffer> &,cl::CommandQueue &quene) const;

// Adding any constant to the kernel
    template<typename T>
    void addkernelarg(std::size_t i, T const & arg, cl::Kernel & kernel,cl::CommandQueue &quene) const;
//In case of an outputargument, this overload is called and the buffer appended to the list
    template<typename T>
    void addkernelarg(std::size_t i, T const & arg, cl::Kernel & kernel,std::vector<cl::Buffer> &,cl::CommandQueue &quene) const;
//Specialization in case of a single buffer
    void addkernelarg(std::size_t i, cl::Buffer const & arg, cl::Kernel & kernel,std::vector<cl::Buffer> &,cl::CommandQueue &quene) const;

//  template<typename T>
//  typename std::enable_if<std::is_same<T,int>::value,void>::type addkernelarg(std::size_t i, T const & arg, cl::Kernel & kernel,cl::CommandQueue &) const;
    char *contents;
};

static const char*
geterrorstring(
    cl_int error)
{
    switch(error)
    {
    case(CL_SUCCESS):                           return "Success";
    case(CL_DEVICE_NOT_FOUND):                  return "Device not found!";
    case(CL_DEVICE_NOT_AVAILABLE):              return "Device not available!";
    case(CL_MEM_OBJECT_ALLOCATION_FAILURE):     return "Memory object allocation failure!";
    case(CL_OUT_OF_RESOURCES):                  return "Out of resources!";
    case(CL_OUT_OF_HOST_MEMORY):                return "Out of host memory!";
    case(CL_PROFILING_INFO_NOT_AVAILABLE):      return "Profiling information not available!";
    case(CL_MEM_COPY_OVERLAP):                  return "Overlap detected in memory copy operation!";
    case(CL_IMAGE_FORMAT_MISMATCH):             return "Image format mismatch detected!";
    case(CL_IMAGE_FORMAT_NOT_SUPPORTED):        return "Image format not supported!";
    case(CL_INVALID_VALUE):                     return "Invalid value!";
    case(CL_INVALID_DEVICE_TYPE):               return "Invalid device type!";
    case(CL_INVALID_DEVICE):                    return "Invalid device!";
    case(CL_INVALID_CONTEXT):                   return "Invalid context!";
    case(CL_INVALID_QUEUE_PROPERTIES):          return "Invalid queue properties!";
    case(CL_INVALID_COMMAND_QUEUE):             return "Invalid command queue!";
    case(CL_INVALID_HOST_PTR):                  return "Invalid host pointer address!";
    case(CL_INVALID_MEM_OBJECT):                return "Invalid memory object!";
    case(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR):   return "Invalid image format descriptor!";
    case(CL_INVALID_IMAGE_SIZE):                return "Invalid image size!";
    case(CL_INVALID_SAMPLER):                   return "Invalid sampler!";
    case(CL_INVALID_BINARY):                    return "Invalid binary!";
    case(CL_INVALID_BUILD_OPTIONS):             return "Invalid build options!";
    case(CL_INVALID_PROGRAM):                   return "Invalid program object!";
    case(CL_INVALID_PROGRAM_EXECUTABLE):        return "Invalid program executable!";
    case(CL_INVALID_KERNEL_NAME):               return "Invalid kernel name!";
    case(CL_INVALID_KERNEL):                    return "Invalid kernel object!";
    case(CL_INVALID_ARG_INDEX):                 return "Invalid index for kernel argument!";
    case(CL_INVALID_ARG_VALUE):                 return "Invalid value for kernel argument!";
    case(CL_INVALID_ARG_SIZE):                  return "Invalid size for kernel argument!";
    case(CL_INVALID_KERNEL_ARGS):               return "Invalid kernel arguments!";
    case(CL_INVALID_WORK_DIMENSION):            return "Invalid work dimension!";
    case(CL_INVALID_WORK_GROUP_SIZE):           return "Invalid work group size!";
    case(CL_INVALID_GLOBAL_OFFSET):             return "Invalid global offset!";
    case(CL_INVALID_EVENT_WAIT_LIST):           return "Invalid event wait list!";
    case(CL_INVALID_EVENT):                     return "Invalid event!";
    case(CL_INVALID_OPERATION):                 return "Invalid operation!";
    case(CL_INVALID_GL_OBJECT):                 return "Invalid OpenGL object!";
    case(CL_INVALID_BUFFER_SIZE):               return "Invalid buffer size!";
    default:                                    return "Unknown error!";
    };

    return "Unknown error";
}



#include "OpenCL.cpp"

#endif /* SRC_CL_INTF_OPENCL_HPP_ */
