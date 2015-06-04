/*
 * OpenCL.cpp
 *
 *  Created on: Jun 4, 2015
 *      Author: hedi7
 */

#include "OpenCL.hpp"


// CL_SOURCE_DIR is defined in the cmake file
#ifndef CL_SOURCE_DIR
    #define CL_SOURCE_DIR "../src"
#endif

#ifndef DEBUG
    #define DEBUG 1
#endif

OpenCL::OpenCL(const char * programpath){
        // Get all platforms from the API
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    if (all_platforms.size() == 0) {
        std::cerr << " No platforms found. Check OpenCL installation!\n"
                << std::endl;
        exit(1);
    }
    //We assume having only one platform and choose the first one
    cl::Platform defaultplatform = all_platforms[0];
    #if !DEBUG
    std::cout << "Using Platform : "
            << defaultplatform.getInfo<CL_PLATFORM_NAME>() << std::endl;
    #endif
    std::vector<cl::Device> all_devices;
    defaultplatform.getDevices(CL_DEVICE_TYPE_GPU, &all_devices);
    if (all_devices.size() == 0) {
        std::cout << " No devices found. Check OpenCL installation!\n";
        exit(1);
    }
    this->device = all_devices.front();
    // Problem is since the class is a template class, the call to Cl_intf<I,O>::device
    // is ambigious. It could mean syntatically that I want to do:
    // (this->device < CL_DEVICE_NAME) > ()
    // To resolve that we explicitally tell the compilter that it's a tempalte
    std::string devicename = this->device.template getInfo<CL_DEVICE_NAME>();
    #if !DEBUG
    std::cout << "Using device: " << devicename << std::endl;
    #endif

    std::string newpath(CL_SOURCE_DIR);
    newpath += "/cl_prog";
    newpath += "/" + std::string(programpath);
#if !DEBUG
    std::cout << "Reading Kernel from " <<newpath <<std::endl;
#endif
    this->context = cl::Context(this->device);
    this->loadProgram(newpath.c_str());

}

OpenCL::OpenCL(OpenCL &other){
    this->platform = other.platform;
    this->device = other.device;
    this-> context = other.context;
    this->contents = new char[strlen(other.contents)];
    strcpy(this->contents,other.contents);
}

OpenCL::~OpenCL() {
    delete [] contents;
}



template <typename T>
void OpenCL::addkernelarg(std::size_t i, T const & arg, cl::Kernel & kernel)
{
    std::cout << "Arg n°" << i << ": " << arg << std::endl;

    // Your code to add your arg to the kernel
}

template <typename T, std::size_t N>
void OpenCL::addkernelarg(std::size_t i, T const (& arg)[N], cl::Kernel & kernel)
{
    std::cout << "Arg n°" << i << ": raw array of size " << N << std::endl;

    // Your code to add your arg to the kernel
}

template <typename T>
void OpenCL::addkernelarg(std::size_t i, std::vector<T> const & arg, cl::Kernel & kernel)
{

    std::cout << "Arg n°" << i << ": std::vector of size: " << arg.size() << std::endl;

    // Your code to add your arg to the kernel
}

//Hook Class, dont do anything here, since it is called at the top P==size(Tp) arg
template<std::size_t P , typename... Tp>
inline typename std::enable_if<P == sizeof...(Tp), void>::type
OpenCL::addkernelargs(std::tuple<Tp...> && t,cl::Kernel &kernel)
{
}

    //Iteration for Class
template<std::size_t P, typename... Tp>
inline typename std::enable_if<P < sizeof...(Tp), void>::type
OpenCL::addkernelargs(std::tuple<Tp...> && t,cl::Kernel &kernel)
    {
        // Type
        typedef typename std::tuple_element<P, std::tuple<Tp...>>::type type;

        // Add the value of the current item from std::get<P> to the args in kernel
        // This function decides which type the kernel arg is
        addkernelarg(P, std::get<P>(t), kernel);

        // Recurse to get the remaining args
        addkernelargs<P + 1, Tp...>(std::forward<std::tuple<Tp...>>(t), kernel);

    }


template<typename ... Tp>
void OpenCL::runKernel(const char* kernelname,const int blocksize,Tp && ... args){

    cl::Program::Sources sources;
    //Include the read out contents from the vector file into the sources to parse
    sources.push_back(std::make_pair(this->contents,strlen(this->contents)+1));
    //Initiate a program from the sources
    cl::Program program(this->context,sources);
    if(program.build({this->device})!=CL_SUCCESS){
        std::cout<<" Error building: "<<program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(this->device)<<"\n";
        exit(1);
    }

    ////////////////////////////////////////////////////
    // Initalize the transfer and executeable objects //
    ////////////////////////////////////////////////////
    // Queue is reponsible for transferring data and the kernel_operator executes the code
    // The quene pushes and returns the buffer objects between host and device
    cl::CommandQueue queue(this->context,this->device);
    cl::Kernel kernel_operator(program,kernelname);


    addkernelargs<0>(std::forward_as_tuple(args...),kernel_operator);


}


void OpenCL::loadProgram(const char *path){
    this->contents = util::file_contents(path);
}



