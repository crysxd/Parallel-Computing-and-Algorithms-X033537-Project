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
    #define DEBUG 0
#endif

inline OpenCL::OpenCL(const char * programpath){
        // Get all platforms from the API
//    std::cout << "Constructor called " <<std::endl;
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    if (all_platforms.size() == 0) {
        std::cerr << " No platforms found. Check OpenCL installation!\n"
                << std::endl;
        exit(1);
    }
    //We assume having only one platform and choose the first one
    cl::Platform defaultplatform = all_platforms[0];
    #if DEBUG
    std::cout << "Using Platform : "
            << defaultplatform.getInfo<CL_PLATFORM_NAME>() << std::endl;
    #endif
    std::vector<cl::Device> all_gpu_devices;
    defaultplatform.getDevices(CL_DEVICE_TYPE_GPU, &all_gpu_devices);

    if (all_gpu_devices.size() == 0) {
        std::cout << " No devices found. Check OpenCL installation!\n";
        exit(1);
    }
    this->device = all_gpu_devices.front();
    // Problem is since the class is a template class, the call to Cl_intf<I,O>::device
    // is ambigious. It could mean syntatically that I want to do:
    // (this->device < CL_DEVICE_NAME) > ()
    // To resolve that we explicitally tell the compilter that it's a tempalte
    std::string devicename = this->device.template getInfo<CL_DEVICE_NAME>();
    #if DEBUG
    std::cout << "Using device: " << devicename << std::endl;
    #endif


    this->context = cl::Context(this->device);
    this->loadProgram(programpath);

    cl::Program::Sources sources;
    //Include the read out contents from the vector file into the sources to parse
    sources.push_back(std::make_pair(this->contents,strlen(this->contents)+1));
    //Initiate a program from the sources
    program = cl::Program(this->context,sources);
    if(program.build({this->device})!=CL_SUCCESS){
        std::cout<<" Error building: "<<program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(this->device)<<"\n";
        exit(1);
    }

}

template<typename T>
inline void OpenCL::readBuffer(cl::Buffer &buf,std::vector<T>& arg){
////    //////////////////////////////////////////////////
////     Initalize the transfer and executeable objects //
////    //////////////////////////////////////////////////
////     Queue is reponsible for transferring data and the kernel_operator executes the code
////     The quene pushes and returns the buffer objects between host and device
    cl::CommandQueue queue(this->context,this->device);

    queue.enqueueReadBuffer(buf,CL_TRUE,0,arg.size()*sizeof(T),&(arg[0]));
    queue.finish();
//  std::cout <<"SIZE " << arg.size()*sizeof(T) <<std::endl;
//  for(auto i = 0; i < arg.size();i++){
//      std::cout << arg.at(i);
//  }
}


inline OpenCL::OpenCL(const OpenCL &other):context(other.context),device(other.device),contents(new char[strlen(other.contents)+1]),program(other.program){
    std::copy(other.contents,other.contents+strlen(other.contents),contents);
    contents[strlen(other.contents)] = {'\0'};
}

inline OpenCL::OpenCL(OpenCL &&other):OpenCL(other.contents){
    swap(*this,other);
}

inline OpenCL& OpenCL::operator=(OpenCL other){
    swap(*this,other);
    return (*this);
}

inline void swap(OpenCL &lhs,OpenCL &rhs){
    using std::swap;
    swap(lhs.contents,rhs.contents);
    swap(lhs.context,rhs.context);
    swap(lhs.device,rhs.device);
    swap(lhs.program,rhs.program);
}

inline OpenCL::~OpenCL() {
    delete [] contents;
}


template<typename T>
inline cl::Buffer OpenCL::putDataOnGPU(std::vector<T> const &data) {
//  cl::Program::Sources sources;
//  //Include the read out contents from the vector file into the sources to parse
//  sources.push_back(std::make_pair(this->contents,strlen(this->contents)+1));
//  //Initiate a program from the sources
//  cl::Program program(this->context,sources);
//  if(program.build({this->device})!=CL_SUCCESS){
//      std::cout<<" Error building: "<<program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(this->device)<<"\n";
//      exit(1);
//  }

//  //////////////////////////////////////////////////
//   Initalize the transfer and executeable objects //
//  //////////////////////////////////////////////////
//   Queue is reponsible for transferring data and the kernel_operator executes the code
//   The quene pushes and returns the buffer objects between host and device
    cl::CommandQueue queue(this->context,this->device);
    cl::Buffer buf(this->context,CL_MEM_READ_WRITE,data.size()*sizeof(T));
//    std::cout << "enqeue\n";
    queue.enqueueWriteBuffer(buf,CL_TRUE,0,data.size()*sizeof(T),&(data[0]));
    queue.finish();
    return buf;
}





inline std::vector<std::size_t> OpenCL::getMaxWorkItemSize() const{
    return this->device.template getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
}

///////////////////////////////////////////////////////////////////////////////////////////
// Kernel adders for sclars, vectors and arrays. These do not add the kernel on any list //
///////////////////////////////////////////////////////////////////////////////////////////


template <typename T>
inline void OpenCL::addkernelarg(std::size_t i, T const & arg, cl::Kernel & kernel,cl::CommandQueue &quene) const
{

    cl::Buffer buffer(this->context,CL_MEM_READ_WRITE,sizeof(T));
//    std::cout << "enqeue\n";
    quene.enqueueWriteBuffer(buffer,CL_FALSE,0,sizeof(T),arg);
    kernel.setArg(i,buffer);

}

template <typename T, std::size_t N>
inline void OpenCL::addkernelarg(std::size_t i, T const (& arg)[N], cl::Kernel & kernel,cl::CommandQueue &quene) const
{
    cl::Buffer buffer(this->context,CL_MEM_READ_WRITE,N*sizeof(T));
//    std::cout << "enqeue\n";
    quene.enqueueWriteBuffer(buffer,CL_FALSE,0,sizeof(T)*N,&arg);
    kernel.setArg(i,buffer);

}

template <typename T>
inline void OpenCL::addkernelarg(std::size_t i, std::vector<T> const & arg, cl::Kernel & kernel,cl::CommandQueue &quene) const
{
    cl::Buffer buffer(this->context,CL_MEM_READ_WRITE,arg.size()*sizeof(T));
//    std::cout << "enqeue\n";
    quene.enqueueWriteBuffer(buffer,CL_FALSE,0,sizeof(T)*arg.size(),&(arg[0]));
    kernel.setArg(i,buffer);

}

//////////////////////////////////////////////////////////////////////
// Pushes data on device and adds kernel into the outputbuffer list //
//////////////////////////////////////////////////////////////////////

template <typename T>
inline void OpenCL::addkernelarg(std::size_t i, T const & arg, cl::Kernel & kernel,std::vector<cl::Buffer> &outputbuffer,cl::CommandQueue &quene) const
{

    kernel.setArg(i,arg);
//  Push back a dummy since we actually dont need to allocate anything for a scalar
    outputbuffer.push_back(cl::Buffer());
}

inline void OpenCL::initProgramQuene(cl::Program* prog, cl::CommandQueue* quene) {
    //    std::cout << "Loading " << path <<std::endl;

//  cl::Program::Sources sources;
//  //Include the read out contents from the vector file into the sources to parse
//  sources.push_back(std::make_pair(this->contents,strlen(this->contents)+1));
//  //Initiate a program from the sources
//  cl::Program program(this->context,sources);
//  if(program.build({this->device})!=CL_SUCCESS){
//      std::cout<<" Error building: "<<program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(this->device)<<"\n";
//      exit(1);
//  }

//  prog = new cl::Program(program);

    ////////////////////////////////////////////////////
    // Initalize the transfer and executeable objects //
    ////////////////////////////////////////////////////
    // Queue is reponsible for transferring data and the kernel_operator executes the code
    // The quene pushes and returns the buffer objects between host and device
    quene = new cl::CommandQueue(this->context,this->device);
}

inline void OpenCL::addkernelarg(std::size_t i, cl::Buffer const & arg,
        cl::Kernel & kernel,std::vector<cl::Buffer> &,cl::CommandQueue &quene) const{
    kernel.setArg(i,arg);
}


template <typename T, std::size_t N>
inline void OpenCL::addkernelarg(std::size_t i, T const (& arg)[N], cl::Kernel & kernel,
        std::vector<cl::Buffer> &outputbuffer,cl::CommandQueue &quene) const
{
    cl::Buffer buffer(this->context,CL_MEM_READ_WRITE,N*sizeof(T));
//    std::cout << "enqeue\n";
    cl_int err = quene.enqueueWriteBuffer(buffer,CL_FALSE,0,sizeof(T)*N,&arg);
    if (err){
            std::cerr << "Error while pushing Array. Errorcode: " << err << std::endl;
    }
    kernel.setArg(i,buffer);
    outputbuffer.push_back(buffer);

}

template <typename T>
void OpenCL::addkernelarg(std::size_t i, std::vector<T> const & arg,
        cl::Kernel & kernel,std::vector<cl::Buffer> &outputbuffer,cl::CommandQueue &quene)const
{
    cl::Buffer buffer(this->context,CL_MEM_READ_WRITE,arg.size()*sizeof(T));
    outputbuffer.push_back(buffer);
//    std::cout << "enqeue\n";
    cl_int err = quene.enqueueWriteBuffer(buffer,CL_FALSE,0,sizeof(T)*arg.size(),&(arg[0]));
    if (err){
        std::cerr << "Error while pushing Vector. Errorcode: " << err << std::endl;
    }
    kernel.setArg(i,buffer);

}


/////////////////////////////////////////////////
// The iteration methods for the variaric args //
/////////////////////////////////////////////////

//Hook Class, dont do anything here, since it is called at the top P==size(Tp) arg
template<std::size_t P , typename... Tp>
inline typename std::enable_if<P == sizeof...(Tp), void>::type
OpenCL::addkernelargs(std::tuple<Tp...> && t,cl::Kernel &kernel,std::vector<cl::Buffer> &outputbuffers,cl::CommandQueue &quene) const{
}

    //Iteration for Class
template<std::size_t P, typename... Tp>
inline typename std::enable_if<P < sizeof...(Tp), void>::type
OpenCL::addkernelargs(std::tuple<Tp...> && t,cl::Kernel &kernel,std::vector<cl::Buffer> &outputbuffers,cl::CommandQueue &quene) const
    {
        // Type
        typedef typename std::tuple_element<P, std::tuple<Tp...>>::type type;

        // Add the value of the current item from std::get<P> to the args in kernel
        // This function decides which type the kernel arg is
        addkernelarg(P, std::get<P>(t), kernel,outputbuffers,quene);

        // Recurse to get the remaining args
        addkernelargs<P + 1, Tp...>(std::forward<std::tuple<Tp...>>(t), kernel,outputbuffers,quene);

    }

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Runs the kernel, given as kernelname. Outputarg is an integer indicating which one the outputargument is. //
// globalsize is usuallty the size of the array and blocksize is the local size for one work group           //
///////////////////////////////////////////////////////////////////////////////////////////////////////////////


template<typename ... Tp>
inline void OpenCL::runKernel(const char* kernelname,std::vector<std::size_t> const & outputargs,
        std::vector<size_t> &globalsize,std::vector<size_t> &blocksize,Tp && ... args) const{
//  Note we currently disabled to pass local worksize
//  Outputargs needs to be smaller than the amount of parameters we have.
    assert(outputargs.size() <= sizeof...(Tp));

//    cl::Program::Sources sources;
//    //Include the read out contents from the vector file into the sources to parse
//    sources.push_back(std::make_pair(this->contents,strlen(this->contents)+1));
//    //Initiate a program from the sources
//    cl::Program program(this->context,sources);
//    if(program.build({this->device})!=CL_SUCCESS){
//        std::cout<<" Error building: "<<program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(this->device)<<"\n";
//        exit(1);
//    }

    ////////////////////////////////////////////////////
    // Initalize the transfer and executeable objects //
    ////////////////////////////////////////////////////
    // Queue is reponsible for transferring data and the kernel_operator executes the code
    // The quene pushes and returns the buffer objects between host and device
    cl::CommandQueue quene(this->context,this->device);
    cl::Kernel kernel_operator(program,kernelname);

//    Stores the all the buffers of this execution in the vector
    std::vector<cl::Buffer> outputbuffers;
    addkernelargs(std::forward_as_tuple(args...),kernel_operator,outputbuffers,quene);

//    Wait for the transfers to finish
    quene.finish();

    #if DEBUG
    std::cout << "Running Kernel : " << kernelname << std::endl;
    #endif

    cl::NDRange *globalrange;
    switch(globalsize.size()){
    case 1:
        globalrange=new cl::NDRange(globalsize[0]);
        break;
    case 2:
        globalrange=new cl::NDRange(globalsize[0],globalsize[1]);
        break;
    case 3:
        globalrange=new cl::NDRange(globalsize[0],globalsize[1],globalsize[2]);
        break;
    default:
        globalrange=new cl::NDRange();
    }

//    cl::NDRange *localrange;
//    switch(blocksize.size()){
//    case 1:
//      localrange=new cl::NDRange(blocksize[0]);
//      break;
//    case 2:
//      localrange=new cl::NDRange(blocksize[0],blocksize[1]);
//      break;
//    case 3:
//      localrange=new cl::NDRange(blocksize[0],blocksize[1],blocksize[2]);
//      break;
//
//    }

    cl::Event event;
    ///////////////////////////////////
    // Calculation is being executed //
    ///////////////////////////////////
    //The first range is the offset for the arrays, second is the global range, third the local one
    cl_int ret = quene.enqueueNDRangeKernel(kernel_operator,cl::NullRange,*globalrange,cl::NullRange,NULL,&event);
    if (ret != 0) {
        cerr<< " Error when Executing the kernel \"" << kernelname << "\" Code: " << ret << endl;
        cerr<< " Reason : " << geterrorstring(ret) << endl;
    }
    event.wait();
//    Reading the results
    for(auto outputarg : outputargs)
    {
        readargs<0>(std::forward_as_tuple(args...), outputarg,outputbuffers.at(outputarg),quene);
    }
//    Finish
    quene.finish();


}

//#include <chrono>

template<typename ... Tp>
inline void OpenCL::runKernelnoOut(const char* kernelname, std::vector<size_t> &globalsize,std::vector<size_t> &blocksize,Tp && ... args) const{
//    auto start = std::chrono::system_clock::now();

    ////////////////////////////////////////////////////
    // Initalize the transfer and executeable objects //
    ////////////////////////////////////////////////////
    // Queue is reponsible for transferring data and the kernel_operator executes the code
    // The quene pushes and returns the buffer objects between host and device
    cl::CommandQueue queue(this->context,this->device);
    cl::Kernel kernel_operator(program,kernelname);

//    Stores the all the buffers of this execution in the vector
    std::vector<cl::Buffer> outputbuffers;
    addkernelargs(std::forward_as_tuple(args...),kernel_operator,outputbuffers,queue);

//    Wait for the transfers to finish
    queue.finish();

//    std::chrono::duration<double> elapsed_seconds1 = std::chrono::system_clock::now() - start;
    #if DEBUG
    std::cout << "Running Kernel : " << kernelname << std::endl;
    #endif

    cl::NDRange *globalrange;
    switch(globalsize.size()){
    case 1:
        globalrange=new cl::NDRange(globalsize[0]);
        break;
    case 2:
        globalrange=new cl::NDRange(globalsize[0],globalsize[1]);
        break;
    case 3:
        globalrange=new cl::NDRange(globalsize[0],globalsize[1],globalsize[2]);
        break;
    default:
        globalrange=new cl::NDRange();
    }

    cl::Event event;
    ///////////////////////////////////
    // Calculation is being executed //
    ///////////////////////////////////
    //The first range is the offset for the arrays, second is the global range, third the local one
    cl_int ret = queue.enqueueNDRangeKernel(kernel_operator,cl::NullRange,*globalrange,cl::NullRange,NULL,&event);
    if (ret != 0) {
        cerr<< " Error when Executing the kernel \"" << kernelname << "\" Code: " << ret << endl;
        cerr<< " Reason : " << geterrorstring(ret) << endl;
    }
    event.wait();
//    std::chrono::duration<double> elapsed_seconds2 = std::chrono::system_clock::now() - start;
//    std::cout << "kernel times " << elapsed_seconds1.count() << ", "  << elapsed_seconds2.count() << "\n";
//    Finished
}
//
//template<typename Tp>
//inline void OpenCL::runKernel(const char* kernelname,
//      std::vector<std::size_t>& globalsize, std::vector<size_t>& blocksize,
//      Tp&& args) const {
//}

//Finished the iteration
template<std::size_t P,typename... Tp>
typename std::enable_if< P == sizeof...(Tp), void>::type OpenCL::readargs(
        std::tuple<Tp ...>&& t,std::size_t outputarg,cl::Buffer &outbuf,cl::CommandQueue &quene) const{
//  std::cout << E << std::endl;
}

//  Start of the iteration
template<std::size_t P, typename... Tp>
typename std::enable_if< P < sizeof...(Tp), void>::type OpenCL::readargs
(std::tuple<Tp...> && t,std::size_t outputarg,cl::Buffer &outbuf,cl::CommandQueue &quene) const{
    if(P == outputarg)
        {
        readarg(std::get<P>(t), outbuf,quene);
        }
    else{
        readargs<P + 1, Tp...>(std::forward<std::tuple<Tp...>>(t), outputarg,outbuf,quene);

    }
}


//std::vector is given
template<typename T>
inline void OpenCL::readarg(std::vector<T> & arg, cl::Buffer &buf,cl::CommandQueue &quene) const{
    quene.enqueueReadBuffer(buf,CL_FALSE,0,arg.size()*sizeof(T),&(arg[0]));
}

//Array is given
template <typename T, std::size_t N>
inline void OpenCL::readarg(T (&arg)[N], cl::Buffer &buf,cl::CommandQueue &quene) const{
    quene.enqueueReadBuffer(buf,CL_FALSE,0,N*sizeof(T),arg);
}

//Constant scalar
template <typename T>
inline void OpenCL::readarg(T &arg, cl::Buffer &buf,cl::CommandQueue &quene) const{
//  Opencl scalars are passed by value, therefore they cant be retrieved by the kernel function
//  quene.enqueueReadBuffer(buf,CL_FALSE,0,sizeof(T),&arg);
}




inline void OpenCL::loadProgram(const char *path){
    std::string newpath(CL_SOURCE_DIR);
        newpath += "/cl_prog";
        newpath += "/" + std::string(path);
    #if DEBUG
        std::cout << "Reading Kernel from " <<newpath <<std::endl;
    #endif
    this->contents = util::file_contents(newpath.c_str());

//    cl::Program::Sources sources;
//  //Include the read out contents from the vector file into the sources to parse
//  sources.push_back(std::make_pair(this->contents,strlen(this->contents)+1));
//  //Initiate a program from the sources
//  cl::Program program(this->context,sources);
//  if(program.build({this->device})!=CL_SUCCESS){
//      std::cout<<" Error building: "<<program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(this->device)<<"\n";
//      exit(1);
//  }
//  this->program = program;
}
