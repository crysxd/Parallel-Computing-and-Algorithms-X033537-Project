/*
 * ClInterface.cpp
 *
 *  Created on: May 3, 2015
 *      Author: hedi7
 */

// CL_SOURCE_DIR is defined in the cmake file
#ifndef CL_SOURCE_DIR
    #define CL_SOURCE_DIR "../src"
#endif

#ifndef DEBUG
	#define DEBUG 1
#endif

template <typename I,typename O> Cl_Interface<I,O>::Cl_Interface(const char* path) {
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
    newpath += "/" + std::string(path);
#if !DEBUG
    std::cout << "Reading Kernel from " <<newpath <<std::endl;
#endif
    this->context = cl::Context(this->device);
	this->loadProgram(newpath.c_str());

}


template<typename I,typename O>
Cl_Interface<I,O>::~Cl_Interface() {
    delete [] contents;
}

template<typename I, typename O>
Cl_Interface<I,O>::Cl_Interface(const Cl_Interface &other){
    if(this != &other){
        this->platform = other.platform;
        this->device = other.device;
        this-> context = other.context;
        this->contents = new char[strlen(other.contents)];
        strcpy(this->contents,other.contents);
    }
}


/*
Reads the given path in for further processing.
Keep in mind that path is pointing at the cl_prog directory
After loadProgram, u like usually to call runkernel()
 */
template <typename I,typename O>
void Cl_Interface<I,O>::loadProgram(const char* path){

    this->contents = util::file_contents(path);
}


template<typename I,typename O>
void Cl_Interface<I,O>::readResult(std::vector<util::GPU_Buffer> outputbuffers,std::vector<std::vector<O>> *output){
    cl::CommandQueue queue(this->context,this->device);


    ////////////////////////
    // Read in the result //
    ////////////////////////
    for (int i = 0; i != output->size(); i++)
    {
        //The enqueueWriteBuffer method takes an array not an std::vector, so we need to
        //cast it to an array
        O* arr_out = &((*output)[i][0]);
        //Read out the results
//        std::cout << sizeof(O)*(*output)[i].size() << std::endl;
        queue.enqueueReadBuffer(outputbuffers[i].buffer,CL_TRUE,0,sizeof(O)*outputbuffers[i].datalength,arr_out);
    }
    // Wait for the reading buffer to complete
    queue.finish();
}


template<typename I, typename O>
void Cl_Interface<I,O>::chainKernel(const char* kernelname,const int blocksize,const std::vector<util::GPU_Buffer> &inputbuffer,const std::vector<util::GPU_Buffer> &outputbuffer){
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



    int n_inputargs = inputbuffer.size();
    if (n_inputargs <=0 || outputbuffer.size() <=0){
        std::cerr << " Error, either input or output vectors have zero length " << std::endl;
        exit(1);
    }

     for (int i = 0; i != n_inputargs; i++)
    {
        //The enqueueWriteBuffer method takes an array not an std::vector, so we need to
        //cast it to an array
        kernel_operator.setArg(i,inputbuffer[i].buffer);
    }

    for (int i = 0; i != outputbuffer.size(); i++)
    {
        //The enqueueWriteBuffer method takes an array not an std::vector, so we need to
        //cast it to an array
        // Append the outputargs after the input ones
        kernel_operator.setArg(n_inputargs + i, outputbuffer[i].buffer);
    }
    cl::Event event;
    // cout << "INPUT " << input[0].datalength << " OUTPUT " << output[0].datalength <<endl;
    ///////////////////////////////////
    // Calculation is being executed //
    ///////////////////////////////////
    //The first range is the offset for the arrays, second is the global range, third the local one
    cl_int errcode = queue.enqueueNDRangeKernel(kernel_operator,cl::NullRange,cl::NDRange(outputbuffer[0].datalength),cl::NDRange(blocksize),NULL,&event);
    if (errcode != 0 ){
        cerr << "ERROR while executing the Kernel " << kernelname  << " Errorcode : " << errcode << endl;
    }
    event.wait();

}


/*
Runs the given kernel with the input parameters and the output parameters.
Input needs to be an ordered std::vector of arguments, for the given kernel.
e.g if the kernel is __kernel void something (float *a, float *b)
then input needs to be a list with 2 elements, each of them representing one float
array.

output is only indicating the amount of output parameters. So it only needs to be initialized
with the size of the output parameters, not an actual value. Values will be nonetheless
overwritten.

and example call with two arrays (a and c), where in a the subarrays will be processed
and c is the resulting array:

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

std::vector<std::vector<float>> c;
std::vector<float> d(3);
c.push_back(d);

Cl_Interface<float,float> clinterface("vectoradd.cl");
clinterface.runKernel("vector_add_gpu",a,&c);
 */
template <typename I,typename O>
void Cl_Interface<I,O>::runKernel(const char* kernelname,const int blocksize,const std::vector<std::vector<I>> &input,std::vector<std::vector<O>> *output){
    std::vector<util::GPU_Buffer> outputbuffers = this->runKernelBuffer(kernelname,blocksize,input,output);

//     ////////////////////////////////////////////////////
//     // Initalize the transfer and executeable objects //
//     ////////////////////////////////////////////////////
//     // Queue is reponsible for transferring data and the kernel_operator executes the code
//     // The quene pushes and returns the buffer objects between host and device
    cl::CommandQueue queue(this->context,this->device);


    ////////////////////////
    // Read in the result //
    ////////////////////////
    for (int i = 0; i != output->size(); i++)
    {
        //The enqueueWriteBuffer method takes an array not an std::vector, so we need to
        //cast it to an array
        O* arr_out = &((*output)[i][0]);

        //Read out the results
//        std::cout << sizeof(O)*(*output)[i].size() << std::endl;
        queue.enqueueReadBuffer(outputbuffers[i].buffer,CL_TRUE,0,sizeof(O)*(*output)[i].size(),arr_out);
    }
    // Wait for the reading buffer to complete
    queue.finish();


}





template <typename I,typename O>
std::vector<util::GPU_Buffer> Cl_Interface<I,O>::runKernelBuffer(const char* kernelname,const int blocksize,const std::vector<std::vector<I>> &input,std::vector<std::vector<O>> *output){
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

    int n_inputargs = input.size();
    if (n_inputargs <=0 || output->size() <=0){
        std::cerr << " Error, either input or output vectors have zero length " << std::endl;
        exit(1);
    }
    ////////////////////////////////////////////////////////
    // Write the input arrays from the host to the device //
    ////////////////////////////////////////////////////////

    for (int i = 0; i != n_inputargs; i++)
    {
        cl::Buffer buffer(this->context,CL_MEM_READ_WRITE,input[i].size()*sizeof(I));
        //The enqueueWriteBuffer method takes an array not an std::vector, so we need to
        //cast it to an array
        const I* arr_inp = &(input[i][0]);
#if !DEBUG
        for(auto j=0u;j < input[i].size() ;j++){
            std::cout <<" Input array " << i << " : Value  " <<arr_inp[j] << " size : " << sizeof(I)*input[i].size() <<std::endl;
        }
#endif
        //Copy the memory from host to device
        queue.enqueueWriteBuffer(buffer,CL_TRUE,0,sizeof(I)*input[i].size(),arr_inp);
        kernel_operator.setArg(i,buffer);
    }
    //Store the current output buffers on the device, need them later to receive the data
    std::vector<util::GPU_Buffer> outputbuffers;

    //store the amount of elements in one of the vectors
    int n_gpu_range = (*output)[0].size();
    for (int i = 0; i != output->size(); i++)
    {
        cl::Buffer buffer(this->context,CL_MEM_READ_WRITE,(*output)[i].size()*sizeof(O));
        //The enqueueWriteBuffer method takes an array not an std::vector, so we need to
        //cast it to an array
        // Append the outputargs after the input ones
        kernel_operator.setArg(n_inputargs + i, buffer );

        util::GPU_Buffer gpubuf = {buffer,n_gpu_range};
        outputbuffers.push_back(gpubuf);

    }
    // Wait for the quene to tansfer the data
    queue.finish();

    #if !DEBUG
    std::cout << "Running Kernel : " << kernelname << std::endl;
    #endif

    cl::Event event;
    ///////////////////////////////////
    // Calculation is being executed //
    ///////////////////////////////////
    //The first range is the offset for the arrays, second is the global range, third the local one
    cl_int ret = queue.enqueueNDRangeKernel(kernel_operator,cl::NullRange,cl::NDRange(n_gpu_range),cl::NDRange(blocksize),NULL,&event);
    if (ret != 0) {
        cerr<< " Error when Executing the kernel. Code: " << ret << endl;
    }
    event.wait();

    return outputbuffers;
}

