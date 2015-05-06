/*
 * ClInterface.cpp
 *
 *  Created on: May 3, 2015
 *      Author: hedi7
 */

// CL_SOURCE_DIR is defined in the cmake file
#ifndef CL_SOURCE_DIR
    #define CL_SOURCE_DIR "SOURCE"
#endif

template <typename I,typename O> Cl_Interface<I,O>::Cl_Interface(const char* path) {
    cl_int err;
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
    // std::cout << "Using platform: "<<default_platform.getInfo<CL_PLATFORM_NAME>()<<"\n";
    std::cout << "Using Platform : "
            << defaultplatform.getInfo<CL_PLATFORM_NAME>() << std::endl;

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
    std::cout << "Using device: " << devicename << std::endl;

    std::string newpath(CL_SOURCE_DIR);
    newpath += "cl_prog";
    newpath += "/" + std::string(path);
    this->context = new cl::Context(this->device);
	this->loadProgram(newpath.c_str());

}


template<typename I,typename O>
Cl_Interface<I,O>::~Cl_Interface() {
	delete context;
	delete contents;
}


/*
Reads the given path in for further processing.
After loadProgram, u like usually to call runkernel()
 */
template <typename I,typename O>
void Cl_Interface<I,O>::loadProgram(const char* path){

    this->contents = util::file_contents(path);

}


template <typename I,typename O>
void Cl_Interface<I,O>::runKernel(const char* kernelname,const std::vector<std::vector<I>> &input,std::vector<std::vector<O>> *output){

    cl::Program::Sources sources;
    //Include the read out contents from the vector file into the sources to parse
    sources.push_back(std::make_pair(this->contents,strlen(this->contents)+1));
    //Initiate a program from the sources
    cl::Program program(*(this->context),sources);
    if(program.build({this->device})!=CL_SUCCESS){
        std::cout<<" Error building: "<<program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(this->device)<<"\n";
        exit(1);
    }

    // typename std::vector<std::vector<I>>::const_iterator row;
    // typename std::vector<I>::const_iterator col;
    // typename std::vector<std::vector<I>>::size_type i;

    // std::vector <cl::Buffer> buffers;
    // Generate all the buffers :

    // The quene pushes and returns the buffer objects between host and device
    cl::CommandQueue queue(*(this->context),this->device);
    cl::Kernel kernel_operator(program,kernelname);

    int n_inputargs = input.size();
    if (n_inputargs <=0 || output->size() <=0){
        std::cerr << " Error, either input or output vectors have zero length " << std::endl;
        exit(1);
    }


    for (int i = 0; i != n_inputargs; i++)
    {
        cl::Buffer buffer(*(this->context),CL_MEM_READ_WRITE,input[i].size()*sizeof(I));
        //The enqueueWriteBuffer method takes an array not an std::vector, so we need to
        //cast it to an array
        const I* arr_inp = &(input[i][0]);
        std::cout <<" Input array " << i << " : " <<arr_inp[0] <<std::endl;
        //Copy the memory from host to device
        queue.enqueueWriteBuffer(buffer,CL_TRUE,0,sizeof(I)*input[i].size(),arr_inp);
        kernel_operator.setArg(i,buffer);
    }
    //Store the current output buffers on the device
    std::vector<cl::Buffer> outputbuffers;

    for (int i = 0; i != output->size(); i++)
    {
        cl::Buffer buffer(*(this->context),CL_MEM_READ_WRITE,output[i].size()*sizeof(O));
        //The enqueueWriteBuffer method takes an array not an std::vector, so we need to
        //cast it to an array
        O* arr_out = &((*output)[i][0]);
        // Append the outputargs after the input ones
        kernel_operator.setArg(n_inputargs + i, buffer );
        outputbuffers.push_back(buffer);
        // queue.enqueueWriteBuffer(buffer,CL_TRUE,0,sizeof(O)*output[i].size(),arr_out);
    }
    queue.finish();
    // cl::Buffer buffer_A(*(this->context),CL_MEM_READ_WRITE,sizeof(float)*10);
    // cl::Buffer buffer_B(*(this->context),CL_MEM_READ_WRITE,sizeof(float)*10);
    // cl::Buffer buffer_C(*(this->context),CL_MEM_READ_WRITE,sizeof(float)*10);

    // float A[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    // float B[] = {0, 1, 2, 0, 1, 2, 0, 1, 2, 0};


    std::cout << "Running Kernel : " << kernelname << std::endl;
    // kernel_operator.setArg(0,buffer_A);
    // kernel_operator.setArg(1,buffer_B);
    // kernel_operator.setArg(2,buffer_C);

    cl::Event event;
    //The first range is the offset for the arrays, second is the global range, third the local one
    //Calculates the kernel
    queue.enqueueNDRangeKernel(kernel_operator,cl::NullRange,cl::NDRange(output->size()),cl::NullRange,NULL,&event);
    event.wait();

    O* out;
    for (int i = 0; i != output->size(); i++)
    {
        //The enqueueWriteBuffer method takes an array not an std::vector, so we need to
        //cast it to an array
        O* arr_out = &((*output)[i][0]);

        O out[10];
        // Append the outputargs after the input ones
        // Outputbuffers are the buffers which were initialized to transfer
        // data from host to device
        queue.enqueueReadBuffer(outputbuffers[i],CL_TRUE,0,sizeof(O)*output[i].size(),out);
        // out = arr_out;
        std::cout << out[0] << std::endl;
        // for (int i = 0; i < output[i].size(); i++)
        // {
        //     std::cout << arr_out[i] << " " << i << " "<<output[i].size() << std::endl;
        // }
    }
    // float C[10];
    //read result C from the device to array C
    // queue.enqueueReadBuffer(buffer_C,CL_TRUE,0,sizeof(float)*10,C);
    queue.finish();

    std::cout<<" result: \n";
    // std::cout << out[0] << std::endl;
    // for(int i=0;i<10;i++){
    //     std::cout<<C[i]<<" = "<<A[i]<< " + " << B[i] << std::endl;
    // }


}

