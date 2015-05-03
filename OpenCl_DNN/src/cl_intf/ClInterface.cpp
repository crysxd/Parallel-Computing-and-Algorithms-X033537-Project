/*
 * ClInterface.cpp
 *
 *  Created on: May 3, 2015
 *      Author: hedi7
 */

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

	this->context = new cl::Context(this->device);

	this->loadProgram(path);

}

template<typename I,typename O>
Cl_Interface<I,O>::~Cl_Interface() {
	delete context;
	delete contents;
}

template <typename I,typename O>
void Cl_Interface<I,O>::loadProgram(const char* path){
    cl_int err;
    this->contents = util::file_contents(path);

}

template <typename I,typename O>
void Cl_Interface<I,O>::runKernel(const char* kernelname){

    cl::Program::Sources sources;
    sources.push_back(std::make_pair(this->contents,strlen(this->contents)+1));
    cl::Program program(*(this->context),sources);
    if(program.build({this->device})!=CL_SUCCESS){
        std::cout<<" Error building: "<<program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(this->device)<<"\n";
        exit(1);
    }
    cl::Buffer buffer_A(*(this->context),CL_MEM_READ_WRITE,sizeof(float)*10);
    cl::Buffer buffer_B(*(this->context),CL_MEM_READ_WRITE,sizeof(float)*10);
    cl::Buffer buffer_C(*(this->context),CL_MEM_READ_WRITE,sizeof(float)*10);

    float A[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    float B[] = {0, 1, 2, 0, 1, 2, 0, 1, 2, 0};

    cl::CommandQueue queue(*(this->context),this->device);

    //Copy the memory from host to device
    queue.enqueueWriteBuffer(buffer_A,CL_TRUE,0,sizeof(float)*10,A);
    queue.enqueueWriteBuffer(buffer_B,CL_TRUE,0,sizeof(float)*10,B);
    queue.finish();
    cl::Kernel vectoradd(program,kernelname);
    vectoradd.setArg(0,buffer_A);
    vectoradd.setArg(1,buffer_B);
    vectoradd.setArg(2,buffer_C);

    cl::Event event;
    //The first range is the offset for the arrays, second is the global range, third the local one
    queue.enqueueNDRangeKernel(vectoradd,cl::NullRange,cl::NDRange(10),cl::NullRange,NULL,&event);
    event.wait();

    float C[10];
    //read result C from the device to array C
    queue.enqueueReadBuffer(buffer_C,CL_TRUE,0,sizeof(float)*10,C);
    queue.finish();

    std::cout<<" result: \n";
    for(int i=0;i<10;i++){
        std::cout<<C[i]<<" = "<<A[i]<< " + " << B[i] << std::endl;
    }


}

