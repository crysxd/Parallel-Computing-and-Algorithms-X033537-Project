/*
 * GPUFeedForward.cpp
 *
 *  Created on: Jun 28, 2015
 *      Author: hedi7
 */

#include "GPUFeedForward.h"

GPUFeedForward::GPUFeedForward():_cl("gpu_kernel.cl"){
}

GPUFeedForward::~GPUFeedForward() {
    // TODO Auto-generated destructor stub
}

void GPUFeedForward::addLayer(u_int32_t neurons) {
    this->netsize.push_back(neurons);
}

GPUFeedForward::GPUFeedForward(HiddenLayer& input): GPUFeedForward(){
    this->layers.push_back(input);
}

void GPUFeedForward::createBuffers(cl::Context& context, cl::Program& program) {
    u_int32_t layerssize = this->netsize.size();
    this->sizeofnet = this->layers.size()*sizeof(HiddenLayer);
    this->sizeofinput = sizeof(float)*this->netsize[0];
    this->sizeoftarget = sizeof(float)*this->netsize.back();
    this->sizeofoutput = sizeof(float)*this->netsize.back();
    this->lastlayerind = this->netsize.size()-1;

    //Create memory buffers
//  this->netsizeBuffer = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
//      sizeof(u_int32_t)*layerssize, &(this->netsize)[0]);

//  this->layersBuffer  = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
//          sizeofnet, &(this->layers)[0]);
//  for(int i=0;i<this->layers.size();i++){
//      std::cout << this->layers.at(i).numberOfNodes<<std::endl;
//  }
//  this->inputBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeofinput);
//
//  this->targetBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeoftarget);
//
//  this->outputBuffer = cl::Buffer(context, CL_MEM_READ_WRITE, sizeofoutput);
//
//  //Create input kernels
//  inputkernel = cl::Kernel(program, "setinput");
//  inputkernel.setArg(0,layersBuffer);
//  inputkernel.setArg(1,inputBuffer);

    //Create feed forward kernels
//  feedforwardkernel = cl::Kernel(program, "feedforward");
//  feedforwardkernel.setArg(0, layersBuffer);
//  feedforwardkernel.setArg(1, this->netsizeBuffer);
//
//
//  //Create kernel to write outputs to the output buffer
//  getOutput = cl::Kernel(program, "writeOutputToBuffer");
//  getOutput.setArg(0, this->layersBuffer);
//  getOutput.setArg(1, this->outputBuffer);
//  getOutput.setArg(2, this->lastlayerind);

}

void GPUFeedForward::feedforward(std::vector<float> inputmat) {
    assert(inputmat .size() == this->netsize.at(0));
    this->createLayers();
    this->program = _cl.createProgram();
//  this->queue = _cl.createQuene(context);
    this->createBuffers(this->context,this->program);
//  for(int i=0; i < inputmat.size();i++){
//      std::cout << inputmat[i]<<std::endl;
//  }
//  queue.enqueueWriteBuffer(inputBuffer, CL_TRUE, 0, this->sizeofinput,&(inputmat)[0]);
//  queue.enqueueNDRangeKernel(this->inputkernel,cl::NullRange, cl::NDRange(this->netsize[0]), cl::NullRange);
//
//  //Now compute the output
//  unsigned int offset = 0;
//  cl::Event event;
//  for (auto i = 1u; i != this->netsize.size(); ++i)
//  {
//      queue.enqueueNDRangeKernel(this->feedforwardkernel, cl::NDRange(offset), cl::NDRange(netsize[i]), cl::NullRange,NULL,&event);
//      offset += netsize[i];
//  }
//  event.wait();
//  queue.enqueueNDRangeKernel(this->getoutputkernel,cl::NullRange, cl::NDRange(this->netsize[this->lastlayerind]), cl::NullRange);
//  float *outputArray = new float[this->netsize[this->lastlayerind]];
//  queue.enqueueReadBuffer(outputBuffer, CL_TRUE,0,sizeofoutput,outputArray);
//
//  for(int i =0; i < this->netsize[this->lastlayerind];i++){
//      std::cout << outputArray[i];
//  }
}

void GPUFeedForward::createLayers() {
    HiddenLayer *inputLayer = newInputLayer(this->netsize.at(0));
//  HiddenLayer inputLayer(this->netsize.at(0));
    layers.push_back(*inputLayer);
//  layers.push_back(inputLayer);

    //Create the rest of the layers
    for (auto i = 1u; i < this->netsize.size(); ++i)
    {
        HiddenLayer *hidlayer = newHiddenLayer(this->netsize.at(i),this->netsize.at(i-1));
        layers.push_back(*hidlayer);
//      layers.push_back(HiddenLayer(this->netsize.at(i),this->netsize.at(i-1)));
    }
}
