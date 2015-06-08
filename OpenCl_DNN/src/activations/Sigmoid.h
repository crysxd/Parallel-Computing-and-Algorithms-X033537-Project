/*
 * Sigmoid.h
 *
 *  Created on: Apr 30, 2015
 *      Author: richman
 */

#ifndef ACTIVATIONS_SIGMOID_H_
#define ACTIVATIONS_SIGMOID_H_

#include "Activation.h"
#include "ClInterface.hpp"
#include "util.hpp"
//#include "OpenCL.hpp"
#include "CLMatrix.hpp"

class Sigmoid: public Activation {
public:
	Sigmoid();
	virtual ~Sigmoid();

	Sigmoid(const Sigmoid&);

//	template <typename T>
//	void activateMat(CL_Matrix<T> &src,CL_Matrix<T> *out);
    // activates the sigmoid function. Used the GPU. Puts the vector f onto the GPU
    // and returns the result from the device copied to the host
	virtual std::vector<float> activate(std::vector<float>& f);
    // The same as activate, except that no back-copy is used
	std::vector<util::GPU_Buffer> activateKeep(std::vector<float>& f);
    // EXPERIMENTAL! Activates the sigmoid on the device and calculates it while still
    // keeping all data on the device. The input variable in will be modified on the device!
	void chainActivate(std::vector<util::GPU_Buffer>* in);
    // Copies back the data if a buffer is given
	virtual std::vector<float> activate(std::vector<util::GPU_Buffer>& f);

	Sigmoid& operator=(const Sigmoid&);

private:
	Cl_Interface<float,float> _cl_intf;
//	OpenCL _newintf;
	const char *_kernelname = "sigmoid";
};


#endif /* ACTIVATIONS_SIGMOID_H_ */
