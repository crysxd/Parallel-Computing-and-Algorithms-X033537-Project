/*
 * HiddenLayer.h
 *
 *  Created on: Apr 30, 2015
 *      Author: richman
 */

#ifndef HIDDENLAYER_HIDDENLAYER_H_
#define HIDDENLAYER_HIDDENLAYER_H_

#include "Activation.h"
#include "Cost.h"
#include <vector>

typedef std::vector<double> doubvec;
typedef std::vector<std::vector<double>> doubmat;

class HiddenLayer {
public:
	HiddenLayer(Activation *activation,u_int32_t dim);
	virtual ~HiddenLayer();

	void propagate(doubvec &vec,doubvec weight, doubvec);

//	Getters and setters
	u_int32_t getDim() const {
		return dim;
	}


private:
	Activation *activation;
	u_int32_t dim;

};

#endif /* HIDDENLAYER_HIDDENLAYER_H_ */
