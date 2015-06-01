/*
 * HiddenLayer.h
 *
 *  Created on: Apr 30, 2015
 *      Author: richman
 */

#ifndef HIDDENLAYER_HIDDENLAYER_H_
#define HIDDENLAYER_HIDDENLAYER_H_

#include "../activations/Activation.h"
#include "../cost/Cost.h"
#include <vector>

typedef std::vector<double> doubvec;

class HiddenLayer {
public:
	HiddenLayer(Activation *activation,int in_dim,int out_dim);
	virtual ~HiddenLayer();

	void propagate(doubvec &vec);

//	Getters and setters
	int getInDim() const {
		return in_dim;
	}

	void setInDim(int inDim) {
		in_dim = inDim;
	}

	int getOutDim() const {
		return out_dim;
	}

	void setOutDim(int outDim) {
		out_dim = outDim;
	}

private:
	Activation *activation;
	int in_dim;
	int out_dim;

};

#endif /* HIDDENLAYER_HIDDENLAYER_H_ */
