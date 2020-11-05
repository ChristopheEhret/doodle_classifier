#pragma once
//#include "stdafx.h"
#include <armadillo>

using namespace arma;

class Activation {
public :
	Activation();

	virtual fvec activate(fvec z) const { return fvec(); }
	virtual fvec derivative(fvec z) const { return fvec(); }
	virtual Activation* copy() const { return nullptr; }
};

class Sigmoid : public Activation {
public : 
	Sigmoid();

	static float sigmoid(float x);

	fvec activate(fvec z) const;
	fvec derivative(fvec z) const;
	Activation* copy() { return new Sigmoid(); }
};

class Softmax : public Activation {
public:
	Softmax();

	fvec activate(fvec z) const;
	fvec derivative(fvec z) const;
	Activation* copy() { return new Softmax(); }
};