// #include "stdafx.h"
#include "Activation.h"

Activation::Activation() {

}

Sigmoid::Sigmoid() {

}

Softmax::Softmax() {

}

float Sigmoid::sigmoid(float x) {
	return 1 / (1 + exp(-x));
}

fvec Sigmoid::activate(fvec z) const {
	fvec v = z;
	std::for_each(v.begin(), v.end(), [](float &f) { f = sigmoid(f); });
	return v;
}

fvec Sigmoid::derivative(fvec z) const {
	fvec v = z;
	std::for_each(v.begin(), v.end(), [](float &f) { f = sigmoid(f) * (1 - sigmoid(f)); });
	return v;
}

fvec Softmax::activate(fvec z) const {
	fvec v = z;
	float s = 0;
	std::for_each(v.begin(), v.end(), [&s](float &f) { s += exp(f); });
	std::for_each(v.begin(), v.end(), [s](float &f) { f = exp(f) / s; });

	return v;
}

fvec Softmax::derivative(fvec z) const {
	fvec v = activate(z);
	std::for_each(v.begin(), v.end(), [](float &f) { f *= (1 - f); });
	return v;
}