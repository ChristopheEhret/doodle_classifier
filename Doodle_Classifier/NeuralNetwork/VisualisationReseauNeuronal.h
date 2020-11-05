#pragma once

#include "ReseauNeuronal.h"
#include <SFML/Graphics.hpp>

class VisualisationReseauNeuronal : public ReseauNeuronal {
public:
	VisualisationReseauNeuronal();
	VisualisationReseauNeuronal(const ReseauNeuronal &rn);

	virtual ~VisualisationReseauNeuronal();

	virtual fvec feedforward(fvec input);
	
	void open();
	void close();
	void show();
private:
	sf::RenderWindow *window = nullptr;
	std::vector<fvec> dernieresEtapes = {};

	const sf::Vector2f wSize = sf::Vector2f(400, 400);
	const int r = 15;

	bool showing = false;
};