// #include "stdafx.h"
#include "VisualisationReseauNeuronal.h"

VisualisationReseauNeuronal::VisualisationReseauNeuronal() : ReseauNeuronal() {

}

VisualisationReseauNeuronal::VisualisationReseauNeuronal(const ReseauNeuronal &rn) : ReseauNeuronal(rn) {

}

VisualisationReseauNeuronal::~VisualisationReseauNeuronal() {
	if (window) {
		window->close();
		delete window;
	}

	dernieresEtapes.clear();
}

fvec VisualisationReseauNeuronal::feedforward(fvec entrees) {

	if (listePoids.size() >= 1)
		if (entrees.n_rows != listePoids[0].n_cols) {
			std::cout << "Erreur de calcul : le nombre d'entrees donn�es ne correcpond pas au nombre d'entrees du r�seau neuronal" << std::endl;
			return fvec(0.f);
		}


	//On commence avec les entrees
	fvec sorties_couchePrecedente = entrees;

	dernieresEtapes.clear();
	dernieresEtapes.push_back(entrees);

	//Pour chaque couche,
	for (int i = 0; i < listePoids.size(); i++) {
		//on multiplie la matrice des poids de la couche actuelle avec les sorties de la couche de neurones pr�c�dentes
		sorties_couchePrecedente = listePoids[i] * sorties_couchePrecedente;
		//on ajoute le biais
		sorties_couchePrecedente += listeBiais[i];

		sorties_couchePrecedente = listeActFunc[index_act_func[i]]->activate(sorties_couchePrecedente);

		//Application de la fonction d'activation
		/*if (listeActFunc[i] == SIGMOID) {
		//On applique a chaque terme la fonction sigmoide
		std::for_each(sorties_couchePrecedente.begin(), sorties_couchePrecedente.end(), [](float &f) { f = sigmoide(f); });
		}
		else if (listeActFunc[i] == SOFTMAX) {
		//On applique a chaque terme la fonction softmax
		float s = 0;
		for (int j = 0; j < sorties_couchePrecedente.size(); j++) {
		if (isnan(sorties_couchePrecedente(j))) {
		cout << "oui";
		}
		}
		//std::for_each(sorties_couchePrecedente.begin(), sorties_couchePrecedente.end(), [](float &f) { f = sigmoide(f); });
		std::for_each(sorties_couchePrecedente.begin(), sorties_couchePrecedente.end(), [&s](float &f) { s += exp(f); });
		std::for_each(sorties_couchePrecedente.begin(), sorties_couchePrecedente.end(), [s](float &f) { f = exp(f) / s; });

		for (int j = 0; j < sorties_couchePrecedente.size(); j++) {
		if (isnan(sorties_couchePrecedente(j))) {
		cout << "oui";
		}
		}
		}*/
		if (showing)
			dernieresEtapes.push_back(sorties_couchePrecedente);


	}
	//Les donn�es de sorties sont les sorties de la deni�re couche de neurones
	fvec sorties = sorties_couchePrecedente;

	return sorties;
}

void VisualisationReseauNeuronal::open() {
	window =  new sf::RenderWindow(sf::VideoMode(wSize.x, wSize.y), "Visualisation !");
	window->clear(sf::Color::White);
	showing = true;
}

void VisualisationReseauNeuronal::show() {
	sf::Event event;
	while (window->pollEvent(event))
	{
		//Fermeture de la fen�tre lorsque l'utilisateur le souhaite
		if (event.type == sf::Event::Closed)
			window->close();
	}

	if (dernieresEtapes.size() != structure.size())
		return;

	window->clear(sf::Color::White);

	int indexResultatFin = 0;
	for (int i = 1; i < dernieresEtapes[dernieresEtapes.size() - 1].n_elem; i++) {
		if (dernieresEtapes[dernieresEtapes.size() - 1](i) > dernieresEtapes[dernieresEtapes.size() - 1](indexResultatFin))
			indexResultatFin = i;
	}

	float w = wSize.x / (dernieresEtapes.size() + 1);

	for (int i = 0; i < dernieresEtapes.size(); i++) {
		float h = wSize.y / (dernieresEtapes[i].n_elem + 1);

		for (int j = 0; j < dernieresEtapes[i].n_elem; j++) {
			sf::Color color(0, dernieresEtapes[i](j) * 255, 0);
			if (dernieresEtapes[i](j) < 0) {
				color.r = color.g;
				color.g = 0;
			}
				
			sf::CircleShape c(r);
			c.setOrigin(sf::Vector2f(r, r));
			c.setPosition(sf::Vector2f(w*(i + 1), h*(j + 1)));
			c.setFillColor(color);

			if (i < dernieresEtapes.size() - 1) {
				float h2 = wSize.y / (dernieresEtapes[i + 1].n_elem + 1);

				for (int k = 0; k < dernieresEtapes[i + 1].n_elem; k++) {
					sf::Vertex line[] =
					{
						sf::Vertex(sf::Vector2f(w*(i + 1), h*(j + 1))),
						sf::Vertex(sf::Vector2f(w*(i + 2), h2*(k + 1)))
					};

					line[0].color = color;
					line[1].color = color;
					window->draw(line, 2, sf::Lines);
				}
			}
			else if (j == indexResultatFin) {
				sf::CircleShape c2(r * 1.3);
				c2.setOrigin(sf::Vector2f(r * 1.3, r * 1.3));
				c2.setPosition(sf::Vector2f(w*(i + 1), h*(j + 1)));
				c2.setFillColor(sf::Color::Blue);
				window->draw(c2);
			}

			window->draw(c);
		}
	}

	window->display();
}

void VisualisationReseauNeuronal::close() {
	window->close();
	delete window;
	window = nullptr;
	
	showing = false;
}