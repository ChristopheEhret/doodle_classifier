// #include "stdafx.h"
#include "ReseauNeuronal.h"
#include <algorithm>

float dsigmoide(float &x) {
	return sigmoide(x) * (1 - sigmoide(x));
}

ReseauNeuronal::ReseauNeuronal() {
	//Cr�e une structure vide
	structure = {};

	//Cr�e des listes vides
	listePoids = {};
	listeBiais = {};
	listeActFunc = {new Sigmoid(), new Softmax()};
	index_act_func = {};

	//Initialise le g�n�rateur de nombres al�atoires
	generator = new std::minstd_rand(std::chrono::system_clock::now().time_since_epoch().count());	
}

ReseauNeuronal::ReseauNeuronal(std::vector<int> struc) {

	//Recup�re la structure pass�e en argument
	structure = struc;

	//Initialise le g�n�rateur de nombres al�atoires
	generator = new std::minstd_rand(std::chrono::system_clock::now().time_since_epoch().count());

	//Pour chaque couche de la structure, on cr�e une matrice qui contient les poids liant la couche et la couche suivante
	for (int i = 0; i < structure.size() - 1; i++) {
		//Creation de la matrice des poids
		fmat p(structure[i + 1], structure[i]);
		//Rempli la matrice des poids avec des valeurs al�atoires
		std::for_each(p.begin(), p.end(), [this](float &f) { f = this->random(); });
		//p.randu();
		
		//Creation du vecteur des biais
		fvec b(structure[i + 1]);
		//Rempli la matrice des poids avec des valeurs al�atoires
		std::for_each(b.begin(), b.end(), [this](float &f) { f = this->random(); });
		//b.randu();

		listePoids.push_back(p);
		listeBiais.push_back(b);
		learningRates.push_back(learningRate);
	}

	listeActFunc = { new Sigmoid(), new Softmax() };
	for (int i = 0; i < listePoids.size(); i++) {
		index_act_func.push_back(SIGMOID);
	}
}

ReseauNeuronal::ReseauNeuronal(std::vector<fmat> poids, std::vector<fvec> biais) {
	//V�rifie de la coherence du nombre de couches
	if (biais.size() != poids.size()) {
		std::cout << "Erreur d'initialisation du r�seau neuronal : le nombre de couches de biais ne correspond pas au nombre de couches de neurones " << std::endl;
		*this = ReseauNeuronal();
		return;
	}

	//Met le nombre d'entrees dans la structure
	if (poids.size() != 0)
		structure.push_back(poids[0].n_cols);
	
	//On se balade dans les couches
	for (int i = 0; i < poids.size(); i++) {
		//On v�rifie la coherence des tailles des diff�rentes matrices
		if (biais[i].n_rows != poids[i].n_rows) {
			std::cout << "Erreur d'initialisation du r�seau neuronal : le nombre de biais ne correspond pas au nombre de neurones " << std::endl;
			std::cout << "(Couche " << i + 1 << ")" << std::endl;
			*this = ReseauNeuronal();
			return;
		}

		//V�rifie qu'il y a bien autant d'entrees dans la couche actuelle que de sortie dans la couche pr�c�dente
		if(i >= 1 && poids[i - 1].n_rows != poids[i].n_cols) {
			std::cout << "Erreur d'initialisation du r�seau neuronal : le nombre de neurones n'est pas coh�rent d'une couche � l'autre " << std::endl;
			std::cout << "(Entre la couche " << i << " et la couche " << i+1 << ")" << std::endl;
			*this = ReseauNeuronal();
			return;
		}

		//Initialise le r�seau
		structure.push_back(poids[i].n_rows);

		listePoids.push_back(poids[i]);
		listeBiais.push_back(biais[i]);
		learningRates.push_back(learningRate);
	}

	listeActFunc = { new Sigmoid(), new Softmax() };
	for (int i = 0; i < listePoids.size(); i++) {
		index_act_func.push_back(SIGMOID);
	}

	//Initialise le g�n�rateur de nombres al�atoires
	generator = new std::minstd_rand(std::chrono::system_clock::now().time_since_epoch().count());

}

ReseauNeuronal::ReseauNeuronal(std::vector<std::vector<std::vector<float>>> poids, std::vector<std::vector<float>> biais) {
	//V�rifie de la coherence du nombre de couches
	if (biais.size() != poids.size()) {
		std::cout << "Erreur d'initialisation du r�seau neuronal : le nombre de couches de biais ne correspond pas au nombre de couches de neurones " << std::endl;
		*this = ReseauNeuronal();
		return;
	}

	if (poids.size() != 0) 
		if(poids[0].size() != 0)
			structure.push_back(poids[0][0].size());
	
	//On se balade dans les couches
	for (int i = 0; i < poids.size(); i++) {
		//On v�rifie la coherence des tailles des diff�rentes matrices
		if (poids[i].size() == 0 || biais[i].size() != poids[i].size()) {
			std::cout << "Erreur d'initialisation du r�seau neuronal : le nombre de biais ne correspond pas au nombre de neurones " << std::endl;
			std::cout << "(Couche " << i + 1 << ")" << std::endl;
			*this = ReseauNeuronal();
			return;
		}

		//V�rifie qu'il y a bien autant d'entrees dans la couche actuelle que de sortie dans la couche pr�c�dente
		if (i >= 1 && poids[i - 1].size() != poids[i][0].size()) {
			std::cout << "Erreur d'initialisation du r�seau neuronal : le nombre de neurones n'est pas coh�rent d'une couche � l'autre " << std::endl;
			std::cout << "(Entre la couche " << i << " et la couche " << i + 1 << ")" << std::endl;
			*this = ReseauNeuronal();
			return;
		}


		//Initialise le r�seau
		structure.push_back(poids[i].size());

		//Matrices des poids
		fmat p(poids[i].size(), poids[i][0].size());
		//On se balade dans les neurones de la couche
		for (int j = 0; j < poids[i].size(); j++) {
			//V�rifie la coh�rence de la taille de la matrice
			if (poids[i][j].size() != p.n_cols) {
				std::cout << "Erreur d'initialisation du r�seau neuronal : le nombre de poids n'est pas coh�rent d'un neurone � l'autre " << std::endl;
				std::cout << "(Couche " << i + 1 << "; Neurone " << j + 1 << ")" << std::endl;
			}

			//On se balade dans les poids du neurone
			for (int k = 0; k < poids[i][j].size(); k++) {
				p(j, k) = poids[i][j][k];
			}
		}

		//Vecteur des biais
		fvec b = biais[i];

		listePoids.push_back(p);
		listeBiais.push_back(b);
		learningRates.push_back(learningRate);
	}

	listeActFunc = { new Sigmoid(), new Softmax() };
	for (int i = 0; i < listePoids.size(); i++) {
		index_act_func.push_back(SIGMOID);
	}
	//Initialise le g�n�rateur de nombres al�atoires
	generator = new std::minstd_rand(std::chrono::system_clock::now().time_since_epoch().count());
}

ReseauNeuronal::ReseauNeuronal(const ReseauNeuronal &rn) {
	//Contructeur de copie

	structure = rn.structure;
	listeActFunc.clear();
	listeActFunc = { new Sigmoid(), new Softmax() };
	index_act_func = rn.index_act_func;

	fonctionCout = rn.fonctionCout;
	learningRates = rn.learningRates;

	listePoids = rn.listePoids;
	listeBiais = rn.listeBiais;

	generator = new std::minstd_rand(std::chrono::system_clock::now().time_since_epoch().count());
}

ReseauNeuronal::~ReseauNeuronal() {
	delete generator;

	for (Activation* a : listeActFunc)
		delete a;
}

float ReseauNeuronal::random(float val_min, float val_max) {
	//Distribution r�elle entre val_min et val_max
	std::uniform_real_distribution<float> distribution(val_min, val_max);

	//G�n�re un nombre al�atoire
	return distribution(*generator);
}

void ReseauNeuronal::setActivationFuction(activationIndex i, int c) {
	if (c >= listeActFunc.size()) {
		//Si c est trop grand -> erreur
		cout << "Erreur d'initialisation : La fonction d'activation ne peut pas �tre attibu�e � la couche " << c + 1;
		cout << "car le r�seau n'a que " << listePoids.size() << " couches." << endl;
		return;
	}
	
	//Remplace la fonction d'activation
	index_act_func[c] = i;
}

void ReseauNeuronal::setActivationFuction(vector<activationIndex> listef) {
	if (listef.size() != listePoids.size()) {
		//Si listef est trop grand -> erreur
		cout << "Erreur d'initialisation : le nombre de fonctions d'activation n'est pas le m�me que le nombre de couche." << endl;
		return;
	}

	//Remplace les fonctions d'activation
	index_act_func = listef;
}

void ReseauNeuronal::setLearningRates(vector<float> nlr) {
	if (nlr.size() !=learningRates.size()) {
		//Si la liste est trop grande -> erreur
		cout << "Erreur d'initialisation : le nombre de learning rates n'est pas le m�me que le nombre de couche." << endl;
		return;
	}

	//Remplace les fonctions d'activation
	learningRates = nlr;
}


fvec ReseauNeuronal::feedforward(fvec entrees) {

	if (listePoids.size() >= 1)
		if (entrees.n_rows != listePoids[0].n_cols) {
			std::cout << "Erreur de calcul : le nombre d'entrees donn�es ne correcpond pas au nombre d'entrees du r�seau neuronal" << std::endl;
			return fvec(0.f);
		}


	//On commence avec les entrees
	fvec sorties_couchePrecedente = entrees;

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
	}

	//Les donn�es de sorties sont les sorties de la deni�re couche de neurones
	fvec sorties = sorties_couchePrecedente;

	return sorties;
}

void ReseauNeuronal::backPropagation(fvec entrees, fvec cible) {
	if (listePoids.size() >= 1) {
		if (entrees.n_rows != listePoids[0].n_cols) {
			std::cout << "Erreur de calcul : le nombre d'entrees donn�es ne correcpond pas au nombre d'entrees du r�seau neuronal" << std::endl;
			return;
		}
		if (cible.n_rows != listePoids[listePoids.size() - 1].n_rows) {
			std::cout << "Erreur de calcul : le nombre de sorties donn�es ne correcpond pas au nombre de sorties du r�seau neuronal" << std::endl;
			return;
		}
	}


	//On calcule la sortie de ce r�seau neuronal, en gardant les sorties de la couche concern�e � chaque �tape
	std::vector<fvec> listeEtapes;
	std::vector<fvec> listeActivations;

	//On commence avec les entrees
	fvec sorties_couchePrecedente = entrees;

	listeEtapes.push_back(sorties_couchePrecedente);
	listeActivations.push_back(sorties_couchePrecedente);

	//Pour chaque couche,
	for (int i = 0; i < listePoids.size(); i++) {
		//on multiplie la matrice des poids de la couche actuelle avec les sorties de la couche de neurones pr�c�dentes
		sorties_couchePrecedente = listePoids[i] * sorties_couchePrecedente;
		//on ajoute le biais
		sorties_couchePrecedente += listeBiais[i];

		listeEtapes.push_back(sorties_couchePrecedente);

		sorties_couchePrecedente = listeActFunc[index_act_func[i]]->activate(sorties_couchePrecedente);

		//On enregistre les sorties de la couche en question
		listeActivations.push_back(sorties_couchePrecedente);

		//listeActivations[i].print();
		//listeEtapes[i].print();
	}

	//Les donn�es de sorties sont les sorties de la deni�re couche de neurones
	fvec sorties = sorties_couchePrecedente;


	if (false) {

		//On calcule l'erreur de la couche de sortie
		fvec erreurs_couchePrecedente = cible - sorties;
		//	fvec erreurs_couchePrecedente = sorties - cible;

			//On channge chaque couche en partant de la derni�re et en remontant jusqu'� la premi�re
		for (int i = listePoids.size() - 1; i >= 0; i--) {
			//Calcul du gandiant de la couche actuelle
			//G = lr*E*(O*(1-O))
			//G : gradiant
			//lr : learning rate
			//E : Erreur de la couche
			//O : Sorties de la couche
			//* : multiplication terme � terme
			fvec gradian_coucheActuelle = listeActivations[i + 1];

			for (int n = 0; n < gradian_coucheActuelle.n_elem; n++) {
				float x = gradian_coucheActuelle(n);

				x *= 1 - x;
				x *= erreurs_couchePrecedente(n);
				x *= learningRate;

				gradian_coucheActuelle(n) = x;
			}

			//Calcul du changement de la matrice des poids
			//delta = G*It 
			//It : transpos�e de la matrice
			fmat delta_poids = gradian_coucheActuelle * listeActivations[i].t();

			listePoids[i] += delta_poids;
			listeBiais[i] += gradian_coucheActuelle;

			//Calcul de l'erreur de la couche pr�c�dente 
			erreurs_couchePrecedente = listePoids[i].t() * erreurs_couchePrecedente;
		}
	}
	else {
		vector<fvec> listeErreurs;
		std::fill_n(std::back_inserter(listeErreurs), listePoids.size(), fvec());
		fvec erreur;

		if (fonctionCout == QUADRATIC)
			erreur = sorties - cible;
		else if (fonctionCout == CROSSENTROPY) {
			erreur = fvec(sorties.n_elem);
			for (int n = 0; n < erreur.n_elem; n++) {
				erreur(n) = (sorties(n) - cible(n)) / (sorties(n) * (1 - sorties(n)));
			}
		}



		for (int i = listePoids.size() - 1; i >= 0; i--) {
			fvec der = listeActFunc[index_act_func[i]]->derivative(listeEtapes[i + 1]);
			for (int n = 0; n < erreur.n_elem; n++) {
				/*if (listeActFunc[i] == SIGMOID)
					erreur(n) = erreur(n) * dsigmoide(listeEtapes[i + 1](n));
					//erreur(n) *= listeActivations[i + 1](n) * (1 - listeActivations[i + 1](n));
				else if (listeActFunc[i] == SOFTMAX) {
					erreur(n) *= listeActivations[i + 1](n) * (1 - listeActivations[i + 1](n));
					if (isnan(erreur(n))) {
						cout << "oui";
					}
				}*/
				erreur(n) *= der(n);
			}

			listeErreurs[i] = erreur;

			//			erreur.print();

						//On change les poids avant le calcul de l'erreur pr�c�dente => pas correct
						//Calcul de toutes les erreurs avant application? (voir carnet p.5) 

			/*if (regularisation)
				listePoids[i] *= 1 - (learningRates[i] * param_regularisation);
			
			listePoids[i] -= learningRates[i] * (erreur * listeActivations[i].t());
			listeBiais[i] -= learningRates[i] * erreur;*/
			//learningRates[i] *= 0.999999;
			
			erreur = listePoids[i].t() * erreur;
		}

		for (int i = listePoids.size() - 1; i >= 0; i--) {
			if (regularisation)
				listePoids[i] *= 1 - (learningRates[i] * param_regularisation);

			listePoids[i] -= learningRates[i] * (listeErreurs[i] * listeActivations[i].t());
			listeBiais[i] -= learningRates[i] * listeErreurs[i];
		}
	}
}

void ReseauNeuronal::backPropagation(vector<fvec> entrees, vector<fvec> cibles) {
	if (entrees.size() != cibles.size()) {
		std::cout << "Il n'y a pas autant d'entrees que de cibles pour l'entrainement du RN. Celui-ci n'a pas ete effectue." << std::endl;
		return;
	}

	if (listePoids.size() >= 1) {
		for (int i = 0; i < entrees.size(); i++) {
			if (entrees[i].n_rows != listePoids[0].n_cols) {
				std::cout << "Erreur de calcul : la taille des entr�es donn�es ne correspond pas au nombre d'entrees du r�seau neuronal" << std::endl;
				return;
			}
			if (cibles[i].n_rows != listePoids[listePoids.size() - 1].n_rows) {
				std::cout << "Erreur de calcul : le taille des cibles donn�es ne correspond pas au nombre de sorties du r�seau neuronal" << std::endl;
				return;
			}
		}
	}

	vector<vector<fvec>> listeEtapes = {};
	vector<vector<fvec>> listeActivations = {};
	std::fill_n(std::back_inserter(listeEtapes), entrees.size(), vector<fvec>());
	std::fill_n(std::back_inserter(listeActivations), entrees.size(), vector<fvec>());


	/*for (fvec e : entrees) {
		vector<fvec> etape = {e};
		vector<fvec> activation = {e};

		fvec sortie_couche_prec = e;
		for (int i = 0; i < listePoids.size(); i++) {
			sortie_couche_prec = listePoids[i] * sortie_couche_prec;
			sortie_couche_prec += listeBiais[i];
			
			etape.push_back(sortie_couche_prec);

			sortie_couche_prec = listeActFunc[index_act_func[i]]->activate(sortie_couche_prec);

			activation.push_back(sortie_couche_prec);
		}

		listeEtapes.push_back(etape);
		listeActivations.push_back(activation);
	}*/

	for (int i = 0; i < listePoids.size(); i++) {
		for (int j = 0; j < entrees.size(); j++) {
			
			if (i == 0) {
				listeEtapes[j].push_back(entrees[j]);
				listeActivations[j].push_back(entrees[j]);
			}

			entrees[j] = listePoids[i] * entrees[j];
			entrees[j] += listeBiais[i];

			listeEtapes[j].push_back(entrees[j]);

			entrees[j] = listeActFunc[index_act_func[i]]->activate(entrees[j]);

			listeActivations[j].push_back(entrees[j]);
		}
	}

	for (int i = 0; i < entrees.size(); i++) {
		fvec erreur;
		fvec sortie = listeActivations[i][listePoids.size()];

		if (fonctionCout == QUADRATIC)
			erreur = sortie  - cibles[i];
		else if (fonctionCout == CROSSENTROPY) {
			erreur = fvec(sortie.n_elem);
			for (int n = 0; n < erreur.n_elem; n++) {
				erreur(n) = (sortie(n) - cibles[i](n)) / (sortie(n) * (1 - sortie(n)));
			}
		}

		for (int j = listePoids.size() - 1; j >= 0; j--) {
			fvec der = listeActFunc[index_act_func[j]]->derivative(listeEtapes[i][j + 1]);
			for (int n = 0; n < erreur.n_elem; n++)
				erreur(n) *= der(n);

			if (regularisation)
				listePoids[j] *= 1 - (learningRates[j] * param_regularisation / entrees.size());

			listePoids[j] -= learningRates[j] * (erreur * listeActivations[i][j].t()) / entrees.size();
			listeBiais[j] -= learningRates[j] * erreur / entrees.size();
			
			erreur = listePoids[j].t() * erreur;
		}
	}
}

//Affiche les poids du r�seau neuronal
void ReseauNeuronal::print() {
	for (int i = 0; i < listePoids.size(); i++) {
		std::cout << "---------------------" << std::endl;
		std::cout << "Couche " << i + 1 << std::endl;
		std::cout << "Poids : " << std::endl;
		listePoids[i].print();
		std::cout << "Biais : " << std::endl;
		listeBiais[i].print();
	}
}

void ReseauNeuronal::save(string filePath) {
	//Sauvegarde de la structure et de chaque �l�ments des chaque matrices (poids ou biais)
	//Rep�res:
	//[ : Taille de la structure (nb de couches)
	//{ : Debut de la structure
	//} : Fin de la structure
	//> : Debut d'une matrice de poids
	//< : Debut d'une matrice de biais
	//; : Fin d'une matrice
	//+ : Fonctions d'activation
	//$ : Fonction cout
	//* : Fin du fichier
	
	ofstream out(filePath);

	//Taille de la structure
	out << "[ " << structure.size() << endl;

	//Structure
	out << "{ ";

	for (int i = 0; i < structure.size(); i++) {
		out << structure[i] << " ";
	}

	out << "}" << endl;

	//Matrices
	for (int k = 0; k < listePoids.size(); k++) {

		//Matrice de poids
		out << "> ";

		for (int i = 0; i < listePoids[k].n_rows; i++) {
			for (int j = 0; j < listePoids[k].n_cols; j++) {
				out << listePoids[k](i, j) << " ";
			}
			out << "! ";
		}

		out << ";" << endl;

		//Matrice de biais
		out << "< ";

		for (int i = 0; i < listeBiais[k].n_elem; i++) {
			out << listeBiais[k](i) << " ";
		}

		out << ";" << endl;
	}
	
	//Fonctions d'activation
	out << "+ ";
	for (activationIndex f : index_act_func)
		out << f << " ";
	out << endl;

	//Fonction cout
	out << "$ " << fonctionCout << endl;


	//Fin du fichier
	out << "*" << endl;

	out.close();
}

//Sauvegarde le r�seau neuronal sous forme de listes, utils� pour l'impl�mentation du r�seau neuronal dans un programme arduino
void ReseauNeuronal::save_arduino(string filePath) {

	ofstream out(filePath);

	for (int i = 0; i < listePoids.size(); i++) {

		out << "{";
		for (int j = 0; j < listePoids[i].n_rows; j++) {
			if (j != 0)
				out << ",";
			out << "{";
			for (int k = 0; k < listePoids[i].n_cols; k++) {
				if (k != 0)
					out << ",";
				out << listePoids[i](j, k);
			}
			out << "}";
		}
		out << "}" << endl;

		out << "{";
		for (int j = 0; j < listeBiais[i].n_elem; j++) {
			if (j != 0)
				out << ",";
			out << listeBiais[i](j);
		}
		out << "}" << endl;
	}
}

bool ReseauNeuronal::load(string filePath) {
	//Chargement de la structure et de chaque �l�ments des chaque matrices (poids ou biais)
	//Rep�res:
	//[ : Taille de la structure (nb de couches)
	//{ : Debut de la structure
	//} : Fin de la structure
	//> : Debut d'une matrice de poids
	//< : Debut d'une matrice de biais
	//; : Fin d'une matrice
	//+ : Fonctions d'activation
	//$ : Fonction cout
	//* : Fin du fichier
	//Si il y a des in.get(c) qui semblent seuls et inutiles, ils servent � lire des caract�res en trop, tels que des espaces ou des retours � la ligne

	//Chargement du fichier
	ifstream in(filePath);

	if (in.fail()) {
		cout << "Erreur lors du chargement du reseau : " << filePath << " n'existe pas." << endl;
		*this = ReseauNeuronal();
		in.close();
		return false;
	}


	///Taille de la structure

	char c = ' ';
	//V�rification du rep�re
	if (!in.get(c) || c != '[') {
		cout << "Erreur lors du chargement du reseau : " << filePath << " ne contient pas les donn�es de la taille de la structure." << endl;
		*this = ReseauNeuronal();
		in.close();
		return false;
	}

	int strucSize = -1;
	in >> strucSize;

	//Si pas/mal lue, erreur
	if (strucSize == -1) {
		cout << "Erreur lors du chargement du reseau : " << filePath << " ne contient pas les donn�es de la taille de la structure." << endl;
		*this = ReseauNeuronal();
		in.close();
		return false;
	}


	///Structure

	in.get(c);
	//V�rification du rep�re
	if (!in.get(c) || c != '{') {
		cout << "Erreur lors du chargement du reseau : " << filePath << " ne contient pas les donnees de la structure." << endl;
		*this = ReseauNeuronal();
		in.close();
		return false;
	}

	//Lecture de la structure
	vector<int> nStructure;
	for (int i = 0; i < strucSize; i++) {
		//Lecture des valeurs
		int s = -1;
		in >> s;

		//Si pas/mal lue, erreur
		if (s == -1) {
			cout << "Erreur lors du chargement du reseau : " << filePath << " ne contient pas les donnees de la structure." << endl;
			*this = ReseauNeuronal();
			in.close();
			return false;

		}

		nStructure.push_back(s);
	}
	
	in.get(c);
	//V�rification du rep�re
	if (!in.get(c) || c != '}') {
		cout << "Erreur lors du chargement du reseau : " << filePath << " ne contient pas les donnees de la structure." << endl;
		*this = ReseauNeuronal();
		in.close();
		return false;
	}

	


	///Reseau neuronal
	/*this->~ReseauNeuronal();
	*this = ReseauNeuronal(nStructure);*/

	structure = nStructure;

	listePoids.clear();
	listeBiais.clear();
	index_act_func.clear();

	//Pour chaque couche de la structure, on cr�e une matrice qui contient les poids liant la couche et la couche suivante
	for (int i = 0; i < structure.size() - 1; i++) {
		//Creation de la matrice des poids
		fmat p(structure[i + 1], structure[i]);
		p.zeros();
		//p.randu();

		//Creation du vecteur des biais
		fvec b(structure[i + 1]);
		b.zeros();
		//b.randu();

		listePoids.push_back(p);
		listeBiais.push_back(b);
	}

	///A ENLEVER
	generator = new std::minstd_rand(std::chrono::system_clock::now().time_since_epoch().count());

	//Couche par couche
	for (int k = 0; k < listePoids.size(); k++) {
		in.get(c);
		//V�rification du rep�re
		if (!in.get(c) || c != '>') {
			cout << "Erreur lors du chargement du reseau : " << filePath << " ne contient pas les donnees des poids." << endl;
			cout << "Couche : " << k << endl;
			*this = ReseauNeuronal();
			in.close();
			return false;
		}


		///Poids des neurones

		//Neurone par neurone
		for (int i = 0; i < listePoids[k].n_rows; i++) {
			for (int j = 0; j < listePoids[k].n_cols; j++) {
				//Lecture des valeurs
				float p = -2;
				in >> p;

				//Si pas/mal lue, erreur
				if (p == -2) {
					cout << "Erreur lors du chargement du reseau : " << filePath << " ne contient pas les donnees des poids." << endl;
					cout << "Couche : " << k << "; Neurone : " << i << "; Poids : " << j << endl;
					*this = ReseauNeuronal();
					in.close();
					return false;
				}

				listePoids[k](i, j) = p;
			}

			in.get(c);
			//V�rification du rep�re
			if (!in.get(c) || c != '!') {
				cout << "Erreur lors du chargement du reseau : " << filePath << " ne contient pas les donnees des poids." << endl;
				cout << "Couche : " << k << "; Neurone : " << i << endl;
				*this = ReseauNeuronal();
				in.close();
				return false;
			}
		}

		in.get(c);
		//V�rification du rep�re
		if (!in.get(c) || c != ';') {
			cout << "Erreur lors du chargement du reseau : " << filePath << " ne contient pas les donnees des poids." << endl;
			cout << "Couche : " << k << endl;
			*this = ReseauNeuronal();
			in.close();
			return false;
		}


		///Biais des neurones

		in.get(c);
		//V�rification du rep�re
		if (!in.get(c) || c != '<') {
			cout << "Erreur lors du chargement du reseau : " << filePath << " ne contient pas les donnees des biais." << endl;
			cout << "Couche : " << k << endl;
			*this = ReseauNeuronal();
			in.close();
			return false;
		}

		//Neurone par neurone
		for (int i = 0; i < listeBiais[k].n_elem; i++) {
			//Lecture des valeurs
			float b = -2;
			in >> b;

			//Si pas/mal lue, erreur
			if (b == -2) {
				cout << "Erreur lors du chargement du reseau : " << filePath << " ne contient pas les donnees des biais." << endl;
				cout << "Couche : " << k << "; Neurone : " << i << endl;
				*this = ReseauNeuronal();
				in.close();
				return false;
			}

			listeBiais[k](i) = b;
		}

		in.get(c);
		//V�rification du rep�re
		if (!in.get(c) || c != ';') {
			cout << "Erreur lors du chargement du reseau : " << filePath << " ne contient pas les donnees des biais." << endl;
			cout << "Couche : " << k << endl;
			*this = ReseauNeuronal();
			in.close();
			return false;
		}
	}

	///Fonctions d'activation

	in.get(c);
	//V�rification du rep�re
	if (!in.get(c) || c != '+') {
		cout << "Erreur lors du chargement du reseau : " << filePath << "  ne contient pas les donnees des fonctions d'activation." << endl;
		*this = ReseauNeuronal();
		in.close();
		return false;
	}
	listeActFunc = { new Sigmoid(), new Softmax() };
	for (int i = 0; i < structure.size() - 1; i++) {
		//Lecture des valeurs
		int f = -1;
		in >> f;

		//Si pas/mal lue, erreur
		if(f < 0  || f >= nbFonctionAvtivations) {
			cout << "Erreur lors du chargement du reseau : " << filePath << "  ne contient pas les donnees des fonctions d'activation." << endl;
			*this = ReseauNeuronal();
			in.close();
			return false;
		}

		activationIndex af = static_cast<activationIndex>(f);
		index_act_func.push_back(af);
	}

	///Fonction cout

	in.get(c);
	in.get(c);
	//V�rification du rep�re
	if (!in.get(c) || c != '$') {
		cout << "Erreur lors du chargement du reseau : " << filePath << "  ne contient pas les donnees des fonctions co�t." << endl;
		*this = ReseauNeuronal();
		in.close();
		return false;
	}

	//Lecture de la valeur
	int f = -1;
	in >> f;

	//Si pas/mal lue, erreur
	if (f < 0 || f >= nbFonctionsCout) {
		cout << "Erreur lors du chargement du reseau : " << filePath << "  ne contient pas les donnees des fonctions d'activation." << endl;
		*this = ReseauNeuronal();
		in.close();
		return false;
	}

	costFunction cf = static_cast<costFunction>(f);
	fonctionCout = cf;


	///Fin du fichier

	in.get(c);
	//V�rification du rep�re
	if (!in.get(c) || c != '*') {
		cout << "Erreur lors du chargement du reseau : " << filePath << " est trop gros pour le r�seau." << endl;
		*this = ReseauNeuronal();
		in.close();
		return false;
	}

	in.close();

	for (int i = 0; i < listePoids.size(); i++) {
		learningRates.push_back(learningRate);
	}

	return true;
}

//Croise les poids des connections de parent1 et parent2
ReseauNeuronal* ReseauNeuronal::reproduction(ReseauNeuronal *parent1, ReseauNeuronal *parent2) {
	srand(std::chrono::system_clock::now().time_since_epoch().count());

	//V�rifie si les deux parents sont compatibles
	if (parent1->structure != parent2->structure) {
		cout << "Erreur lors de la reproduction : Les deux parents n'ont pas la meme structure" << std::endl;
		return nullptr;
	}
	///A CHANGER
	/*else if (parent1->listeActFunc != parent2->listeActFunc) {
		cout << "Erreur lors de la reproduction : Les deux parents n'ont pas les memes fonctions d'activation" << std::endl;
		cout << "La reproduction va se faire mais il y a des risques de dysfonctionnement" << std::endl;
	}*/

	//Cr�� le nouvel enfant
	ReseauNeuronal *enfant = new ReseauNeuronal(parent1->structure);
	
	//Cr�� les matrices de poids et de biais
	for (int i = 0; i < enfant->listePoids.size(); i++) {
		//Cr�� la matrice de poids
		fmat p(enfant->listePoids[i].n_rows, enfant->listePoids[i].n_cols);
		
		//La remplit avec les poids des deux parents
		for (int j = 0; j < enfant->listePoids[i].n_elem; j++) {
			//Pour chaque poids, on choisi au hasard lequel duquel des deux parents il h�rite
			if (rand() % 2 == 0)
				p(j) = parent1->listePoids[i](j);
			else
				p(j) = parent2->listePoids[i](j);
		}

		//M�me chose avec les biais
		fvec b(enfant->listeBiais[i].n_rows);
		for (int j = 0; j < enfant->listeBiais[i].n_elem; j++) {
			if (rand() % 2 == 0)
				b(j) = parent1->listeBiais[i](j);
			else
				b(j) = parent2->listeBiais[i](j);
		}

		enfant->listePoids[i] = p;
		enfant->listeBiais[i] = b;
	}

	enfant->setActivationFuction(parent1->index_act_func);

	return enfant;
}

//Mutation du r�seau neuronal
void ReseauNeuronal::mutation() {
	srand(std::chrono::system_clock::now().time_since_epoch().count());

	//Chaque poids et chaque biais a une chance d'�tre modifi� de mani�re al�atoire.
	for (int i = 0; i < listePoids.size(); i++) {
		for (int j = 0; j < listePoids[i].n_elem; j++) {
			if (this->random(0, 1) < mutationRate)
				listePoids[i](j) += this->random(-0.2, 0.2);
		}
		for (int j = 0; j < listeBiais[i].n_elem; j++) {
			if (this->random(0, 1) < mutationRate)
				listeBiais[i](j) += this->random(-0.2, 0.2);
		}
	}
}