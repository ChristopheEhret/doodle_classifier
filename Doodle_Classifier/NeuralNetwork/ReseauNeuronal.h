#pragma once
//#include "stdafx.h"
#include <armadillo>
#include "Activation.h"

//Fonction d'activation des neurones
static float sigmoide(float &x) {
	return 1 / (1 + exp(-x));
}

using namespace arma;
using namespace std;
class ReseauNeuronal {
public:
	//Listes de fonctions d'activations disponibles
	enum activationIndex {
		SIGMOID,
		SOFTMAX
	};
	char nbFonctionAvtivations = 2;

	//Liste des fonctions co�t disponibles
	enum costFunction {
		QUADRATIC,
		CROSSENTROPY
	};
	char nbFonctionsCout = 2;

	//Constructeurs
	ReseauNeuronal();
	ReseauNeuronal(std::vector<int> struc);
	ReseauNeuronal(std::vector<fmat> poids, std::vector<fvec> biais);
	ReseauNeuronal(std::vector<std::vector<std::vector<float>>> poids, std::vector<std::vector<float>> biais);
	ReseauNeuronal(const ReseauNeuronal &rn);

	virtual ~ReseauNeuronal();

	//Resultat du r�seau neuronal avec 'input' pour entr�e
	virtual fvec feedforward(fvec entree);

	//Entraine le r�seau avec 'entree' qui a pour r�sultat correct 'cible' 
	void backPropagation(fvec entree, fvec cible);
	//Entraine le r�seau en batch
	void backPropagation(vector<fvec> entrees, vector<fvec> cibles);
	
	//Permet de changer le 'learning rate'
	void setLearningRate(float lr) { learningRate = lr; }

	//Permet de changer la fonction d'activation de toutes les couches
	void setActivationFuction(activationIndex i) { for_each(index_act_func.begin(), index_act_func.end(), [&i](activationIndex &func) {func = i; }); }
	//Permet de changer la fonction d'activation de la couche c
	void setActivationFuction(activationIndex f, int c);
	//Permet de changer la fonction d'activation de chaque couche
	void setActivationFuction(std::vector<activationIndex> listef);
	//Permet de changer la fonction co�t
	void setCostFunction(costFunction f) { fonctionCout = f; }

	//Permet de changer le learning rate de toutes les couches
	void setLearningRates(float nlr) { for_each(learningRates.begin(), learningRates.end(), [&nlr](float &lr) {lr = nlr; }); }
	//Permet de changer le learning rate de chaque couche
	void setLearningRates(vector<float> nlr);

	//Permet de changer le g�n�rateur de nombres al�atoires
	//Si vous manipulez plusieurs r�seaux neuronaux, il est vivement conseill� de donner le m�me g�n�rateur de nombres
	//Dans le cas contraire, il y a de fortes chances que les nombres g�n�r�s soient les m�me d'un r�seau neuronal sur l'autre
	void setRandomGenerator(std::minstd_rand *n_generator) { generator = n_generator; }

	//Affiche le r�seau neuronal
	void print();

	//Sauvegarde le reseau neuronal dans un fichier
	void save(string filePath);
	//Sauvegarde le r�seau neuronal dans un format adapt� au programme arduino
	void save_arduino(string filePath);
	//Charge un reseau neuronal depuis un fichier
	bool load(string filePath);

	//Croise deux r�seaux neuronaux parents
	static ReseauNeuronal* reproduction(ReseauNeuronal *parent1, ReseauNeuronal *parent2);

	//Mutation
	void mutation();
	float fitness = 0;

	void set_regularisaion(bool r) { regularisation = r; }
	void set_param_regularisation(float pR) { param_regularisation = pR; }

protected:
	//Structure du r�seau neuronal, incluant le nombre d'entrees
	std::vector<int> structure;

	//Gen�rateur de nombres al�atoires
	std::minstd_rand *generator;

	float random(float val_min = -1, float val_max = 1);

	//Liste de matrices contenant les poids des liens entre chaque couche de neurone
	std::vector<fmat> listePoids;
	//Liste de matrices contenant les biais de chaque neurones d'une couche
	std::vector<fvec> listeBiais;

	//Liste des fonctions d'activation disponibles
	std::vector<Activation*> listeActFunc;
	//Lists des index des fonctions d'activation de chaque couche
	std::vector<activationIndex> index_act_func;
	//Fonction co�t
	costFunction fonctionCout = QUADRATIC;

	//Importance de chaque session d'apprentissage en backpropagation
	float learningRate = 0.01;
	vector<float> learningRates;

	//Chance qu'a chaque neurone et chaque poids d'�tre mut�
	float mutationRate = 0.1;
public:
	bool regularisation = false;
	float param_regularisation = 0.1;
};