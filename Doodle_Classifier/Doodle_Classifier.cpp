// Doodle_Classifier.cpp�: d�finit le point d'entr�e pour l'application console.
//

#include "stdafx.h"
#include <SFML/Graphics.hpp>
#include "NeuralNetwork/ReseauNeuronal.h"

using namespace std;


const float trainingRatio = 0.8;
const float testingRatio = 1 - trainingRatio;

const int nbImages = 100000;
const int imageSize = 784;

const int cursorSize = 5;

#define NBIMAGES 5

struct category {
public:

	//Data of the category
	string label;
	char* trainingData;
	char* testingData;

	//Load data and initialise the label
	void loadData(string l) {
		label = l;
		loadData();
	}

	//Load data
	void loadData() {
		ifstream drawingsFile("data_" + to_string(nbImages) + "/" + label + to_string(nbImages) + ".dat", ios_base::binary);

		if (drawingsFile.fail()) {
			cout << "Error while loading " << label << " file : the file doesn't exist." << endl;
			cout << "Loading will not be done" << endl;
			return;
		}

		drawingsFile.seekg(0, drawingsFile.end);
		int length = drawingsFile.tellg();
		if (length != nbImages * imageSize) {
			drawingsFile.close();
			cout << "Error while loading " << label << " file : size doesn't match the number of drawings." << endl;
			cout << "Loading will be done but the data migth be incorrect/incomplete." << endl;
		}

		drawingsFile.seekg(0, drawingsFile.beg);

		char *drawingsData = new char[length];

		if (drawingsFile.read(drawingsData, length).fail()) {
			cout << "Error while loading " << label << " file : Error while loading." << endl;
		}

		drawingsFile.close();

		int trainingDataSize = trainingRatio * imageSize * nbImages;
		int testingDataSize = testingRatio * imageSize * nbImages;

		trainingData = new char[trainingDataSize];
		copy_n(drawingsData, trainingDataSize, trainingData);

		testingData = new char[testingDataSize];
		copy_n(drawingsData + trainingDataSize, testingDataSize, testingData);
	}

};

//Lists
vector<category> categories;
vector<string> dataLabels;

//Neural Network
ReseauNeuronal nn;

void guess(sf::RenderWindow *f) {

	//Get the drawing
	sf::Texture t;
	t.create(280, 280);
	t.update(*f);

	//Resize the window
	f->setSize(sf::Vector2u(28, 28));

	//Put the drawing back in the resized window
	sf::Sprite s(t);
	f->draw(s);


	sf::Texture t2;
	t2.create(28, 28);
	t2.update(*f);

	f->setSize(sf::Vector2u(280, 280));
	f->clear(sf::Color::White);

	f->display();

	sf::Image img = t2.copyToImage();

	if (img.getSize().x * img.getSize().y != imageSize) {
		cout << "Error : drawn image is not at the right size" << endl;
		return;
	}
	
	fvec input(imageSize);
	int side = std::sqrt(imageSize);

	for (int i = 0; i < imageSize; i++) {
		int x = i % side;
		int y = i / side;

		input(i) = (255 - float(img.getPixel(x, y).r)) / 255.f;
	}

	fvec guess = nn.feedforward(input);

	if (guess.n_elem != categories.size()) {
		cout << "Error : not enough output" << endl;
		return;
	}

	int indexHaut = 0;
	for (int i = 1; i < guess.n_elem; i++) {
		if (guess(indexHaut) < guess(i))
			indexHaut = i;
	}

	cout << "it's a : " << categories[indexHaut].label << endl;
}

fvec makeInput(int categoryI, int imageI) {
	fvec input(imageSize);
	copy_n(categories[categoryI].trainingData + imageSize * imageI, imageSize, input.begin());
	for_each(input.begin(), input.end(), [](float &f) { f = (f >= 0) ? f : (255 + f); f /= 255.f; });

	return input;
}

static void trainThread(ReseauNeuronal* rn, mutex* lock, vector<vector<fvec>> inputs, vector<vector<fvec>> outputs) {
	
	for (int i = 0; i < inputs.size(); i++) {
		lock->lock();
		rn->backPropagation(inputs[i], outputs[i]);
		lock->unlock();
	}

}

void trainEpoch(sf::RenderWindow *window) {

	int nbTrainingImages = nbImages * trainingRatio;

	//Initialise randomness
	minstd_rand g(std::chrono::system_clock::now().time_since_epoch().count());
	uniform_int_distribution<int> d(0, nbImages * trainingRatio);

	int miniBatch_size = 1;

	//Train the Neural Network
	for (int i = 0; i < 200 * categories.size() / miniBatch_size; i++) {

		vector<fvec> inputs;
		vector<fvec> outputs;

		for (int j = 0; j < miniBatch_size; j++) {
			int categoryIndex = rand() % categories.size();

			fvec target(categories.size());
			target.fill(0);
			target(categoryIndex) = 1;


			int imageIndex = d(g);

			fvec input(imageSize);
			copy_n(categories[categoryIndex].trainingData + imageSize * imageIndex, imageSize, input.begin());
			for_each(input.begin(), input.end(), [](float &f) { f = (f >= 0) ? f : (255 + f); f /= 255.f; });

			inputs.push_back(input);
			outputs.push_back(target);
		}

		/*int side = int(std::sqrt(imageSize));
		sf::Image img;
		img.create(side, side);
		for (int i = 0; i < imageSize; i++) {
		int x = i % side;
		int y = i / side;

		img.setPixel(x, y, sf::Color( 255 - input(i) * 255, 255 - input(i) * 255, 255 - input(i) * 255));
		}

		sf::Texture t;
		t.loadFromImage(img);

		window->clear(sf::Color::White);

		sf::Sprite s(t);
		s.scale(10, 10);
		window->draw(s);

		window->display();*/


		nn.backPropagation(inputs, outputs);
	}

	// for (int i = 0; i < categories.size(); i++) {
	// 	for (int j = 0; j < NBIMAGES; j++) {
	// 		fvec input(imageSize);
	// 		copy_n(categories[i].trainingData + imageSize * j, imageSize, input.begin());
	// 		for_each(input.begin(), input.end(), [](float &f) { f = (f >= 0) ? f : (255 + f); f /= 255.f; });


	// 		fvec target(categories.size());
	// 		target.fill(0);
	// 		target(i) = 1;

	// 		nn.backPropagation(input, target);

	// 	}
	// }
}

float testAllImages() {
	int nbTestingImages = nbImages * testingRatio;
	nbTestingImages = 200;
	float correct = 0;
	float cost = 0;

	for (int i = 0; i < categories.size(); i++) {
		for (int j = 0; j < nbTestingImages; j++) {
			fvec input(imageSize);
			copy_n(categories[i].testingData + imageSize * j, imageSize, input.begin());
			for_each(input.begin(), input.end(), [](float &f) { f = (f >= 0) ? f : (255 + f); f /= 255.f; });

			fvec guess = nn.feedforward(input);

			int highIndex = 0;
			for (int k = 1; k < guess.n_elem; k++) {
				if (guess(highIndex) < guess(k))
					highIndex = k;

				if (k == i)
					cost += pow(1 - guess(k), 2);
				else
					cost += pow(guess(k), 2);
			}

			if (i == highIndex)
				correct++;
		}
	}

	float pourcentage = correct / (categories.size() * nbTestingImages);
	cout << "Prec : " << pourcentage << "; Cout : " << cost << endl;

	return pourcentage;
}

float testValidationData() {
	
	float correct = 0;
	for (int i = 0; i < categories.size(); i++) {
		for (int j = 0; j < NBIMAGES; j++) {

			fvec input(imageSize);
			copy_n(categories[i].testingData + imageSize * j, imageSize, input.begin());
			for_each(input.begin(), input.end(), [](float &f) { f = (f >= 0) ? f : (255 + f); f /= 255.f; });

			fvec guess = nn.feedforward(input);

			int highIndex = 0;
			for (int k = 1; k < guess.n_elem; k++) {
				if (guess(highIndex) < guess(k))
					highIndex = k;
			}

			if (i == highIndex)
				correct++;

		}
	}

	//std::cout << "V:" << correct << std::endl;

	return (correct / (categories.size() * NBIMAGES));
}

float testTrainingData() {

	float correct = 0;

	for (int i = 0; i < categories.size(); i++) {
		for (int j = 0; j < NBIMAGES; j++) {
			fvec input(imageSize);
			copy_n(categories[i].trainingData + imageSize * j, imageSize, input.begin());
			for_each(input.begin(), input.end(), [](float &f) { f = (f >= 0) ? f : (255 + f); f /= 255.f; });

			fvec guess = nn.feedforward(input);

			int highIndex = 0;
			for (int k = 1; k < guess.n_elem; k++) {
				if (guess(highIndex) < guess(k))
					highIndex = k;
			}

			if (i == highIndex)
				correct++;
		}
	}

	//std::cout << "E:" << correct << std::endl;

	return (correct / (NBIMAGES * categories.size()));
}

float getValidationCost() {
	float cost = 0;

	for (int i = 0; i < categories.size(); i++) {
		for (int j = 0; j < NBIMAGES; j++) {

			fvec input(imageSize);
			copy_n(categories[i].testingData + imageSize * j, imageSize, input.begin());
			for_each(input.begin(), input.end(), [](float &f) { f = (f >= 0) ? f : (255 + f); f /= 255.f; });

			fvec guess = nn.feedforward(input);

			for (int k = 0; k < guess.n_elem; k++) {
				if (k == i)
					cost += pow(1 - guess(k), 2);
				else
					cost += pow(guess(k), 2);
			}
		}
	}

	return cost;
}

float getTrainingCost() {
	float cost = 0;

	for (int i = 0; i < categories.size(); i++) {
		for (int j = 0; j < NBIMAGES; j++) {
			fvec input(imageSize);
			copy_n(categories[i].trainingData + imageSize * j, imageSize, input.begin());
			for_each(input.begin(), input.end(), [](float &f) { f = (f >= 0) ? f : (255 + f); f /= 255.f; });

			fvec guess = nn.feedforward(input);

			for (int k = 0; k < guess.n_elem; k++) {
				if (k == i)
					cost += pow(1 - guess(k), 2);
				else
					cost += pow(guess(k), 2);
			}
		}
	}

	return cost;

}

int main()
{

	dataLabels = { "cat", "plane", "car", "bicycle", "mushroom", "cloud" };
	//dataLabels = { "mushroom", "cloud"};

	ofstream o("Graph/Data.txt");
	o.close();

	//Create white window
	sf::RenderWindow window(sf::VideoMode(280, 280), "Dessine !");
	window.clear(sf::Color::White);
	window.display();

	for (int i = 0; i < dataLabels.size(); i++) {
		category c;
		c.label = dataLabels[i];
//		c.loadData(dataLabels[i]);

		categories.push_back(c);
	}

	//nn = ReseauNeuronal({ imageSize, 500, 200, int(categories.size()) });
	nn = *(new ReseauNeuronal({ imageSize, 500, 200, int(categories.size()) }));
	nn.regularisation = false;
	nn.param_regularisation = 0.001;
	nn.setLearningRates(0.06);
	//nn.setLearningRates(0.04);

//	cout << nn.param_regularisation << std::endl;

	cout << "Voulez vous utiliser le reseau neuronal deja entraine ? (Y/N)" << endl;

	char rep = ' ';
	cin >> rep;

	bool loaded = false;
	if (rep == 'Y' || rep == 'y') {
		loaded = nn.load("NN_data");
		if (!loaded) {
			cout << "Echec de lecture des donnees du reseau : Le reseau va etre reentraine." << endl;
			nn = ReseauNeuronal({ imageSize, 200, 200, int(categories.size()) });
		}
	}

	//nn.listeActFunc = { new Sigmoid(), new Softmax() };

	if (!loaded) {
		float precision = -1;

		for (int i = 0; i < categories.size(); i++) {
			categories[i].loadData();
		}

		do {
			cout << "Entrez la precision du reseau neuronal (entre 0 et 1)" << endl;
			cin >> precision;
		} while (precision < 0 || precision >= 1);

		
		float prec = 0;
		do {
			trainEpoch(&window);

			nn.save("NN_data");

			prec = testValidationData();
			//prec = testAllImages();

			float errT = getTrainingCost();
			float errV = getValidationCost();

			float tPrec = testTrainingData();

			ofstream save("Graph/Data.txt", std::ios_base::app);
			save << errV << " " << errT << std::endl;
			//save << prec << " " << tPrec << std::endl;

			std::cout << "Prec : Entrainement : " << tPrec << "; Validation : " << prec << std::endl;
			std::cout << "Err : Entrainement : " << errT << "; Validation : " << errV << std::endl;
			std::cout << "-----------" << std::endl;
		} while (prec < precision);
	}
	

	cout << "hey" << endl;

	while (window.isOpen()) {

		sf::Event event;
		while (window.pollEvent(event))
		{
			//Fermeture de la fen�tre lorsque l'utilisateur le souhaite
			if (event.type == sf::Event::Closed)
				window.close();
		}

		if (sf::Mouse::isButtonPressed(sf::Mouse::Left)) {
			//To draw
			sf::Vector2i mousePos = sf::Mouse::getPosition();
			mousePos -= window.getPosition();
			mousePos -= sf::Vector2i(9, 39);

			if (mousePos.x >= 0 && mousePos.x <= 280 && mousePos.y >= 0 && mousePos.y <= 280) {
				sf::CircleShape cursor(cursorSize);
				cursor.setPosition(sf::Vector2f(mousePos) - sf::Vector2f(cursorSize / 2, cursorSize / 2));
				cursor.setFillColor(sf::Color(sf::Color::Black));
				
				window.draw(cursor);
				window.display();
			}
		}

		//To guess
		if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::End) || sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Escape)) {
			guess(&window);
			window.display();

			while (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::End) || sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Escape));
		}
	}

	for (int i = 0; i < categories.size(); i++) {
		delete categories[i].testingData;
		delete categories[i].trainingData;
	}
    return 0;
}

