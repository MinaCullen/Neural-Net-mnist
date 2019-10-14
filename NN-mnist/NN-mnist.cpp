// ConsoleApplication1.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
// ConsoleApplication1.cpp : Defines the entry point for the console application.
//
#include <vector>
#include <random>
#include <iostream>
#include <string>
#include <tuple>
#include <algorithm>
#include <fstream>
#include <ctime>
#include <dense>
#include "mnist/mnist_reader_less.hpp"

using namespace std;
using namespace Eigen;
using namespace mnist;

struct network {
private: //
	int num_layers;
	vector<int> sizes;
	vector<MatrixXf> biases;
	vector<MatrixXf> weights;
	float sigmoid(float input) { //the sigmoid function
		return 1 / (1 + exp(-1 * input));
	}
	VectorXf sigmoid_elem_wise(VectorXf input) {//method for applying the sigmoid function to every element in a vector
		for (int i = 0; i < input.size(); i++) {
			input[i] = sigmoid(input[i]);
		}
		return input;
	}
	VectorXf sigmoid_prime(VectorXf input) { //the derivative of the sigmoid function.
		VectorXf one_vec(input.size()); // i only needed an element-wise implementation
		for (int i = 0; i < input.size(); i++) {
			one_vec[i] = 1;
		}
		VectorXf x = sigmoid_elem_wise(input);
		VectorXf y = (one_vec - sigmoid_elem_wise(input));
		VectorXf output(x.size());
		for (int i = 0; i < x.size(); i++) {
			output[i] = x[i] * y[i];
		}
		return output;
	}
	VectorXf hadamard_product(VectorXf x, VectorXf y) { //(element-wise product)
		VectorXf out(x.size());
		for (int i = 0; i < x.size(); i++) {
			out[i] = x[i] * y[i];
		}
		return out;
	}
	VectorXf quadratic_cost_derivative(VectorXf output_activations, VectorXf y) { return output_activations - y; } //the basic cost function used
	VectorXf feed_forward(VectorXf a) { //image data is input and fed forward through the network
		for (int i = 0; i < num_layers - 1; i++) { //a guess is output as a vector (last layer), which is returned
			VectorXf v = (weights[i] * a) + biases[i]; //general equation for activation in the network
			v = sigmoid_elem_wise(v);
			a = v;
		}
		return a;
	}
	void stochastic_gradient_descent(vector<tuple<VectorXf, VectorXf>> training_data, int epochs, int mini_batch_size, float learning_rate, vector<tuple<VectorXf, VectorXf>> test_data = {}) {
		int n_test = 0;
		n_test = test_data.size();
		int n = training_data.size();
		for (int i = 0; i < epochs; i++) {
			random_shuffle(training_data.begin(), training_data.end());
			vector<VectorXf> training_labels;
			vector<vector<tuple<VectorXf, VectorXf>>> mini_batches;
			for (int i = 0; i < n; i += mini_batch_size) { //create minibatches
				vector<tuple<VectorXf, VectorXf>> mb;
				for (int i1 = i; i1 < i + mini_batch_size; i1++) {
					mb.push_back(training_data[i1]);
				}
				mini_batches.push_back(mb);
			}
			int count = 0;
			for (vector<tuple<VectorXf, VectorXf>> mini_batch : mini_batches)
			{
				update_mini_batch(mini_batch, learning_rate);//pass in mini-batch, use this to find approxiation for nabla_b and nabla_w (this is the SGD part)
				count++;
			}
			if (n_test != 0) {
				cout << "Epoch " << i << ": " << evaluate(test_data) << "/" << n_test << endl; //provide a 'progress report' (slower)
			}
			else {
				cout << "Epoch " << i << " complete" << endl; //no 'progress report' (faster)
			}
		}
	}
	void update_mini_batch(vector<tuple<VectorXf, VectorXf>> mini_batch, float learning_rate) {
		vector<MatrixXf> nabla_b;
		vector<MatrixXf> nabla_w;
		for (MatrixXf b : biases) //making zero matricies
		{
			MatrixXf m = b;
			m.setZero(b.rows(), b.cols());
			nabla_b.push_back(m);
		}
		for (MatrixXf w : weights)
		{
			MatrixXf m = w;
			m.setZero(w.rows(), w.cols());
			nabla_w.push_back(m);
		}
		for (tuple<VectorXf, VectorXf> x : mini_batch)
		{
			tuple<vector<MatrixXf>, vector<MatrixXf>> t = backprop(x);//get gradient of cost function with respect to weights and biases for this image
			vector<MatrixXf> delta_nabla_b = get<0>(t);//gradient of cost func. with respect to biases for this training image
			vector<MatrixXf> delta_nabla_w = get<1>(t);//gradient of cost func. with respect to weights for this training image
			vector<MatrixXf> temp_b;
			for (int i = 0; i < nabla_b.size(); i++) {
				temp_b.push_back(nabla_b[i] + delta_nabla_b[i]);
			}
			nabla_b = temp_b;
			vector<MatrixXf> temp_w;
			for (int i = 0; i < nabla_w.size(); i++) {
				temp_w.push_back(nabla_w[i] + delta_nabla_w[i]);
			}
			nabla_w = temp_w; //sum delta_nablas (for w and b) for each image in minibatch
		}
		vector<MatrixXf> temp_w;
		for (int i = 0; i < weights.size(); i++) {
			temp_w.push_back(weights[i] - ((learning_rate / mini_batch.size())*nabla_w[i])); //taking average for minibatch to get approximation for gradient of cost with respect to w and b
		}
		weights = temp_w; //update weights using this approximation
		vector<MatrixXf> temp_b;
		for (int i = 0; i < biases.size(); i++) {
			temp_b.push_back(biases[i] - ((learning_rate / mini_batch.size())*nabla_b[i]));
		}
		biases = temp_b;
	}
	tuple<vector<MatrixXf>, vector<MatrixXf>> backprop(tuple<VectorXf, VectorXf> t) {
		VectorXf x = get<0>(t);
		VectorXf y = get<1>(t);
		vector<MatrixXf> nabla_b;
		vector<MatrixXf> nabla_w;
		for (MatrixXf b : biases) //making zero matricies
		{
			MatrixXf m = b;
			m.setZero(b.rows(), b.cols());
			nabla_b.push_back(m);
		}
		for (MatrixXf w : weights)
		{
			MatrixXf m = w;
			m.setZero(w.rows(), w.cols());
			nabla_w.push_back(m);
		}
		VectorXf activation = x;
		vector<VectorXf> activations;
		activations.push_back(activation);
		vector<VectorXf> z_vecs;
		for (int i = 0; i < num_layers - 1; i++) { //feed forwards to get activations of all layers (and zs)
			auto z = weights[i] * activation + biases[i];
			z_vecs.push_back(z);
			activation = sigmoid_elem_wise(z);
			activations.push_back(activation);
		}
		VectorXf temp1 = quadratic_cost_derivative(activations[activations.size() - 1], y);
		VectorXf temp2 = sigmoid_prime(z_vecs[z_vecs.size() - 1]);
		VectorXf delta = hadamard_product(temp1, temp2); //delta of last layer
		nabla_b[nabla_b.size() - 1] = delta; //change in cost with respect to biases in last layer
		nabla_w[nabla_w.size() - 1] = delta * activations[activations.size() - 2].transpose(); //change in cost with respect to biases in last layer
		for (int l = 2; l < num_layers; l++) { //backwards pass - find deltas for each layer
			auto z = z_vecs[z_vecs.size() - l];
			VectorXf sp = sigmoid_prime(z);
			VectorXf temp1 = (weights[weights.size() - l + 1].transpose() * delta);
			delta = hadamard_product(temp1, sp);
			nabla_b[nabla_b.size() - l] = delta; //change in cost with respect to biases in layer
			nabla_w[nabla_w.size() - l] = delta * activations[activations.size() - l - 1].transpose(); //change in cost with respect to weights in layer
		}
		tuple<vector<MatrixXf>, vector<MatrixXf>> tup = make_tuple(nabla_b, nabla_w);
		return tup;
	}
	int evaluate(vector<tuple<VectorXf, VectorXf>> test_data) { //checking how the network performs
		int correct = 0;
		int count = 0;
		for (tuple<VectorXf, VectorXf> data : test_data) {
			VectorXf x = get<0>(data);
			VectorXf image = x;
			VectorXf y = get<1>(data);
			x = feed_forward(x);
			Index max_row_x, max_col_x;
			float max_x = x.maxCoeff(&max_row_x, &max_col_x);
			Index max_row_y, max_col_y;
			float max_y = y.maxCoeff(&max_row_y, &max_col_y);
			if (max_row_x == max_row_y) { correct++; }
		}
		return correct;
	}
	void show_mnist(VectorXf image, int label = -1, int guess = -1) {
		int count = 0;
		for (int i = 0; i < 784; i++) {
			if (image[i] > 0.5) { cout << "@"; }
			else { cout << " "; }
			count++;
			if (count == 28) {
				cout << endl;
				count = 0;
			}
		}
		if (label != -1) { cout << endl << label << endl << guess << endl; }
	}
public:
	network(vector<int> Sizes, bool init_w_and_b = true) {
		default_random_engine generator;
		normal_distribution<float> distribution_bias(0, 1);
		num_layers = Sizes.size();
		sizes = Sizes;
		if (init_w_and_b) {
			for (int y = 1; y < num_layers; y++) {
				MatrixXf m(Sizes[y], 1);
				for (int y1 = 0; y1 < m.rows(); y1++) {
					m(y1, 0) = distribution_bias(generator);
				}
				biases.push_back(m);
			}
			for (int x = 0; x < num_layers - 1; x++) {
				MatrixXf m(Sizes[x + 1], Sizes[x]);
				for (int x1 = 0; x1 < m.rows(); x1++) {
					for (int y1 = 0; y1 < m.cols(); y1++) {
						normal_distribution<float> distribution_weight(0, 1 / sqrt(sizes[x])); //init weights with new method: 1/sqrt(n_in)
						//m(x1, y1) = distribution_bias(generator);  //random weight init
					}
				}
				weights.push_back(m);
			}
		}
	}
	void show_weights_and_biases() {
		cout << "weights" << endl;
		int count = 1;
		for (MatrixXf m : weights)
		{
			cout << endl << count << m << endl << endl;
			count++;
		}
		cout << "biases" << endl;
		count = 1;
		for (MatrixXf m : biases)
		{
			cout << endl << count << m << endl << endl;
			count++;
		}
	}
	void run_network(vector<VectorXf> training_labels, vector<VectorXf> training_images, vector<VectorXf> test_labels = {}, vector<VectorXf> test_images = {}) {
		vector<tuple<VectorXf, VectorXf>> training_data;
		for (int i = 0; i < training_labels.size(); i++) {
			training_data.push_back(make_tuple(training_images[i], training_labels[i]));
		}
		vector<tuple<VectorXf, VectorXf>> test_data;
		for (int i = 0; i < test_labels.size(); i++) {
			test_data.push_back(make_tuple(test_images[i], test_labels[i]));
		}
		stochastic_gradient_descent(training_data, 30, 10, 3, test_data);
	}
	vector<MatrixXf> get_weights() { return weights; }
	vector<MatrixXf> get_biases() { return biases; }
	void demonstrate(vector<VectorXf> images, int amount) {//output (bad) ASCII representations of each image along with the network's guess
		random_shuffle(images.begin(), images.end()); //for demo purposes - not used for learning
		for (int i = 0; i < amount; i++) {
			VectorXf guess = feed_forward(images[i]);
			Index max_row, max_col;
			float max_x = guess.maxCoeff(&max_row, &max_col);
			show_mnist(images[i]);
			cout << max_row << endl << endl;
		}
	}
};
tuple<vector<VectorXf>, vector<VectorXf>, vector<VectorXf>, vector<VectorXf>, vector<VectorXf>> load_data() {
	auto dataset = read_dataset<float, float>();
	vector<VectorXf> training_labels;
	vector<VectorXf> training_labels_temp;
	for (auto label : dataset.training_labels) {
		vector<float> label1 = { 0,0,0,0,0,0,0,0,0,0 };
		for (int i = 0; i < 10; i++) {
			if (label == i) {
				label1[i] = 1;
			}
		}
		float* ptr = &label1[0];
		training_labels_temp.push_back(Map<VectorXf>(ptr, label1.size()));
	}
	training_labels = training_labels_temp;
	vector<VectorXf> training_images;
	vector<VectorXf> training_images_temp;
	for (vector<float> image : dataset.training_images) {
		float* ptr = &image[0];
		training_images_temp.push_back(Map<VectorXf>(ptr, image.size()));
	}
	training_images = training_images_temp;
	vector<VectorXf> test_labels;
	vector<VectorXf> test_labels_temp;
	for (auto label : dataset.test_labels) {
		vector<float> label1 = { 0,0,0,0,0,0,0,0,0,0 };
		for (int i = 0; i < 10; i++) {
			if (label == i) {
				label1[i] = 1;
			}
		}
		float* ptr = &label1[0];
		test_labels_temp.push_back(Map<VectorXf>(ptr, label1.size()));
	}
	test_labels = test_labels_temp;
	vector<VectorXf> test_images;
	vector<VectorXf> test_images_temp;
	for (vector<float> image : dataset.test_images) {
		float* ptr = &image[0];
		test_images_temp.push_back(Map<VectorXf>(ptr, image.size()));
	}
	test_images = test_images_temp;
	test_images.erase(test_images.begin() + 100, test_images.end());
	vector<VectorXf> test_lab = training_labels;
	vector<VectorXf> test_img = training_images;
	test_lab.erase(test_lab.begin(), test_lab.end() - 10000);
	test_img.erase(test_img.begin(), test_img.end() - 10000);
	training_labels.erase(training_labels.begin() + 50000, training_labels.end());
	training_images.erase(training_images.begin() + 50000, training_images.end());
	return make_tuple(training_labels, training_images, test_lab, test_img, test_images);
}
void train_net() {
	vector<int> vec{ 784,100,10 };
	network* net = new network(vec);
	auto data = load_data();
	(*net).run_network(get<0>(data), get<1>(data), get<2>(data), get<3>(data));
	(*net).demonstrate(get<4>(data), 50);
}


int main()
{
	srand(unsigned(time(0)));
	train_net();
	cout << "done";
	cin.ignore(2);
	return 0;
}
