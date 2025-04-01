#include <iostream>
#include <vector>
#include <cmath>
#include "NeuralUtils.h"
class IAStruct
{
public:
    std::vector<double> input = std::vector<double>(4);
    std::vector<double> hidden = std::vector<double>(16);
    std::vector<double> output = std::vector<double>(4);

    std::vector<std::vector<double>> weights_input_hidden = std::vector<std::vector<double>>(4, std::vector<double>(16));
    std::vector<std::vector<double>> weights_hidden_output = std::vector<std::vector<double>>(16, std::vector<double>(4));

    std::vector<double> bias_hidden = std::vector<double>(16);
    std::vector<double> bias_output = std::vector<double>(4);

    std::vector<double> expected_output = std::vector<double>(4);

    double sigmoid(double x);
    double sigmoid_derivative(double x);
    void activate(std::vector<double>& neurons);
    void forward(IAStruct& net);
    void backpropagate(IAStruct& net, double learning_rate);


private:
    NeuralUtils utils;
};
