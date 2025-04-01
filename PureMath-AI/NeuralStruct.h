#ifndef NEURALSTRUCT_H
#define NEURALSTRUCT_H

#include <vector>
#include "NeuralUtils.h"

class IAStruct {
public:
    std::vector<double> input, hidden, output, expected_output;
    std::vector<std::vector<double>> weights_input_hidden, weights_hidden_output;
    std::vector<double> bias_hidden, bias_output;
    NeuralUtils utils;
    virtual void forward(IAStruct& net);
    virtual void backpropagate(IAStruct& net, double learning_rate);
    virtual ~IAStruct() {}
    void useNeuron(bool MLP, bool CNN);

protected:
    double sigmoid(double x);
    double sigmoid_derivative(double x);
    void activate(std::vector<double>& neurons);
};

#endif