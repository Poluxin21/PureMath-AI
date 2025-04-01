#include "NeuralStruct.h"


double IAStruct::sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

double IAStruct::sigmoid_derivative(double x)
{
    return x * (1.0 - x);
}
void IAStruct::activate(std::vector<double>& neurons) {
    for (double& n : neurons) {
        n = sigmoid(n);
    }
}

void IAStruct::forward(IAStruct& net) {
    net.hidden = utils.multiply(net.input, net.weights_input_hidden, net.bias_hidden);
    net.activate(net.hidden);

    net.output = utils.multiply(net.hidden, net.weights_hidden_output, net.bias_output);
    net.activate(net.output);
}

void IAStruct::backpropagate(IAStruct& net, double learning_rate) {
    std::vector<double> output_error(net.output.size());

    for (size_t i = 0; i < net.output.size(); i++) {
        output_error[i] = net.output[i] - net.expected_output[i];
    }

    for (size_t i = 0; i < net.weights_hidden_output.size(); i++) {
        for (size_t j = 0; j < net.weights_hidden_output[i].size(); j++) {
            double gradient = output_error[j] * net.hidden[i];
            net.weights_hidden_output[i][j] -= learning_rate * gradient;
        }
    }

    std::vector<double> hidden_error(net.hidden.size(), 0.0);
    for (size_t j = 0; j < net.hidden.size(); j++) {
        for (size_t k = 0; k < net.output.size(); k++) {
            hidden_error[j] += output_error[k] * net.weights_hidden_output[j][k];
        }
    }

    for (size_t i = 0; i < net.weights_input_hidden.size(); i++) {
        for (size_t j = 0; j < net.weights_input_hidden[i].size(); j++) {
            double gradient = hidden_error[j] * net.sigmoid_derivative(net.hidden[j]) * net.input[i];
            net.weights_input_hidden[i][j] -= learning_rate * gradient;
        }
    }

}