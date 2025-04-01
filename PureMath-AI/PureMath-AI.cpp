#include <iostream>
#include <vector>
#include <cmath>
#include <random>

#include "NeuralUtils.h"
#include "NeuralStruct.h"

int main()
{
    IAStruct net;
    NeuralUtils utils;

    net.expected_output = { 0.0, 1.0, 0.0, 0.0 };

    net.input.clear();
    for (int i = 0; i < 4; i++) {
        net.input.push_back(utils.randomDouble(0.0, 1.0));
    }

    for (auto& row : net.weights_input_hidden)
        for (double& w : row) w = utils.xavier_normal(4);

    for (auto& row : net.weights_hidden_output)
        for (double& w : row) w = utils.xavier_normal(16);
    
    for (double& b : net.bias_hidden) b = utils.randomBias();
    for (double& b : net.bias_output) b = utils.randomBias();

    bool treinar = true;

    while (treinar) {
        net.forward(net);

        std::cout << "Saida: ";
        for (double o : net.output) {
            std::cout << o << " ";
        }
        std::cout << std::endl;

        double loss = utils.meanSquaredError(net.output, net.expected_output);
        std::cout << "Erro: " << loss << std::endl;

        net.backpropagate(net, 0.01);

        if (loss <= 0.01) treinar = false;
    }


    return 0;
}
