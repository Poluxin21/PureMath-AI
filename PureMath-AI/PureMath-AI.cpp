#include <iostream>
#include <vector>
#include <cmath>
#include <random>

#include "Convolve.h"
#include "NeuralUtils.h"
#include "NeuralStruct.h"

int main() {
    NeuralUtils utils;
    IAStruct* net = nullptr;

    std::string tipoRede;
    std::cout << "Escolha a IA (MLP ou CNN): ";
    std::cin >> tipoRede;

    if (tipoRede == "CNN") {
        Convolve* cnn = new Convolve();
        net = cnn;

        int input_size = 4;
        int hidden_size = input_size - cnn->filter_size + 1;
        int output_size = 4; 

        cnn->initialize(input_size, hidden_size, output_size);
    }
    else {
        net = new IAStruct();

        int input_size = 4;
        int hidden_size = 4;
        int output_size = 4;

        net->weights_input_hidden.resize(input_size, std::vector<double>(hidden_size));
        net->weights_hidden_output.resize(hidden_size, std::vector<double>(output_size));
        net->bias_hidden.resize(hidden_size);
        net->bias_output.resize(output_size);

        for (auto& row : net->weights_input_hidden) {
            for (double& weight : row) {
                weight = utils.xavier_normal(input_size);
            }
        }

        for (auto& row : net->weights_hidden_output) {
            for (double& weight : row) {
                weight = utils.xavier_normal(hidden_size);
            }
        }

        for (double& bias : net->bias_hidden) {
            bias = utils.randomBias();
        }

        for (double& bias : net->bias_output) {
            bias = utils.randomBias();
        }
    }

    net->expected_output = { 0.0, 1.0, 0.0, 0.0 };

    net->input.clear();
    for (int i = 0; i < 4; i++) {
        net->input.push_back(utils.randomDouble(0.0, 1.0));
    }

    bool treinar = true;
    while (treinar) {
        net->forward(*net);

        std::cout << "Saída: ";
        for (double o : net->output) std::cout << o << " ";
        std::cout << std::endl;

        double loss = utils.meanSquaredError(net->output, net->expected_output);
        std::cout << "Erro: " << loss << std::endl;

        net->backpropagate(*net, 0.01);
        if (loss <= 0.01) treinar = false;
    }

    delete net;
    return 0;
}
