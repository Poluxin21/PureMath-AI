#include "Convolve.h"
#include <random>

Convolve::Convolve() {
    initializeFilters();
}

void Convolve::initialize(int input_size, int hidden_size, int output_size) {
    weights_hidden_output.resize(hidden_size, std::vector<double>(output_size));
    bias_output.resize(output_size);

    for (auto& row : weights_hidden_output) {
        for (double& weight : row) {
            weight = utils.xavier_normal(hidden_size);
        }
    }

    for (double& bias : bias_output) {
        bias = utils.randomBias();
    }
}

void Convolve::initializeFilters() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(-1.0, 1.0);

    filters.resize(filter_size, std::vector<double>(filter_size));
    for (auto& row : filters)
        for (double& val : row)
            val = dist(gen);
}

std::vector<double> Convolve::applyConvolution(const std::vector<double>& input) {
    int output_size = input.size() - filter_size + 1;

    if (output_size <= 0) {
        throw std::runtime_error("Erro: entrada menor que o tamanho do filtro.");
    }

    std::vector<double> feature_map(output_size, 0.0);

    for (size_t i = 0; i < output_size; i++) {
        double sum = 0.0;
        for (size_t j = 0; j < filter_size; j++) {
            sum += input[i + j] * filters[0][j];
        }
        feature_map[i] = sum;
    }
    return feature_map;
}

void Convolve::forward(IAStruct& net) {
    net.hidden = applyConvolution(net.input); // Convolução na entrada
    activate(net.hidden); // Ativação após convolução
    net.output = utils.multiply(net.hidden, net.weights_hidden_output, net.bias_output);
    activate(net.output);
}

void Convolve::backpropagate(IAStruct& net, double learning_rate) {
    //
}
