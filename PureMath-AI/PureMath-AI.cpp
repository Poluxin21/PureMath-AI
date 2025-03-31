// PureMath-AI.cpp : Este arquivo contém a função 'main'. A execução do programa começa e termina ali.

#include <iostream>
#include <vector>
#include <cmath>
#include <random>

double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double x)
{
    return x * (1.0 - x);
}


struct IAStruct
{
    std::vector<double> input = std::vector<double>(4);
    std::vector<double> hidden = std::vector<double>(16);
    std::vector<double> output = std::vector<double>(4);

    std::vector<std::vector<double>> weights_input_hidden = std::vector<std::vector<double>>(4, std::vector<double>(16));
    std::vector<std::vector<double>> weights_hidden_output = std::vector<std::vector<double>>(16, std::vector<double>(4));

    std::vector<double> bias_hidden = std::vector<double>(16);
    std::vector<double> bias_output = std::vector<double>(4);

    std::vector<double> expected_output = std::vector<double>(4);
};

std::vector<double> multiply(std::vector<double>& neurons, std::vector<std::vector<double>>& weights, std::vector<double>& bias) {
    int neurons_next_layer = weights[0].size();
    std::vector<double> result(neurons_next_layer, 0.0);

    for (int j = 0; j < neurons_next_layer; j++) {
        for (int i = 0; i < neurons.size(); i++) {
            result[j] += neurons[i] * weights[i][j]; 
        }
        result[j] += bias[j];
    }
    return result;
}

double randomDouble(double min, double max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(min, max);
    return dis(gen);
}

double randomBias() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0.1, 0.5);
    return dist(gen);
}

void activate(std::vector<double>& neurons) {
    for (double& n : neurons) {
        n = sigmoid(n);
    }
}

void forward(IAStruct& net) {
    net.hidden = multiply(net.input, net.weights_input_hidden, net.bias_hidden);
    activate(net.hidden);

    net.output = multiply(net.hidden, net.weights_hidden_output, net.bias_output);
    activate(net.output);
}

void backpropagate(IAStruct& net, double learning_rate) {
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
            double gradient = hidden_error[j] * sigmoid_derivative(net.hidden[j]) * net.input[i];
            net.weights_input_hidden[i][j] -= learning_rate * gradient;
        }
    }

}


// ESTUDAR MAIS ESSA FUNÇÃO
double meanSquaredError(const std::vector<double>& output, const std::vector<double>& expected) {
    double sum = 0.0;
    for (size_t i = 0; i < output.size(); i++) {
        double error = output[i] - expected[i];
        sum += error * error;  // Eleva ao quadrado
    }
    return sum / output.size();
}

// ESTUDAR MAIS ESSA FUNÇÃO
double xavier_normal(int fan_in) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, sqrt(1.0 / fan_in));
    return dist(gen);
}

int main()
{
    IAStruct net;

    net.expected_output = { 0.0, 1.0, 0.0, 0.0 };

    net.input.clear();
    for (int i = 0; i < 4; i++) {
        net.input.push_back(randomDouble(0.0, 1.0));
    }

    for (auto& row : net.weights_input_hidden)
        for (double& w : row) w = xavier_normal(4);

    for (auto& row : net.weights_hidden_output)
        for (double& w : row) w = xavier_normal(16);
    
    for (double& b : net.bias_hidden) b = randomBias();
    for (double& b : net.bias_output) b = randomBias();

    bool treinar = true;

    while (treinar) {
        forward(net);

        std::cout << "Saida: ";
        for (double o : net.output) {
            std::cout << o << " ";
        }
        std::cout << std::endl;

        double loss = meanSquaredError(net.output, net.expected_output);
        std::cout << "Erro: " << loss << std::endl;

        backpropagate(net, 0.01);

        if (loss <= 0.01) treinar = false;
    }


    return 0;
}
