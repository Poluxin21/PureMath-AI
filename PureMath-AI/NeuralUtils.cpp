#include "NeuralUtils.h"

double NeuralUtils::randomDouble(double min, double max) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(min, max);
    return dis(gen);
}

double NeuralUtils::randomBias() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0.1, 0.5);
    return dist(gen);
}

double NeuralUtils::xavier_normal(int fan_in) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, sqrt(1.0 / fan_in));
    return dist(gen);
}

double NeuralUtils::meanSquaredError(const std::vector<double>& output, const std::vector<double>& expected) {
    double sum = 0.0;
    for (size_t i = 0; i < output.size(); i++) {
        double error = output[i] - expected[i];
        sum += error * error;  // Eleva ao quadrado
    }
    return sum / output.size();
}

std::vector<double> NeuralUtils::multiply(std::vector<double>& neurons, std::vector<std::vector<double>>& weights, std::vector<double>& bias) {
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