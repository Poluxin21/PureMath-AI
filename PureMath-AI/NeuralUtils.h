#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <random>

class NeuralUtils
{
public:
    double randomDouble(double min, double max);
    double randomBias();
    double xavier_normal(int fan_in);
    double meanSquaredError(const std::vector<double>& output, const std::vector<double>& expected);
    std::vector<double> multiply(std::vector<double>& neurons, std::vector<std::vector<double>>& weights, std::vector<double>& bias);
};