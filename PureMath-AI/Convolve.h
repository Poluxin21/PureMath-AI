#ifndef CNNSTRUCT_H
#define CNNSTRUCT_H

#include "NeuralStruct.h"
#include <vector>

class Convolve : public IAStruct {
public:
    Convolve();
    void forward(IAStruct& net) override;
    void backpropagate(IAStruct& net, double learning_rate) override;
    void initialize(int input_size, int hidden_size, int output_size);
    int filter_size = 3;

private:
    std::vector<std::vector<double>> filters;

    void initializeFilters();
    std::vector<double> applyConvolution(const std::vector<double>& input);

    NeuralUtils utils;
};

#endif
