#pragma once
#include <string>
#include <vector>

#include "nn_dataset.h"
#include "nn_logger.h"
#include "nn_math.h"
#include "nn_optimizer.h"
#include "nn_timer.h"

#include <iostream>
#include <fstream>
class NeuralNetFacade {
public:
    NeuralNetFacade();

    bool loadDataset(const std::string& path);
    void createModel(int inputSize, int hiddenSize, int outputSize);
    void configureTraining(double learningRate, int epochs);

    void train();
    double evaluate();
    std::vector<double> predict(const std::vector<double>& input);

    bool saveModel(const std::string& path);
    bool loadModel(const std::string& path);

private:
    // параметры
    int inputSize_;
    int hiddenSize_;
    int outputSize_;
    int epochs_;
    double learningRate_;

    bool datasetLoaded_;
    bool modelCreated_;

    // служебные методы
    void log(const std::string& message);
};