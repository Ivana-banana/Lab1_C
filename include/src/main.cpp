#include <iostream>
#include <cassert>
#include "../nn_facade.h"

using namespace nn;

// ���� ��������� ��������
void testMatrix() {
    Logger::info("Running Matrix Tests...");

    Matrix A(2, 3);
    A.at(0, 0) = 1.0f; A.at(0, 1) = 2.0f; A.at(0, 2) = 3.0f;
    A.at(1, 0) = 4.0f; A.at(1, 1) = 5.0f; A.at(1, 2) = 6.0f;

    Matrix B = A.transpose();
    assert(B.rows == 3 && B.cols == 2);
    assert(B.at(0, 0) == 1.0f && B.at(2, 1) == 6.0f);

    Matrix C(2, 3);
    C.zero();
    Matrix D = A + C;
    assert(D == A);

    Matrix E = A.scale(2.0f);
    assert(E.at(0, 0) == 2.0f);

    Logger::info("Matrix Tests Passed.");
};

// ���� ��������� ��������
void testDataset() {
    Logger::info("Running Dataset Tests...");

    Dataset ds = Dataset::generateXOR();
    assert(ds.size() == 4);
    assert(ds.data[0].input.cols == 2);
    assert(ds.data[0].target.cols == 1);

    Dataset lin = Dataset::generateLinear(10);
    assert(lin.size() == 10);

    Logger::info("Dataset Tests Passed.");
};

// ���� ������������
void testOptimizer() {
    Logger::info("Running Optimizer Tests...");

    Matrix w(1, 1);
    w.at(0, 0) = 10.0f;

    Matrix grad(1, 1);
    grad.at(0, 0) = 1.0f;

    SGD sgd(0.5f);
    sgd.step(w, grad);

    // 10 - 0.5 * 1 = 9.5
    assert(std::abs(w.at(0, 0) - 9.5f) < 1e-5f);

    Logger::info("Optimizer Tests Passed.");
};

// ������������ ������������� ������ ��� ������ ���� �������� (��� ������ ����)
void demoTrainingStep() {
    Logger::info("=== Demo: Manual Training Step ===");

    // 1. ���������� ������
    Dataset ds = Dataset::generateXOR();
    Sample sample = ds.data[0]; // 0, 0 -> 0

    // 2. ������������� ����� (�������, ��� ����� ��� ����)
    // ���� 2, ����� 1
    Matrix weights(2, 1);
    weights.at(0, 0) = 0.5f;
    weights.at(1, 0) = 0.5f;
    Matrix bias(1, 1);
    bias.at(0, 0) = 0.1f;

    SGD optimizer(0.1f);

    // 3. Forward Pass (�������)
    // Z = X * W + B
    Matrix z = sample.input * weights;
    z.at(0, 0) += bias.at(0, 0);

    // Activation
    Matrix output = z.apply(nn::math::sigmoid);

    Logger::debug("Output before training: " + std::to_string(output.at(0, 0)));

    // 4. ���������� ������ � ��������� (MSE derivative)
    // Loss = (out - target)^2
    // dLoss/dOut = 2 * (out - target)
    Matrix dLoss = output - sample.target; // ��������� (��� ��������� 2)

    // Backprop ����� ���������
    Matrix dActiv = dLoss.hadamard(output.apply([](float y) { return nn::math::sigmoid_deriv(y); }));

    // ��������� �����
    Matrix dWeights = sample.input.transpose() * dActiv;
    Matrix dBias = dActiv.sum_rows();

    // 5. ��� ������������
    optimizer.step(weights, dWeights);
    optimizer.step(bias, dBias);

    Logger::info("Weights updated successfully.");
    Logger::debug("New Weight 0: " + std::to_string(weights.at(0, 0)));
};

int main() {
    // ��������� ������ �����������
    Logger::setLevel(LogLevel::INFO);

    try {
        testMatrix();
        testDataset();
        testOptimizer();
        demoTrainingStep();

        std::cout << "\nAll tests completed successfully." << std::endl;
    }
    catch (const std::exception& e) {
        Logger::error(std::string("Exception caught: ") + e.what());
        return 1;
    }
    int i;
    std::cin >> i; 
    return 0;
};