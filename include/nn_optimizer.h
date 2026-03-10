#pragma once
#include "nn_math.h"
#include "nn_logger.h"

namespace nn {

    // Интерфейс оптимизатора
    class Optimizer {
    public:
        float learningRate;
        Optimizer(float lr) : learningRate(lr) {}
        virtual ~Optimizer() {}

        // Обновление параметра: weight = weight - lr * gradient
        virtual void step(Matrix& weight, const Matrix& gradient) = 0;
    };

    // Стохастический градиентный спуск (SGD)
    class SGD : public Optimizer {
    public:
        SGD(float lr) : Optimizer(lr) {}

        void step(Matrix& weight, const Matrix& gradient) override {
            if (weight.rows != gradient.rows || weight.cols != gradient.cols) {
                Logger::error("Optimizer dimension mismatch");
                return;
            }
            // w = w - lr * grad
            Matrix update = gradient.scale(learningRate);
            weight = weight - update;
        }
    };
}