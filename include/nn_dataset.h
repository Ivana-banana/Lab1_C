#pragma once
#include "nn_math.h"
#include <vector>
#include <random>

namespace nn {

    struct Sample {
        Matrix input;  // ��������� ������ 1 x N
        Matrix target; // ��������� ������ 1 x M
    };

    class Dataset {
    public:
        std::vector<Sample> data;

        void add(const std::vector<float>& in, const std::vector<float>& out) {
            Sample s;
            s.input = Matrix(1, in.size());
            s.target = Matrix(1, out.size());
            for (size_t i = 0; i < in.size(); ++i) s.input.at(0, i) = in[i];
            for (size_t i = 0; i < out.size(); ++i) s.target.at(0, i) = out[i];
            data.push_back(s);
        }

        // ��������� ������ XOR
        static Dataset generateXOR() {
            Dataset ds;
            ds.add({ 0.0f, 0.0f }, { 0.0f });
            ds.add({ 0.0f, 1.0f }, { 1.0f });
            ds.add({ 1.0f, 0.0f }, { 1.0f });
            ds.add({ 1.0f, 1.0f }, { 0.0f });
            return ds;
        }

        // ��������� �������� ������
        static Dataset generateLinear(int count, float k = 2.0f, float b = 1.0f) {
            Dataset ds;
            std::mt19937 gen(123);
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);
            for (int i = 0; i < count; ++i) {
                float x = dist(gen);
                float y = k * x + b + (dist(gen) - 0.5f) * 0.1f;
                ds.add({ x }, { y });
            }
            return ds;
        }

        size_t size() const { return data.size(); }
    };
}