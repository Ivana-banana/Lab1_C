#pragma once
#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <functional>
#include <random> 

namespace nn {

    // ������� ��� ����������
    class Matrix {
    public:
        int rows, cols;
        std::vector<float> data;

        Matrix() : rows(0), cols(0) {}
        Matrix(int r, int c) : rows(r), cols(c), data(r* c, 0.0f) {}

        // ������ � ��������
        float& at(int r, int c) {
            if (r < 0 || r >= rows || c < 0 || c >= cols) throw std::out_of_range("Matrix index out of range");
            return data[r * cols + c];
        }
        float at(int r, int c) const {
            if (r < 0 || r >= rows || c < 0 || c >= cols) throw std::out_of_range("Matrix index out of range");
            return data[r * cols + c];
        }

        // ���������� ���������� ����������
        void randomize(float min = -1.0f, float max = 1.0f) {
            static std::mt19937 gen(42);
            std::uniform_real_distribution<float> dist(min, max);
            for (auto& val : data) val = dist(gen);
        }

        void zero() { std::fill(data.begin(), data.end(), 0.0f); }

        // ����������������
        Matrix transpose() const {
            Matrix res(cols, rows);
            for (int i = 0; i < rows; ++i)
                for (int j = 0; j < cols; ++j)
                    res.at(j, i) = at(i, j);
            return res;
        }

        // ��������� ������
        Matrix operator*(const Matrix& other) const {
            if (cols != other.rows) throw std::runtime_error("Matrix dim mismatch for multiplication");
            Matrix res(rows, other.cols);
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < other.cols; ++j) {
                    float sum = 0.0f;
                    for (int k = 0; k < cols; ++k) {
                        sum += at(i, k) * other.at(k, j);
                    }
                    res.at(i, j) = sum;
                }
            }
            return res;
        }

        // ������������ ��������
        Matrix operator+(const Matrix& other) const {
            if (rows != other.rows || cols != other.cols) throw std::runtime_error("Matrix dim mismatch for addition");
            Matrix res(rows, cols);
            for (size_t i = 0; i < data.size(); ++i) res.data[i] = data[i] + other.data[i];
            return res;
        }

        // ������������ ���������
        Matrix operator-(const Matrix& other) const {
            if (rows != other.rows || cols != other.cols) throw std::runtime_error("Matrix dim mismatch for subtraction");
            Matrix res(rows, cols);
            for (size_t i = 0; i < data.size(); ++i) res.data[i] = data[i] - other.data[i];
            return res;
        }

        // ������������ ��������� (Hadamard)
        Matrix hadamard(const Matrix& other) const {
            if (rows != other.rows || cols != other.cols) throw std::runtime_error("Matrix dim mismatch for hadamard");
            Matrix res(rows, cols);
            for (size_t i = 0; i < data.size(); ++i) res.data[i] = data[i] * other.data[i];
            return res;
        }

        // ��������� �� ������
        Matrix scale(float s) const {
            Matrix res(rows, cols);
            for (size_t i = 0; i < data.size(); ++i) res.data[i] = data[i] * s;
            return res;
        }

        // ���������� ������� � ������� ��������
        Matrix apply(std::function<float(float)> func) const {
            Matrix res(rows, cols);
            for (size_t i = 0; i < data.size(); ++i) res.data[i] = func(data[i]);
            return res;
        }

        // ����� �� ������� (���������� ������� 1 x cols)
        Matrix sum_rows() const {
            Matrix res(1, cols);
            for (int j = 0; j < cols; ++j) {
                float sum = 0.0f;
                for (int i = 0; i < rows; ++i) sum += at(i, j);
                res.at(0, j) = sum;
            }
            return res;
        }

        bool operator==(const Matrix& other) const {
            if (rows != other.rows || cols != other.cols) return false;
            for (size_t i = 0; i < data.size(); ++i) {
                if (std::abs(data[i] - other.data[i]) > 1e-5f) return false;
            }
            return true;
        }
};

    // �������������� ������� ���������
    namespace math {
        inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
        inline float sigmoid_deriv(float y) { return y * (1.0f - y); } // y - ��� ��� �������������� ��������

        inline float relu(float x) { return x > 0 ? x : 0.0f; }
        inline float relu_deriv(float x) { return x > 0 ? 1.0f : 0.0f; } // x - ��� �������� �� ���������

        inline float tanh_func(float x) { return std::tanh(x); }
        inline float tanh_deriv(float y) { return 1.0f - y * y; } // y - ��� ��� �������������� ��������
    }
}