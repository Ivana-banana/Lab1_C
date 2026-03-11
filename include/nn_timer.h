#pragma once
#include <chrono>
#include <string>
#include <iostream>
#include "nn_logger.h"

namespace nn {

    // Простой таймер для замера времени выполнения
    class Timer {
    private:
        std::chrono::high_resolution_clock::time_point start_time;
        std::chrono::high_resolution_clock::time_point end_time;
        bool is_running;
        std::string label;

    public:
        // Конструктор с именем таймера
        Timer(const std::string& lbl = "Таймер")
            : label(lbl), is_running(false) {
        }

        // Запустить замер
        void start() {
            start_time = std::chrono::high_resolution_clock::now();
            is_running = true;
        }

        // Остановить замер
        void stop() {
            if (is_running) {
                end_time = std::chrono::high_resolution_clock::now();
                is_running = false;
            }
        }

        // Получить прошедшее время в миллисекундах
        double elapsed_ms() const {
            if (is_running) {
                auto now = std::chrono::high_resolution_clock::now();
                return std::chrono::duration<double, std::milli>(now - start_time).count();
            }
            return std::chrono::duration<double, std::milli>(end_time - start_time).count();
        }

        // Получить прошедшее время в секундах
        double elapsed_sec() const {
            return elapsed_ms() / 1000.0;
        }

        // Вывести результат в лог
        void print() const {
            Logger::info(label + ": " + std::to_string(elapsed_ms()) + " мс");
        }
    };

    // Простой макрос для старта таймера
#define NN_TIMER_START(name) \
        nn::Timer timer_##name(#name); \
        timer_##name.start();

    // Простой макрос для остановки и вывода результата
#define NN_TIMER_STOP(name) \
        timer_##name.stop(); \
        timer_##name.print();
}