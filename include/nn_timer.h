#pragma once
#include <chrono>
#include <string>
#include <iostream>
#include <vector>
#include "nn_logger.h"

namespace nn {

    // ������� ������ ��� ������ ������� ����������
    class Timer {
    private:
        std::chrono::high_resolution_clock::time_point start;
        std::chrono::high_resolution_clock::time_point end;
        bool running;
        std::string label;

    public:
        Timer(const std::string& lbl = "Timer") : label(lbl), running(false) {}

        void start_time() {
            start = std::chrono::high_resolution_clock::now();
            running = true;
        }

        void stop() {
            if (running) {
                end = std::chrono::high_resolution_clock::now();
                running = false;
            }
        }

        // ���������� ����� � �������������
        double elapsed_ms() const {
            if (running) {
                auto now = std::chrono::high_resolution_clock::now();
                return std::chrono::duration<double, std::milli>(now - start).count();
            }
            return std::chrono::duration<double, std::milli>(end - start).count();
        }

        // ���������� ����� � ��������
        double elapsed_sec() const {
            return elapsed_ms() / 1000.0;
        }

        // ����� ���������� � ���
        void print() const {
            Logger::info(label + ": " + std::to_string(elapsed_ms()) + " ms");
        }

        // RAII �����: �������������� �����/����
        class ScopedTimer {
        private:
            Timer& timer;
        public:
            ScopedTimer(Timer& t) : timer(t) { timer.start_time(); }
            ~ScopedTimer() { timer.stop(); }
        };
    };

    // ������������� ��� ����� ���������� �� ��������� �������
    class Profiler {
    private:
        struct Record {
            std::string name;
            double total_ms;
            int count;
            double min_ms;
            double max_ms;
        };
        std::vector<Record> records;

    public:
        void record(const std::string& name, double elapsed_ms) {
            for (auto& rec : records) {
                if (rec.name == name) {
                    rec.total_ms += elapsed_ms;
                    rec.count++;
                    rec.min_ms = std::min(rec.min_ms, elapsed_ms);
                    rec.max_ms = std::max(rec.max_ms, elapsed_ms);
                    return;
                }
            }
            records.push_back({ name, elapsed_ms, 1, elapsed_ms, elapsed_ms });
        }

        void print_summary() const {
            Logger::info("=== Profiler Summary ===");
            for (const auto& rec : records) {
                double avg = rec.total_ms / rec.count;
                std::cout << rec.name << ": "
                    << "avg=" << avg << "ms, "
                    << "min=" << rec.min_ms << "ms, "
                    << "max=" << rec.max_ms << "ms, "
                    << "count=" << rec.count << std::endl;
            }
            Logger::info("========================");
        }

        void clear() { records.clear(); }
    };

    // ���������� ������������� (singleton)
    inline Profiler& global_profiler() {
        static Profiler instance;
        return instance;
    }

    // ������� ��� �������� �������������
#define NN_TIMER_START(name) \
        nn::Timer timer_##name(#name); \
        timer_##name.start();

#define NN_TIMER_END(name) \
        timer_##name.stop(); \
        nn::global_profiler().record(#name, timer_##name.elapsed_ms());

#define NN_TIMER_SCOPE(name) \
        nn::Timer timer_##name(#name); \
        nn::Timer::ScopedTimer scope(timer_##name); \
        auto profiler_guard_##name = [&]() { \
            nn::global_profiler().record(#name, timer_##name.elapsed_ms()); \
        };
}