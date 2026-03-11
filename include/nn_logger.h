#pragma once
#include <iostream>
#include <string>

namespace nn {

    enum class LogLevel { NONE, ERROR, INFO, DEBUG };

    class Logger {
    private:
        static LogLevel currentLevel;
        Logger() {} // Singleton constructor
    public:
        static void setLevel(LogLevel level) { currentLevel = level; }

        static void log(LogLevel level, const std::string& msg) {
            if (level > currentLevel) return;

            std::string prefix = "";
            if (level == LogLevel::ERROR) prefix = "[ERROR] ";
            else if (level == LogLevel::INFO) prefix = "[INFO]  ";
            else if (level == LogLevel::DEBUG) prefix = "[DEBUG] ";

            std::cerr << prefix << msg << std::endl;
        }

        static void info(const std::string& msg) { log(LogLevel::INFO, msg); }
        static void error(const std::string& msg) { log(LogLevel::ERROR, msg); }
        static void debug(const std::string& msg) { log(LogLevel::DEBUG, msg); }
    };

    inline LogLevel Logger::currentLevel = LogLevel::INFO;
}