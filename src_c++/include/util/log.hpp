#ifndef RLD_UTIL_LOG_HPP
#define RLD_UTIL_LOG_HPP

namespace rld {

#define LOG_ERROR(...) logPrint(LOG_LEVEL_ERROR, __VA_ARGS__)
#define LOG_INFO(...) logPrint(LOG_LEVEL_INFO, __VA_ARGS__)

typedef enum {
    LOG_LEVEL_TRACE = 0,
    LOG_LEVEL_DEBUG,
    LOG_LEVEL_INFO,
    LOG_LEVEL_WARN,
    LOG_LEVEL_ERROR,
    LOG_LEVEL_FATAL
} logLevel;

void logPrint(int level, char* fmt, ...);

}  // namespace rld

#endif