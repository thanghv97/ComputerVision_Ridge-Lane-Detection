#include "log.hpp"

#include <cstdarg>
#include <cstdio>
#include <iostream>

namespace rld {

void vAPLogPrint(int level, char* fmt, ...) {
    va_list args;

    switch (level) {
        case LOG_LEVEL_DEBUG:
            std::cout << "DEBUG:\t";
            break;
        case LOG_LEVEL_INFO:
            std::cout << "INFO:\t";
            break;
        case LOG_LEVEL_ERROR:
            std::cout << "ERROR:\t";
            break;
        default:
            return;
    }

    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
}

}  // namespace rld