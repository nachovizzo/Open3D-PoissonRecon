/*
Copyright (c) 2017, Michael Kazhdan
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list
of conditions and the following disclaimer. Redistributions in binary form must
reproduce the above copyright notice, this list of conditions and the following
disclaimer in the documentation and/or other materials provided with the
distribution.

Neither the name of the Johns Hopkins University nor the names of its
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE  GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#ifndef MY_MISCELLANY_INCLUDED
#define MY_MISCELLANY_INCLUDED

#include "PreProcessor.h"

#include <cxxabi.h>
#include <execinfo.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <sys/timeb.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

////////////////
// Time Stuff //
////////////////

inline double Time(void) {
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + double(t.tv_usec) / 1000000;
}
struct Timer {
    Timer(void) {
        _startCPUClock = std::clock(),
        _startWallClock = std::chrono::system_clock::now();
    }
    double cpuTime(void) const {
        return (std::clock() - _startCPUClock) / (double)CLOCKS_PER_SEC;
    };
    double wallTime(void) const {
        std::chrono::duration<double> diff =
                (std::chrono::system_clock::now() - _startWallClock);
        return diff.count();
    }

protected:
    std::clock_t _startCPUClock;
    std::chrono::time_point<std::chrono::system_clock> _startWallClock;
};

/////////////////////////////////////
// Exception, Warnings, and Errors //
/////////////////////////////////////

namespace MKExceptions {
template <typename... Arguments>
void _AddToMessageStream(std::stringstream &stream, Arguments... arguments);
inline void _AddToMessageStream(std::stringstream &stream) { return; }
template <typename Argument, typename... Arguments>
void _AddToMessageStream(std::stringstream &stream,
                         Argument argument,
                         Arguments... arguments) {
    stream << argument;
    _AddToMessageStream(stream, arguments...);
}

template <typename... Arguments>
std::string MakeMessageString(std::string header,
                              std::string fileName,
                              int line,
                              std::string functionName,
                              Arguments... arguments) {
    size_t headerSize = header.size();
    std::stringstream stream;

    // The first line is the header, the file name , and the line number
    stream << header << " " << fileName << " (Line " << line << ")"
           << std::endl;

    // Inset the second line by the size of the header and write the function
    // name
    for (size_t i = 0; i <= headerSize; i++) stream << " ";
    stream << functionName << std::endl;

    // Inset the third line by the size of the header and write the rest
    for (size_t i = 0; i <= headerSize; i++) stream << " ";
    _AddToMessageStream(stream, arguments...);

    return stream.str();
}
struct Exception : public std::exception {
    const char *what(void) const noexcept { return _message.c_str(); }
    template <typename... Args>
    Exception(const char *fileName,
              int line,
              const char *functionName,
              const char *format,
              Args... args) {
        _message = MakeMessageString("[EXCEPTION]", fileName, line,
                                     functionName, format, args...);
    }

private:
    std::string _message;
};

template <typename... Args>
void Throw(const char *fileName,
           int line,
           const char *functionName,
           const char *format,
           Args... args) {
    throw Exception(fileName, line, functionName, format, args...);
}
template <typename... Args>
void Warn(const char *fileName,
          int line,
          const char *functionName,
          const char *format,
          Args... args) {
    std::cerr << MakeMessageString("[WARNING]", fileName, line, functionName,
                                   format, args...)
              << std::endl;
}
template <typename... Args>
void ErrorOut(const char *fileName,
              int line,
              const char *functionName,
              const char *format,
              Args... args) {
    std::cerr << MakeMessageString("[ERROR]", fileName, line, functionName,
                                   format, args...)
              << std::endl;
    exit(0);
}
}  // namespace MKExceptions
#ifndef WARN
#define WARN(...) \
    MKExceptions::Warn(__FILE__, __LINE__, __FUNCTION__, __VA_ARGS__)
#endif  // WARN
#ifndef WARN_ONCE
#define WARN_ONCE(...)                                                         \
    {                                                                          \
        static bool firstTime = true;                                          \
        if (firstTime)                                                         \
            MKExceptions::Warn(__FILE__, __LINE__, __FUNCTION__, __VA_ARGS__); \
        firstTime = false;                                                     \
    }
#endif  // WARN_ONCE
#ifndef THROW
#define THROW(...) \
    MKExceptions::Throw(__FILE__, __LINE__, __FUNCTION__, __VA_ARGS__)
#endif  // THROW
#ifndef ERROR_OUT
#define ERROR_OUT(...) \
    MKExceptions::ErrorOut(__FILE__, __LINE__, __FUNCTION__, __VA_ARGS__)
#endif  // ERROR_OUT

template <typename Value>
bool SetAtomic(volatile Value *value, Value newValue, Value oldValue);
template <typename Data>
void AddAtomic(Data &a, Data b);

template <typename Value>
bool SetAtomic32(volatile Value *value, Value newValue, Value oldValue) {
    uint32_t &_oldValue = *(uint32_t *)&oldValue;
    uint32_t &_newValue = *(uint32_t *)&newValue;
    return __atomic_compare_exchange_n((uint32_t *)value, (uint32_t *)&oldValue,
                                       _newValue, false, __ATOMIC_SEQ_CST,
                                       __ATOMIC_SEQ_CST);
}
template <typename Value>
bool SetAtomic64(volatile Value *value, Value newValue, Value oldValue) {
    uint64_t &_oldValue = *(uint64_t *)&oldValue;
    uint64_t &_newValue = *(uint64_t *)&newValue;
    return __atomic_compare_exchange_n((uint64_t *)value, (uint64_t *)&oldValue,
                                       _newValue, false, __ATOMIC_SEQ_CST,
                                       __ATOMIC_SEQ_CST);
}

template <typename Number>
void AddAtomic32(Number &a, Number b) {
    Number current = a;
    Number sum = current + b;
    uint32_t &_current = *(uint32_t *)&current;
    uint32_t &_sum = *(uint32_t *)&sum;
    while (__sync_val_compare_and_swap((uint32_t *)&a, _current, _sum) !=
           _current)
        current = a, sum = a + b;
}

template <typename Number>
void AddAtomic64(Number &a, Number b) {
    Number current = a;
    Number sum = current + b;
    while (!SetAtomic64(&a, sum, current)) current = a, sum = a + b;
}

template <typename Value>
bool SetAtomic(volatile Value *value, Value newValue, Value oldValue) {
    switch (sizeof(Value)) {
        case 4:
            return SetAtomic32(value, newValue, oldValue);
        case 8:
            return SetAtomic64(value, newValue, oldValue);
        default:
            WARN_ONCE("should not use this function: ", sizeof(Value));
            static std::mutex setAtomicMutex;
            std::lock_guard<std::mutex> lock(setAtomicMutex);
            if (*value == oldValue) {
                *value = newValue;
                return true;
            } else
                return false;
    }
}

template <typename Data>
void AddAtomic(Data &a, Data b) {
    switch (sizeof(Data)) {
        case 4:
            return AddAtomic32(a, b);
        case 8:
            return AddAtomic64(a, b);
        default:
            WARN_ONCE("should not use this function: ", sizeof(Data));
            static std::mutex addAtomicMutex;
            std::lock_guard<std::mutex> lock(addAtomicMutex);
            a += b;
    }
}

//////////////////
// Memory Stuff //
//////////////////
size_t getPeakRSS(void);
size_t getCurrentRSS(void);

struct MemoryInfo {
    static size_t Usage(void) { return getCurrentRSS(); }
    static int PeakMemoryUsageMB(void) { return (int)(getPeakRSS() >> 20); }
};
inline void SetPeakMemoryMB(size_t sz) {
    sz <<= 20;
    struct rlimit rl;
    getrlimit(RLIMIT_AS, &rl);
    rl.rlim_cur = sz;
    setrlimit(RLIMIT_AS, &rl);
}
/*
 * Author:  David Robert Nadeau
 * Site:    http://NadeauSoftware.com/
 * License: Creative Commons Attribution 3.0 Unported License
 *          http://creativecommons.org/licenses/by/3.0/deed.en_US
 */

/**
 * Returns the peak (maximum so far) resident set size (physical
 * memory use) measured in bytes, or zero if the value cannot be
 * determined on this OS.
 */
inline size_t getPeakRSS() {
    struct rusage rusage;
    getrusage(RUSAGE_SELF, &rusage);
    return (size_t)(rusage.ru_maxrss * 1024L);
}

/**
 * Returns the current resident set size (physical memory use) measured
 * in bytes, or zero if the value cannot be determined on this OS.
 */
inline size_t getCurrentRSS() {
    long rss = 0L;
    FILE *fp = NULL;
    if ((fp = fopen("/proc/self/statm", "r")) == NULL)
        return (size_t)0L; /* Can't open? */
    if (fscanf(fp, "%*s%ld", &rss) != 1) {
        fclose(fp);
        return (size_t)0L; /* Can't read? */
    }
    fclose(fp);
    return (size_t)rss * (size_t)sysconf(_SC_PAGESIZE);
}

#endif  // MY_MISCELLANY_INCLUDED
