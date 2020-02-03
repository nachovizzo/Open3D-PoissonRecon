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

#ifndef THREADPOOL_H_
#define THREADPOOL_H_

#include "MyMiscellany.h"

#include <omp.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <future>
#include <string>
#include <thread>
#include <vector>

struct ThreadPool {
    static const size_t chunk_size = 128;
    static bool _Close;
    static volatile unsigned int _RemainingTasks;
    static std::condition_variable _WaitingForWorkOrClose, _DoneWithWork;
    static std::vector<std::thread> _Threads;
    static std::function<void(unsigned int)> _ThreadFunction;

    static void Parallel_for(size_t begin,
                             size_t end,
                             const std::function<void(unsigned int, size_t)>
                                     &iterationFunction) {
        if (begin >= end) {
            return;
        }

        size_t range = end - begin;
        size_t chunks = (range + chunk_size - 1) / chunk_size;
        unsigned int threads = (unsigned int)NumThreads();
        std::atomic<size_t> index;
        index.store(0);

        if (range < chunk_size || threads == 1) {
            for (size_t i = begin; i < end; i++) {
                iterationFunction(0, i);
            }
            return;
        }

        auto _ChunkFunction = [&iterationFunction, begin, end, chunk_size](
                                      unsigned int thread, size_t chunk) {
            const size_t _begin = begin + chunk_size * chunk;
            const size_t _end = std::min<size_t>(end, _begin + chunk_size);
            for (size_t i = _begin; i < _end; i++) iterationFunction(thread, i);
        };
        _ThreadFunction = [&_ChunkFunction, chunks,
                           &index](unsigned int thread) {
            size_t chunk;
            while ((chunk = index.fetch_add(1)) < chunks)
                _ChunkFunction(thread, chunk);
        };

#pragma omp parallel for
        for (size_t c = 0; c < chunks; c++) {
            _ChunkFunction(omp_get_thread_num(), c);
        }
    }

    static unsigned int NumThreads(void) {
        return (unsigned int)_Threads.size() + 1;
    }

    static void Init(
            unsigned int numThreads = std::thread::hardware_concurrency()) {
        if (_Threads.size() && !_Close) {
            _Close = true;
            _WaitingForWorkOrClose.notify_all();
            for (unsigned int t = 0; t < _Threads.size(); t++)
                _Threads[t].join();
        }
        _Close = true;
        numThreads--;
        _Threads.resize(numThreads);
    }

    static void Terminate(void) {
        if (_Threads.size() && !_Close) {
            _Close = true;
            _WaitingForWorkOrClose.notify_all();
            for (unsigned int t = 0; t < _Threads.size(); t++)
                _Threads[t].join();
            _Threads.resize(0);
        }
    }

    template <typename... Functions>
    static void ParallelSections(const Functions &... functions) {
        std::vector<std::future<void>> futures(sizeof...(Functions));
        _ParallelSections(&futures[0], functions...);
        for (size_t t = 0; t < futures.size(); t++) futures[t].get();
    }

    template <typename Function>
    static void _ParallelSections(std::future<void> *futures,
                                  const Function &function) {
        *futures = std::async(std::launch::async, function);
    }

    template <typename Function, typename... Functions>
    static void _ParallelSections(std::future<void> *futures,
                                  const Function &function,
                                  const Functions &... functions) {
        *futures = std::async(std::launch::async, function);
        _ParallelSections(futures + 1, functions...);
    }
};

// Hack to avoid using the linker for now
bool ThreadPool::_Close;
volatile unsigned int ThreadPool::_RemainingTasks;
std::condition_variable ThreadPool::_WaitingForWorkOrClose;
std::condition_variable ThreadPool::_DoneWithWork;
std::vector<std::thread> ThreadPool::_Threads;
std::function<void(unsigned int)> ThreadPool::_ThreadFunction;

#endif  // THREADPOOL_H_
