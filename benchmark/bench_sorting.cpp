#include <benchmark/benchmark.h>

#include <parlay/monoid.h>
#include <parlay/primitives.h>
#include <parlay/random.h>
#include <parlay/io.h>
#include "trigram_words.h"
#include <vector>
using benchmark::Counter;
std::vector<std::pair<int, std::string>> testcase = {{10, "uniform_10"},
                                                     {100, "uniform_100"},
                                                     {1000, "uniform_1000"},
                                                     {5000, "uniform_5000"},
                                                     {7000, "uniform_7000"},
                                                     {8000, "uniform_8000"},
                                                     {10000, "uniform_10000"},
                                                     {15000, "uniform_15000"},
                                                     {20000, "uniform_20000"},
                                                     {50000, "uniform_50000"},
                                                     {100000, "uniform_100000"},
                                                     {1000000, "uniform_1000000"},
                                                     {10000000, "uniform_10000000"},
                                                     {100000000, "uniform_100000000"},
                                                     {1000000000, "uniform_1000000000"},
                                                     {100000, "exp_1"},
                                                     {100, "exp_0.001"},
                                                     {30, "exp_0.0003"},
                                                     {20, "exp_0.0002"},
                                                     {15, "exp_0.00015"},
                                                     {10, "exp_0.0001"},
                                                     {1, "exp_0.00001"},
                                                     {10000,"zipfan_10000"},
                                                     {100000,"zipfan_100000"},
                                                     {1000000,"zipfan_1000000"},
                                                     {10000000,"zipfan_10000000"},
                                                     {100000000,"zipfan_100000000"},
                                                     {1000000000,"zipfan_1000000000"}};
using parlay::parallel_for;
// Use this macro to avoid accidentally timing the destructors
// of the output produced by algorithms that return data
//
// The expression e is evaluated as if written in the context
// auto result_ = (e);
//
#define RUN_AND_CLEAR(e)                                                                                               \
  {                                                                                                                    \
    auto result_ = (e);                                                                                                \
    state.PauseTiming();                                                                                               \
  }                                                                                                                    \
  state.ResumeTiming();

// Use this macro to copy y into x without measuring the cost
// of the copy in the benchmark
//
// The effect of this macro on the arguments x and is equivalent
// to the statement (x) = (y)
#define COPY_NO_TIME(x, y)                                                                                             \
  state.PauseTiming();                                                                                                 \
  (x) = (y);                                                                                                           \
  state.ResumeTiming();

// Report bandwidth and throughput statistics for the benchmark
//
// Arguments:
//  n:             The number of elements processed
//  bytes_read:    The number of bytes read per element processed
//  bytes_written: The number of bytes written per element processed
//
#define REPORT_STATS(n, bytes_read, bytes_written)                                                                     \
  state.counters["       Bandwidth"] =                                                                                 \
      Counter(state.iterations() * (n) * ((bytes_read) + 0.7 * (bytes_written)), Counter::kIsRate);                    \
  state.counters["    Elements/sec"] = Counter(state.iterations() * (n), Counter::kIsRate);                            \
  state.counters["       Bytes/sec"] = Counter(state.iterations() * (n) * (sizeof(T)), Counter::kIsRate);

