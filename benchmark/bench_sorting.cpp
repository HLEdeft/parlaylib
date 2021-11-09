#include <benchmark/benchmark.h>

#include "trigram_words.h"
#include <parlay/io.h>
#include <parlay/monoid.h>
#include <parlay/primitives.h>
#include <parlay/random.h>
#include <vector>
using benchmark::Counter;
std::vector<std::pair<int, std::string>> testcase = {
    {10, "uniform_10"},
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
    {10000, "zipfan_10000"},
    {100000, "zipfan_100000"},
    {1000000, "zipfan_1000000"},
    {10000000, "zipfan_10000000"},
    {100000000, "zipfan_100000000"},
    {1000000000, "zipfan_1000000000"},
    {117185083, "com-orkut.txt"},
    {1468365182, "twitter.txt"},
    {988965964, "yahoo_g9.txt"},
    {2043203933, "sd_arc.txt"},
    {101311613, "enwiki.txt"},
    {1019903190, "webbase2001.txt"},
    {298113762, "uk2002.txt"},
    {1806067135, "com-friendster.txt"}};
std::vector<std::pair<int, std::string>> ngram = {
    {92650278, "/data/zwang358/semisort/ngram/wp_2gram.txt"},
    {376671416, "/data/zwang358/semisort/ngram/wp_3gram.txt"}};
using parlay::parallel_for;
struct UnstablePair {
  uint64_t x, y;
  bool operator<(const UnstablePair &other) const { return x < other.x; }
  bool operator>(const UnstablePair &other) const { return x > other.x; }
  bool operator==(const UnstablePair &other) const {
    return x == other.x && y == other.y;
  }
};
// Use this macro to avoid accidentally timing the destructors
// of the output produced by algorithms that return data
//
// The expression e is evaluated as if written in the context
// auto result_ = (e);
//
#define RUN_AND_CLEAR(e)                                                       \
  {                                                                            \
    auto result_ = (e);                                                        \
    state.PauseTiming();                                                       \
  }                                                                            \
  state.ResumeTiming();

// Use this macro to copy y into x without measuring the cost
// of the copy in the benchmark
//
// The effect of this macro on the arguments x and is equivalent
// to the statement (x) = (y)
#define COPY_NO_TIME(x, y)                                                     \
  state.PauseTiming();                                                         \
  (x) = (y);                                                                   \
  state.ResumeTiming();

// Report bandwidth and throughput statistics for the benchmark
//
// Arguments:
//  n:             The number of elements processed
//  bytes_read:    The number of bytes read per element processed
//  bytes_written: The number of bytes written per element processed
//
#define REPORT_STATS(n, bytes_read, bytes_written)                             \
  state.counters["       Bandwidth"] = Counter(                                \
      state.iterations() * (n) * ((bytes_read) + 0.7 * (bytes_written)),       \
      Counter::kIsRate);                                                       \
  state.counters["    Elements/sec"] =                                         \
      Counter(state.iterations() * (n), Counter::kIsRate);                     \
  state.counters["       Bytes/sec"] =                                         \
      Counter(state.iterations() * (n) * (sizeof(T)), Counter::kIsRate);

template <typename K, typename V> struct mypair {
  K first;
  V second;
};

void scan_inplace__(uint32_t *in, uint32_t n) {
  if (n <= 1024) {
    for (size_t i = 1; i < n; i++)
      in[i] += in[i - 1];
    return;
  }
  uint32_t root_n = (uint32_t)sqrt(n); // split the array into root n blocks
  uint32_t *offset = new uint32_t[root_n - 1];
  parallel_for(
      0, root_n - 1,
      [&](size_t i) {
        offset[i] = 0;
        for (size_t j = i * root_n; j < (i + 1) * root_n; j++)
          offset[i] += in[j];
      },
      1);
  for (size_t i = 1; i < root_n - 1; i++)
    offset[i] += offset[i - 1];
  parlay::parallel_for(
      0, root_n,
      [&](size_t i) {
        if (i == root_n - 1) { // the last one
          for (size_t j = i * root_n + 1; j < n; j++) {
            in[j] += in[j - 1];
          }
        } else {
          for (size_t j = i * root_n + 1; j < (i + 1) * root_n; j++) {
            in[j] += in[j - 1];
          }
        }
      },
      1);
  parlay::parallel_for(
      1, root_n,
      [&](size_t i) {
        if (i == root_n - 1) {
          for (size_t j = i * root_n; j < n; j++) {
            in[j] += offset[i - 1];
          }
        } else {
          for (size_t j = i * root_n; j < (i + 1) * root_n; j++) {
            in[j] += offset[i - 1];
          }
        }
      },
      1);
  delete[] offset;
}
void exponential_generator_int64_(
    uint32_t exp_cutoff, double exp_lambda,
    parlay::sequence<std::pair<uint64_t, uint64_t>> &C) {

  uint32_t n = 1000000000;
  parlay::sequence<uint32_t> nums(exp_cutoff);
  parlay::sequence<std::pair<uint64_t, uint64_t>> B(n);
  // parlay::sequence<std::pair<uint64_t, uint64_t>> C(n);
  parallel_for(
      0, exp_cutoff,
      [&](uint32_t i) {
        nums[i] = (double)n * (exp(-exp_lambda * i) * (1 - exp(-exp_lambda)));
      },
      1);
  uint32_t offset = parlay::reduce(
      nums, parlay::addm<uint32_t>()); // cout << "offset = " << offset << endl;
  nums[0] += (n - offset);
  uint32_t *addr = new uint32_t[exp_cutoff];
  parlay::parallel_for(
      0, exp_cutoff, [&](uint32_t i) { addr[i] = nums[i]; }, 1);
  scan_inplace__(addr, exp_cutoff); // store all addresses into addr[]

  parlay::parallel_for(
      0, exp_cutoff,
      [&](size_t i) {
        uint32_t st = (i == 0) ? 0 : addr[i - 1],
                 ed = (i == (uint32_t)exp_cutoff - 1) ? n : addr[i];
        for (uint32_t j = st; j < ed; j++)
          B[j].first = parlay::hash64_2(i);
      },
      1);
  parlay::parallel_for(
      0, n, [&](size_t i) { B[i].second = parlay::hash64_2(i); }, 1);

  delete[] addr;
  C = parlay::random_shuffle(B, n);
}
void zipfian_generator_int64_(
    uint64_t zipf_s, parlay::sequence<std::pair<uint64_t, uint64_t>> &C) {
  uint32_t n = 1000000000;
  parlay::sequence<uint32_t> nums(zipf_s); // in total zipf_s kinds of keys
  parlay::sequence<std::pair<uint64_t, uint64_t>> B(n);
  // parlay::sequence<std::pair<uint64_t, uint64_t>> C(n);
  uint32_t number = (uint32_t)(n / log(n)); // number= n/ln(n)
  parlay::parallel_for(
      0, zipf_s, [&](uint32_t i) { nums[i] = (uint32_t)(number / (i + 1)); },
      1);
  uint32_t offset = parlay::reduce(
      nums, parlay::addm<uint32_t>()); // cout << "offset = " << offset << endl;
  nums[0] += (n - offset);
  uint32_t *addr = new uint32_t[zipf_s];
  parallel_for(
      0, zipf_s, [&](uint32_t i) { addr[i] = nums[i]; }, 1);
  scan_inplace__(addr, zipf_s); // store all addresses into addr[]
  parallel_for(
      0, zipf_s,
      [&](uint32_t i) {
        uint32_t st = (i == 0) ? 0 : addr[i - 1],
                 ed = (i == zipf_s - 1) ? n : addr[i];
        for (uint32_t j = st; j < ed; j++) {
          B[j].first = parlay::hash64_2(i);
        }
      },
      1);
  parlay::parallel_for(
      0, n, [&](size_t i) { B[i].second = parlay::hash64_2(i); }, 1);
  delete[] addr;
  C = parlay::random_shuffle(B, n);
}
void test_uniform(uint64_t uniform_max_range,
                  parlay::sequence<std::pair<uint64_t, uint64_t>> &C) {
  uint32_t n = 1000000000;
  // std::vector<std::pair<uint64_t, uint64_t>> C(n);
  parlay::parallel_for(
      0, n,
      [&](uint32_t i) {
        C[i].first = parlay::hash64_2(i) % uniform_max_range;
        if (C[i].first > uniform_max_range)
          C[i].first -= uniform_max_range;
        if (C[i].first > uniform_max_range)
          std::cout << "wrong..." << std::endl;
        C[i].first = parlay::hash64_2(C[i].first);
        C[i].second = parlay::hash64_2(i);
      },
      1);
}
void test_graph(const std::string& filename,
                parlay::sequence<std::pair<uint64_t, uint64_t>> &C) {
  std::ifstream fin;
  fin.open("/data/zwang358/semisort/graph/" + filename);
  uint32_t n;
  fin >> n;
  for (uint32_t i = 0; i < n; i++)
    fin >> C[i].first >> C[i].second;
}
template <typename T>
static void bench_integer_sort_pair(benchmark::State &state) {
  size_t testid = state.range(0);
  size_t n = testid < 28 ? 1000000000 : testcase[testid].first;
  const size_t ITERATION = 11;
  parlay::sequence<std::pair<uint64_t, uint64_t>> A(n);
  if (testid < 15)
    test_uniform(testcase[testid].first, A);
  else if (testid < 22)
    exponential_generator_int64_(1000000, testcase[testid].first / 100000.0, A);
  else if (testid < 28)
    zipfian_generator_int64_(testcase[testid].first, A);
  else if (testid < 35)
    test_graph(testcase[testid].second, A);
  // Modify Here!
  parlay::sequence<mypair<uint64_t, uint64_t>> s(n);
  parallel_for(0, n, [&](size_t i) {
    s[i].first = A[i].first;
    s[i].second = A[i].second;
  });
  size_t bits = sizeof(T) * 8;
  auto first = [](const auto &x) -> uint64_t { return x.first; };
  auto sorted =
      parlay::internal::integer_sort(parlay::make_slice(s), first, bits);
  std::cout << "start testcase:" << testcase[testid].second << std::endl;
  while (state.KeepRunningBatch(ITERATION)) {
    for (size_t i = 0; i < ITERATION; i++) {
      RUN_AND_CLEAR(
          parlay::internal::integer_sort(parlay::make_slice(s), first, bits));
    }
  }

  REPORT_STATS(n, 0, 0);
}

template <typename T>
static void bench_integer_sort_inplace_pair(benchmark::State &state) {
  size_t testid = state.range(0);
  size_t n = testid < 28 ? 1000000000 : testcase[testid].first;
  const size_t ITERATION = 11;
   parlay::sequence<std::pair<uint64_t, uint64_t>> A(n);
  if (testid < 15)
    test_uniform(testcase[testid].first, A);
  else if (testid < 22)
    exponential_generator_int64_(1000000, testcase[testid].first / 100000.0, A);
  else if (testid < 28)
    zipfian_generator_int64_(testcase[testid].first, A);
  else if (testid < 35)
    test_graph(testcase[testid].second, A);

  // Modify Here!
  parlay::sequence<mypair<uint64_t, uint64_t>> s(n);
  parallel_for(0, n, [&](size_t i) {
    s[i].first = A[i].first;
    s[i].second = A[i].second;
  });
  size_t bits = sizeof(T) * 8;
  auto first = [](const auto &x) -> uint64_t { return x.first; };
  auto sorted = s;
  parlay::internal::integer_sort_inplace(parlay::make_slice(sorted), first,
                                         bits);
  std::cout << "start testcase:" << testcase[testid].second << std::endl;
  while (state.KeepRunningBatch(ITERATION)) {
    for (size_t i = 0; i < ITERATION; i++) {
      COPY_NO_TIME(sorted, s);
      parlay::internal::integer_sort_inplace(parlay::make_slice(sorted), first,
                                             bits);
    }
  }

  REPORT_STATS(n, 0, 0);
}

template <typename T>
static void bench_sample_sort_stable(benchmark::State &state) {
  size_t testid = state.range(0);
  size_t n = testid < 28 ? 1000000000 : testcase[testid].first;
  const size_t ITERATION = 11;
  parlay::sequence<std::pair<uint64_t, uint64_t>> A(n);
  if (testid < 15)
    test_uniform(testcase[testid].first, A);
  else if (testid < 22)
    exponential_generator_int64_(1000000, testcase[testid].first / 100000.0, A);
  else if (testid < 28)
    zipfian_generator_int64_(testcase[testid].first, A);
  else if (testid < 35)
    test_graph(testcase[testid].second, A);
  // Modify Here!
  parlay::sequence<mypair<uint64_t, uint64_t>> s(n);
  parallel_for(0, n, [&](size_t i) {
    s[i].first = A[i].first;
    s[i].second = A[i].second;
  });
  auto Comp = [&](mypair<uint64_t, uint64_t> a, mypair<uint64_t, uint64_t> b) {
    return a.first < b.first;
  };
  auto sorted = parlay::internal::sample_sort(parlay::make_slice(s),
                                              Comp, true);
  std::cout << "start testcase:" << testcase[testid].second << std::endl;
  while (state.KeepRunningBatch(ITERATION)) {
    for (size_t i = 0; i < ITERATION; i++) {
      RUN_AND_CLEAR(parlay::internal::sample_sort(
          parlay::make_slice(s), Comp, true));
    }
  }

  REPORT_STATS(n, 0, 0);
}
template <typename T>
static void bench_sample_sort_unstable(benchmark::State &state) {
  size_t testid = state.range(0);
  size_t n = testid < 28 ? 1000000000 : testcase[testid].first;
  const size_t ITERATION = 11;
  parlay::sequence<std::pair<uint64_t, uint64_t>> A(n);
  if (testid < 15)
    test_uniform(testcase[testid].first, A);
  else if (testid < 22)
    exponential_generator_int64_(1000000, testcase[testid].first / 100000.0, A);
  else if (testid < 28)
    zipfian_generator_int64_(testcase[testid].first, A);
  else if (testid < 35)
    test_graph(testcase[testid].second, A);
  // Modify Here!
  parlay::sequence<mypair<uint64_t, uint64_t>> s(n);
  parallel_for(0, n, [&](size_t i) {
    s[i].first = A[i].first;
    s[i].second = A[i].second;
  });
  auto Comp = [&](mypair<uint64_t, uint64_t> a, mypair<uint64_t, uint64_t> b) {
    return a.first < b.first;
  };
  std::cout << "start testcase:" << testcase[testid].second << std::endl;

  auto sorted = parlay::internal::sample_sort(parlay::make_slice(s),
                                              Comp, false);
  while (state.KeepRunningBatch(ITERATION)) {
    for (size_t i = 0; i < ITERATION; i++) {
      RUN_AND_CLEAR(parlay::internal::sample_sort(
          parlay::make_slice(s), Comp, false));
    }
  }

  REPORT_STATS(n, 0, 0);
}

template <typename T>
static void bench_sample_sort_unstable_inplace(benchmark::State &state) {
  size_t testid = state.range(0);
  size_t n = testid < 28 ? 1000000000 : testcase[testid].first;
  const size_t ITERATION = 10;
  parlay::sequence<std::pair<uint64_t, uint64_t>> A(n);
  if (testid < 15)
    test_uniform(testcase[testid].first, A);
  else if (testid < 22)
    exponential_generator_int64_(1000000, testcase[testid].first / 100000.0, A);
  else if (testid < 28)
    zipfian_generator_int64_(testcase[testid].first, A);
  else if (testid < 35)
    test_graph(testcase[testid].second, A);
  // Modify Here!
  // auto s = parlay::tabulate(n, [&](uint32_t i) -> UnstablePair {
  //   UnstablePair x;
  //   x.x = A[i].first;
  //   x.y = A[i].second;
  //   return x;
  // });
  parlay::sequence<mypair<uint64_t, uint64_t>> s(n);
  parallel_for(0, n, [&](size_t i) {
    s[i].first = A[i].first;
    s[i].second = A[i].second;
  });
  auto Comp = [&](mypair<uint64_t, uint64_t> a, mypair<uint64_t, uint64_t> b) {
    return a.first < b.first;
  };
  auto sorted = s;
  parlay::internal::sample_sort_inplace(parlay::make_slice(sorted),
                                        Comp);
  std::cout << "start testcase:" << testcase[testid].second << std::endl;
  while (state.KeepRunningBatch(ITERATION)) {
    for (size_t i = 0; i < ITERATION; i++) {
      COPY_NO_TIME(sorted, s);
      parlay::internal::sample_sort_inplace(parlay::make_slice(sorted),
                                            Comp);
    }
  }

  REPORT_STATS(n, 0, 0);
}
template <typename T>
static void bench_group_by_key_sorted(benchmark::State &state) {
  size_t testid = state.range(0);
  size_t n = testid < 28 ? 1000000000 : testcase[testid].first;
  const size_t ITERATION = 11;
  parlay::sequence<std::pair<uint64_t, uint64_t>> A(n);
  if (testid < 15)
    test_uniform(testcase[testid].first, A);
  else if (testid < 22)
    exponential_generator_int64_(1000000, testcase[testid].first / 100000.0, A);
  else if (testid < 28)
    zipfian_generator_int64_(testcase[testid].first, A);
  else if (testid < 35)
    test_graph(testcase[testid].second, A);
  std::cout << "start testcase:" << testcase[testid].second << std::endl;
  auto s = A;

  auto sorted = parlay::group_by_key_sorted(s);
  while (state.KeepRunningBatch(ITERATION)) {
    for (size_t i = 0; i < ITERATION; i++) {
      RUN_AND_CLEAR(parlay::group_by_key_sorted(s));
    }
  }

  REPORT_STATS(n, 0, 0);
}
template <typename T> static void bench_group_by_key(benchmark::State &state) {
  size_t testid = state.range(0);
  size_t n = testid < 28 ? 1000000000 : testcase[testid].first;
  const size_t ITERATION = 10;
  parlay::sequence<std::pair<uint64_t, uint64_t>> A(n);
  if (testid < 15)
    test_uniform(testcase[testid].first, A);
  else if (testid < 22)
    exponential_generator_int64_(1000000, testcase[testid].first / 100000.0, A);
  else if (testid < 28)
    zipfian_generator_int64_(testcase[testid].first, A);
  else if (testid < 35)
    test_graph(testcase[testid].second, A);
  // auto sorted = parlay::group_by_key(parlay::make_slice(s));
  auto in = A;
  auto out = parlay::group_by_key(in);
  std::cout << "start testcase:" << testcase[testid].second << std::endl;
  while (state.KeepRunningBatch(ITERATION)) {
    for (size_t i = 0; i < ITERATION; i++) {
      COPY_NO_TIME(in, A);
      RUN_AND_CLEAR(parlay::group_by_key(in));
    }
  }
  REPORT_STATS(n, 0, 0);
}
template <typename T>
static void bench_group_by_key_sorted_ngram(benchmark::State &state) {
  size_t testid = state.range(0);
  std::ifstream fin;
  fin.open(ngram[testid].second);
  size_t n;
  fin >> n;
  const size_t ITERATION = 11;
  using NG = std::pair<std::string, std::pair<std::string, uint64_t>>;
  parlay::sequence<NG> A(n);
  for (size_t i = 0; i < n; i++)
    fin>>A[i].second.second >> A[i].first >> A[i].second.first;
  std::cout << "start testcase:" << ngram[testid].second << std::endl;
  auto s = A;

  auto sorted = parlay::group_by_key_sorted(s);
  while (state.KeepRunningBatch(ITERATION)) {
    for (size_t i = 0; i < ITERATION; i++) {
      RUN_AND_CLEAR(parlay::group_by_key_sorted(s));
    }
  }

  REPORT_STATS(n, 0, 0);
}
template <typename T>
static void bench_sample_sort_unstable_inplace_ngram(benchmark::State &state) {
  size_t testid = state.range(0);
  std::ifstream fin;
  fin.open(ngram[testid].second);
  size_t n;
  fin >> n;
  const size_t ITERATION = 11;
  using NG = mypair<std::string, mypair<std::string, uint64_t>>;
  parlay::sequence<NG> s(n);
  for (size_t i = 0; i < n; i++)
    fin >> s[i].second.second >> s[i].first >> s[i].second.first;
  auto Comp = [&](NG a, NG b) {
    return a.first < b.first;
  };
  auto sorted = s;
  parlay::internal::sample_sort_inplace(parlay::make_slice(sorted),
                                        Comp);
  std::cout << "start testcase:" << ngram[testid].second << std::endl;
  while (state.KeepRunningBatch(ITERATION)) {
    for (size_t i = 0; i < ITERATION; i++) {
      COPY_NO_TIME(sorted, s);
      parlay::internal::sample_sort_inplace(parlay::make_slice(sorted),
                                            Comp);
    }
  }

  REPORT_STATS(n, 0, 0);
}
template <typename T>
static void bench_sample_sort_stable_ngram(benchmark::State &state) {
  size_t testid = state.range(0);
  std::ifstream fin;
  fin.open(ngram[testid].second);
  size_t n;
  fin >> n;
  const size_t ITERATION = 11;
  using NG = mypair<std::string, mypair<std::string, uint64_t>>;
  parlay::sequence<NG> s(n);
  for (size_t i = 0; i < n; i++)
    fin >> s[i].second.second >> s[i].first >> s[i].second.first;
  auto Comp = [&](NG a, NG b) {
    return a.first < b.first;
  };
  auto sorted = parlay::internal::sample_sort(parlay::make_slice(s),
                                              Comp, true);
  std::cout << "start testcase:" << ngram[testid].second << std::endl;
  while (state.KeepRunningBatch(ITERATION)) {
    for (size_t i = 0; i < ITERATION; i++) {
      RUN_AND_CLEAR(parlay::internal::sample_sort(
          parlay::make_slice(s), Comp, true));
    }
  }

  REPORT_STATS(n, 0, 0);
}
template <typename T>
static void bench_sample_sort_unstable_ngram(benchmark::State &state) {
   size_t testid = state.range(0);
  std::ifstream fin;
  fin.open(ngram[testid].second);
  size_t n;
  fin >> n;
  const size_t ITERATION = 11;
  using NG = mypair<std::string, mypair<std::string, uint64_t>>;
  parlay::sequence<NG> s(n);
  for (size_t i = 0; i < n; i++)
    fin >> s[i].second.second >> s[i].first >> s[i].second.first;
  auto Comp = [&](NG a, NG b) {
    return a.first < b.first;
  };
  std::cout << "start testcase:" << ngram[testid].second << std::endl;

  auto sorted = parlay::internal::sample_sort(parlay::make_slice(s),
                                              Comp, false);
  while (state.KeepRunningBatch(ITERATION)) {
    for (size_t i = 0; i < ITERATION; i++) {
      RUN_AND_CLEAR(parlay::internal::sample_sort(
          parlay::make_slice(s), Comp, false));
    }
  }

  REPORT_STATS(n, 0, 0);
}

// ------------------------- Registration -------------------------------

#define BENCH(NAME, T, args...)                                                \
  BENCHMARK_TEMPLATE(bench_##NAME, T)                                          \
      ->UseRealTime()                                                          \
      ->Unit(benchmark::kMillisecond)                                          \
      ->Args({args});

static void custom_args(benchmark::internal::Benchmark *b) {
  for (size_t i = 0; i < 2; i++) {
    b->Arg(i);
  }
}
// BENCH(integer_sort_pair, unsigned long, 0);
BENCHMARK_TEMPLATE(bench_sample_sort_stable_ngram, unsigned long)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond)
    ->Apply(custom_args);
BENCHMARK_TEMPLATE(bench_sample_sort_unstable_ngram, unsigned long)
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond)
    ->Apply(custom_args);
// BENCHMARK_TEMPLATE(bench_group_by_key_sorted_ngram, unsigned long)
//     ->UseRealTime()
//     ->Unit(benchmark::kMillisecond)
//     ->Apply(custom_args);
// BENCHMARK_TEMPLATE(bench_sample_sort_unstable_inplace_ngram, std::string)
//     ->UseRealTime()
//     ->Unit(benchmark::kMillisecond)
//     ->Apply(custom_args);
// BENCHMARK_TEMPLATE(bench_group_by_key, unsigned long)
//     ->UseRealTime()
//     ->Unit(benchmark::kMillisecond)
//     ->Apply(custom_args);
// BENCHMARK_TEMPLATE(bench_group_by_key_sorted, unsigned long)
//     ->UseRealTime()
//     ->Unit(benchmark::kMillisecond)
//     ->Apply(custom_args);
// BENCHMARK_TEMPLATE(bench_integer_sort_pair, unsigned long)
//     ->UseRealTime()
//     ->Unit(benchmark::kMillisecond)
//     ->Apply(custom_args);
// BENCHMARK_TEMPLATE(bench_integer_sort_inplace_pair, unsigned long)
//     ->UseRealTime()
//     ->Unit(benchmark::kMillisecond)
//     ->Apply(custom_args);
// BENCHMARK_TEMPLATE(bench_sample_sort_stable, unsigned long)
//     ->UseRealTime()
//     ->Unit(benchmark::kMillisecond)
//     ->Apply(custom_args);
// BENCHMARK_TEMPLATE(bench_sample_sort_unstable, unsigned long)
//     ->UseRealTime()
//     ->Unit(benchmark::kMillisecond)
//     ->Apply(custom_args);
// BENCHMARK_TEMPLATE(bench_sample_sort_unstable_inplace, unsigned long)
//     ->UseRealTime()
//     ->Unit(benchmark::kMillisecond)
//     ->Apply(custom_args);