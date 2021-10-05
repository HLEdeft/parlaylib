// A simple benchmark set with good performance coverage
// The main set used to evaluate performance enhancements
// to the library

#include <benchmark/benchmark.h>

#include <parlay/monoid.h>
#include <parlay/primitives.h>
#include <parlay/random.h>
#include <parlay/io.h>
#include "trigram_words.h"
#include <vector>
using benchmark::Counter;
int testid[] = {7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 26,
                27, 28, 29, 30, 33, 34, 35, 36, 37, 38, 41, 42, 43, 44, 45, 46, 47, 48};
int uniform[] = {0,     0,     0,     0,     10,     100,     1000,     5000,      7000,      8000,
                 10000, 15000, 20000, 50000, 100000, 1000000, 10000000, 100000000, 1000000000};
int zipfan[] = {10000, 100000, 1000000, 10000000, 100000000, 1000000000};
double exp_lambda[] = {1, 0.001, 0.0003, 0.0002, 0.00015, 0.0001, 0.00001};
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

template<typename K, typename V>
struct mypair {
  K first;
  V second;
};
void scan_inplace__(uint32_t* in, uint32_t n) {
  if (n <= 1024) {
    // cout << "THRESHOLDS: " << THRESHOLDS << endl;
    for (size_t i = 1; i < n; i++) {
      in[i] += in[i - 1];
    }
    return;
  }
  uint32_t root_n = (uint32_t)sqrt(n);  // split the array into root n blocks
  uint32_t* offset = new uint32_t[root_n - 1];

  parlay::parallel_for(
      0, root_n - 1,
      [&](size_t i) {
        offset[i] = 0;
        for (size_t j = i * root_n; j < (i + 1) * root_n; j++) {
          offset[i] += in[j];
        }
      },
      1);

  for (size_t i = 1; i < root_n - 1; i++)
    offset[i] += offset[i - 1];

  // prefix sum for each subarray
  parlay::parallel_for(
      0, root_n,
      [&](size_t i) {
        if (i == root_n - 1) {  // the last one
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
// void zipfian_generator_int64_ (mypair<uint64_t, uint64_t> *A,
//                                int n, uint32_t zipf_s) {
//     sequence<uint32_t> nums(zipf_s); // in total zipf_s kinds of keys
//     sequence<mypair<uint64_t, uint64_t>> B(n);

//     /* 1. making nums[] array */
//     uint32_t number = (uint32_t) (n / log(n)); // number= n/ln(n)
//     parallel_for (0, zipf_s, [&] (uint32_t i) {
//         nums[i] = (uint32_t) (number / (i+1));
//     }, 1);

//     // the last nums[zipf_s-1] should be (n - \sum{zipf_s-1}nums[])
//     uint32_t offset = reduce(nums, addm<uint32_t>()); // cout << "offset = " << offset << endl;
//     // nums[zipf_s-1] += (n - offset);
//     nums[0] += (n - offset);

//     // checking if the sum of nums[] equals to n
//     if (reduce(nums, addm<uint32_t>()) == (uint32_t)n) {
//         cout << "sum of nums[] == n" << endl;
//     }

//     /* 2. scan to calculate position */
//     uint32_t* addr = new uint32_t[zipf_s];
//     parallel_for (0, zipf_s, [&] (uint32_t i) {
//         addr[i] = nums[i];
//     }, 1);
//     scan_inplace__(addr, zipf_s); // store all addresses into addr[]

//     /* 3. distribute random numbers into A[i].first */
//     parallel_for (0, zipf_s, [&] (uint32_t i) {
//         uint32_t st = (i == 0) ? 0 : addr[i-1],
//                  ed = (i == zipf_s-1) ? n : addr[i];
//         for (uint32_t j = st; j < ed; j++) {
//             B[j].first = parlay::hash64_2(i);
//         }
//     }, 1);
//     parallel_for (0, n, [&] (size_t i){
//         B[i].second = parlay::hash64_2(i);
//     }, 1);

//     /* 4. shuffle the keys */
//     // random_shuffle(A, A + n);
//     sequence<mypair<uint64_t, uint64_t>> C = parlay::random_shuffle(B, n);

//     parallel_for (0, n, [&] (size_t i) {
//         A[i] = C[i];
//     });

//     delete[] addr;
// }
void zipfian_generator_int64_(uint64_t zipf_s, int line, parlay::sequence<std::pair<uint64_t, uint64_t>>& A) {
  // pbbs::sequence<uint64_t> nums(zipf_s); // in total zipf_s kinds of keys
  int n = 1000000000;
  parlay::sequence<uint32_t> nums(zipf_s);  // in total zipf_s kinds of keys
  parlay::sequence<std::pair<uint64_t, uint64_t>> B(n);
  parlay::sequence<std::pair<uint64_t, uint64_t>> C(n);
  /* 1. making nums[] array */
  uint32_t number = (uint32_t)(n / log(n));  // number= n/ln(n)
  parlay::parallel_for(
      0, zipf_s, [&](uint32_t i) { nums[i] = (uint32_t)(number / (i + 1)); }, 1);

  uint32_t offset = parlay::reduce(nums, parlay::addm<uint32_t>());  // cout << "offset = " << offset << endl;
  // nums[zipf_s-1] += (n - offset);
  nums[0] += (n - offset);

  // checking if the sum of nums[] equals to n
  // if (parlay::reduce(nums, parlay::addm<uint32_t>()) == (uint32_t)n) {
  //   std::cout << "sum of nums[] == n" << std::endl;
  // }

  /* 2. scan to calculate position */
  // pbbs::sequence<uint32_t> addr(zipf_s);
  uint32_t* addr = new uint32_t[zipf_s];
  parlay::parallel_for(
      0, zipf_s, [&](uint32_t i) { addr[i] = nums[i]; }, 1);
  scan_inplace__(addr, zipf_s);  // store all addresses into addr[]

  /* 3. distribute random numbers into A[i].first */
  parlay::parallel_for(
      0, zipf_s,
      [&](uint32_t i) {
        uint32_t st = (i == 0) ? 0 : addr[i - 1], ed = (i == zipf_s - 1) ? n : addr[i];
        for (uint32_t j = st; j < ed; j++) {
          B[j].first = parlay::hash64_2(i);
        }
      },
      1);
  parlay::parallel_for(
      0, n, [&](size_t i) { B[i].second = parlay::hash64_2(i); }, 1);
  delete[] addr;
  /* 4. shuffle the keys */
  // random_shuffle(A, A + n);
  C = parlay::random_shuffle(B, n);
  parlay::parallel_for(0, n, [&](size_t i) {
    A[i].first = C[i].first;
    A[i].second = C[i].second;
  });
  // parlay::sequence<std::pair<uint64_t, uint64_t>> s(n);
  std::cout << "running test of zipfan line " << line << std::endl;
}
void exponential_generator_int64_(int exp_cutoff, double exp_lambda, int line,
                                  parlay::sequence<std::pair<uint64_t, uint64_t>>& A) {
  int n = 1000000000;
  parlay::sequence<uint32_t> nums(exp_cutoff);
  parlay::sequence<std::pair<uint64_t, uint64_t>> B(n);
  parlay::sequence<std::pair<uint64_t, uint64_t>> C(n);

  /* 1. making nums[] array */
  parallel_for(
      0, exp_cutoff, [&](uint32_t i) { nums[i] = (double)n * (exp(-exp_lambda * i) * (1 - exp(-exp_lambda))); }, 1);

  uint32_t offset = parlay::reduce(nums, parlay::addm<uint32_t>());  // cout << "offset = " << offset << endl;
  nums[0] += (n - offset);
  // if (parlay::reduce(nums, parlay::addm<uint32_t>()) == (uint32_t)n) {
  //   std::cout << "sum of nums[] == n" << std::endl;
  // }

  /* 2. scan to calculate position */
  uint32_t* addr = new uint32_t[exp_cutoff];
  parlay::parallel_for(
      0, exp_cutoff, [&](uint32_t i) { addr[i] = nums[i]; }, 1);
  scan_inplace__(addr, exp_cutoff);  // store all addresses into addr[]

  /* 3. distribute random numbers into A[i].first */
  parlay::parallel_for(
      0, exp_cutoff,
      [&](size_t i) {
        uint32_t st = (i == 0) ? 0 : addr[i - 1], ed = (i == (uint32_t)exp_cutoff - 1) ? n : addr[i];
        for (uint32_t j = st; j < ed; j++) {
          B[j].first = parlay::hash64_2(i);
        }
      },
      1);
  parlay::parallel_for(
      0, n, [&](size_t i) { B[i].second = parlay::hash64_2(i); }, 1);

  /* 4. shuffle the keys */
  delete[] addr;
  C = parlay::random_shuffle(B, n);
  parlay::parallel_for(0, n, [&](uint32_t i) {
    A[i].first = C[i].first;
    A[i].second = C[i].second;
  });
  std::cout << "running test of exp line " << line << std::endl;
}

void test_uniform(uint64_t uniform_max_range, int line, parlay::sequence<std::pair<uint64_t, uint64_t>>& s) {
  uint32_t n = 1000000000;
  std::vector<std::pair<uint64_t, uint64_t>> A(n);
  parlay::parallel_for(
      0, n,
      [&](uint32_t i) {
        // in order to put all keys in range [0, uniform_max_range]
        A[i].first = parlay::hash64_2(i) % uniform_max_range;
        if (A[i].first > uniform_max_range) A[i].first -= uniform_max_range;
        if (A[i].first > uniform_max_range) std::cout << "wrong..." << std::endl;
        A[i].first = parlay::hash64_2(A[i].first);
        A[i].second = parlay::hash64_2(i);
      },
      1);
  std::cout << "runing test of uniform input with line " << line << std::endl;
  // auto s = parlay::map(A, [](auto x) { return std::make_pair(x.first,
  // x.second); } ); parlay::sequence<std::pair<uint64_t, uint64_t>> s(n);
  // parlay::sequence<mypair<uint64_t, uint64_t>> s(n);
  parlay::parallel_for(0, n, [&](uint32_t i) {
    s[i].first = A[i].first;
    s[i].second = A[i].second;
  });
}
void testgraph(std::string graph_name, parlay::sequence<std::pair<uint64_t, uint64_t>>& s) {
  std::ifstream fin;
  fin.open(graph_name);
  unsigned long long m;
  fin >> m;
  // std::vector<mypair<uint64_t, uint64_t>> a(m);
  // parlay::sequence<std::pair<uint64_t, uint64_t>> a(m);
  // parlay::sequence<std::pair<uint64_t, uint64_t>> s(m);
  //  sequence<mypair<uint64_t, uint64_t>> A1(n);
  //   mypair<uint64_t, uint64_t>* A = new mypair<uint64_t, uint64_t>[n];
  for (unsigned long long i = 0; i < m; i++)
    fin >> s[i].first >> s[i].second;
  std::cout << "running test with input file " << graph_name << std::endl;
}
template<typename T>
static void bench_map(benchmark::State& state) {
  size_t n = state.range(0);
  auto In = parlay::sequence<T>(n, 1);
  auto f = [&](auto x) -> T { return x; };

  for (auto _ : state) {
    RUN_AND_CLEAR(parlay::map(In, f));
  }

  REPORT_STATS(n, 2 * sizeof(T), sizeof(T));
}

template<typename T>
static void bench_tabulate(benchmark::State& state) {
  size_t n = state.range(0);
  auto f = [](size_t i) -> T { return i; };

  for (auto _ : state) {
    RUN_AND_CLEAR(parlay::tabulate(n, f));
  }

  REPORT_STATS(n, sizeof(T), sizeof(T));
}

template<typename T>
static void bench_reduce_add(benchmark::State& state) {
  size_t n = state.range(0);
  auto s = parlay::sequence<T>(n, 1);

  for (auto _ : state) {
    [[maybe_unused]] auto sum = parlay::reduce(s);
  }

  REPORT_STATS(n, sizeof(T), 0);
}

template<typename T>
static void bench_scan_add(benchmark::State& state) {
  size_t n = state.range(0);
  auto s = parlay::sequence<T>(n, 1);

  for (auto _ : state) {
    RUN_AND_CLEAR(parlay::scan(s).first);
  }

  REPORT_STATS(n, 3 * sizeof(T), sizeof(T));
}

template<typename T>
static void bench_pack(benchmark::State& state) {
  size_t n = state.range(0);
  auto flags = parlay::tabulate(n, [](size_t i) -> bool { return i % 2; });
  auto In = parlay::tabulate(n, [](size_t i) -> T { return i; });

  for (auto _ : state) {
    RUN_AND_CLEAR(parlay::pack(In, flags));
  }

  REPORT_STATS(n, 14, 4);  // Why 14 and 4?
}

template<typename T>
static void bench_gather(benchmark::State& state) {
  size_t n = state.range(0);
  parlay::random r(0);
  auto in = parlay::tabulate(n, [&](size_t i) -> T { return i; });
  auto in_slice = parlay::make_slice(in);
  auto idx = parlay::tabulate(n, [&](size_t i) -> T { return r.ith_rand(i) % n; });
  auto idx_slice = parlay::make_slice(idx);
  auto f = [&](size_t i) -> T {
    // prefetching helps significantly
    __builtin_prefetch(&in[idx[i + 4]], 0, 1);
    return in_slice[idx_slice[i]];
  };

  for (auto _ : state) {
    if (n > 4) {
      RUN_AND_CLEAR(parlay::tabulate(n - 4, f));
    }
  }

  REPORT_STATS(n, 10 * sizeof(T), sizeof(T));
}

template<typename T>
static void bench_scatter(benchmark::State& state) {
  size_t n = state.range(0);
  parlay::random r(0);
  parlay::sequence<T> out(n, 0);
  auto out_slice = parlay::make_slice(out);
  auto idx = parlay::tabulate(n, [&](size_t i) -> T { return r.ith_rand(i) % n; });
  auto idx_slice = parlay::make_slice(idx);
  auto f = [&](size_t i) {
    // prefetching makes little if any difference
    //__builtin_prefetch (&out[idx[i+4]], 1, 1);
    out_slice[idx_slice[i]] = i;
  };

  for (auto _ : state) {
    if (n > 4) {
      parlay::parallel_for(0, n - 4, f);
    }
  }

  REPORT_STATS(n, 9 * sizeof(T), 8 * sizeof(T));
}

template<typename T>
static void bench_write_add(benchmark::State& state) {
  size_t n = state.range(0);
  parlay::random r(0);

  auto out = parlay::sequence<std::atomic<T>>(n);
  auto out_slice = parlay::make_slice(out);

  auto idx = parlay::tabulate(n, [&](size_t i) -> T { return r.ith_rand(i) % n; });
  auto idx_slice = parlay::make_slice(idx);

  auto f = [&](size_t i) {
    // putting write prefetch in slows it down
    //__builtin_prefetch (&out[idx[i+4]], 0, 1);
    //__sync_fetch_and_add(&out[idx[i]],1);};
    parlay::write_add(&out_slice[idx_slice[i]], 1);
  };

  for (auto _ : state) {
    if (n > 4) {
      parlay::parallel_for(0, n - 4, f);
    }
  }

  REPORT_STATS(n, 9 * sizeof(T), 8 * sizeof(T));
}

template<typename T>
static void bench_write_min(benchmark::State& state) {
  size_t n = state.range(0);
  parlay::random r(0);

  auto out = parlay::sequence<std::atomic<T>>(n);
  auto out_slice = parlay::make_slice(out);

  auto idx = parlay::tabulate(n, [&](size_t i) -> T { return r.ith_rand(i) % n; });
  auto idx_slice = parlay::make_slice(idx);

  auto f = [&](size_t i) {
    // putting write prefetch in slows it down
    //__builtin_prefetch (&out[idx[i+4]], 1, 1);
    parlay::write_min(&out_slice[idx_slice[i]], (T)i, std::less<T>());
  };

  for (auto _ : state) {
    if (n > 4) {
      parlay::parallel_for(0, n - 4, f);
    }
  }

  REPORT_STATS(n, 9 * sizeof(T), 8 * sizeof(T));
}

template<typename T>
static void bench_count_sort(benchmark::State& state) {
  size_t n = state.range(0);
  size_t bits = state.range(1);
  parlay::random r(0);
  size_t num_buckets = (1 << bits);
  size_t mask = num_buckets - 1;
  auto in = parlay::tabulate(n, [&](size_t i) -> T { return r.ith_rand(i); });
  auto get_key = [&](const T& t) { return t & mask; };
  auto keys = parlay::delayed_map(in, get_key);

  for (auto _ : state) {
    RUN_AND_CLEAR(parlay::internal::count_sort(parlay::make_slice(in), keys, num_buckets));
  }

  REPORT_STATS(n, 0, 0);
}

template<typename T>
static void bench_random_shuffle(benchmark::State& state) {
  size_t n = state.range(0);
  auto in = parlay::tabulate(n, [&](size_t i) -> T { return i; });

  for (auto _ : state) {
    RUN_AND_CLEAR(parlay::random_shuffle(in, n));
  }

  REPORT_STATS(n, 0, 0);
}

template<typename T>
static void bench_histogram(benchmark::State& state) {
  size_t n = state.range(0);
  parlay::random r(0);
  auto in = parlay::tabulate(n, [&](size_t i) -> T { return r.ith_rand(i) % n; });

  for (auto _ : state) {
    RUN_AND_CLEAR(parlay::histogram_by_index(in, (T)n));
  }

  REPORT_STATS(n, 0, 0);
}

template<typename T>
static void bench_histogram_same(benchmark::State& state) {
  size_t n = state.range(0);
  parlay::sequence<T> in(n, (T)10311);

  for (auto _ : state) {
    RUN_AND_CLEAR(parlay::histogram_by_index(in, (T)n));
  }

  REPORT_STATS(n, 0, 0);
}

template<typename T>
static void bench_histogram_few(benchmark::State& state) {
  size_t n = state.range(0);
  parlay::random r(0);
  auto in = parlay::tabulate(n, [&](size_t i) -> T { return r.ith_rand(i) % 256; });

  for (auto _ : state) {
    RUN_AND_CLEAR(parlay::histogram_by_index(in, (T)256));
  }

  REPORT_STATS(n, 0, 0);
}

template<typename T>
static void bench_integer_sort_inplace(benchmark::State& state) {
  // size_t n = state.range(0);
  using par = std::pair<T, T>;
  // parlay::random r(0);
  size_t bits = sizeof(T) * 8;
  // auto S = parlay::tabulate(n, [&](size_t i) -> par { return par(r.ith_rand(i), i); });
  auto first = [](par a) { return a.first; };
  size_t n = 1000000000;
  size_t testid = state.range(0);
  parlay::sequence<std::pair<uint64_t, uint64_t>> in(n);
  if (testid > 3 && testid < 19)
    test_uniform(uniform[testid], testid, in);
  else if (testid > 29 && testid < 36)
    zipfian_generator_int64_(zipfan[testid - 29], testid, in);
  else if (testid > 20 && testid < 28) {
    exponential_generator_int64_(1000000, exp_lambda[testid - 21], testid, in);
  } else {
    switch (testid) {
      case 38: testgraph("/data/zwang358/semisort/graph/com-orkut.txt", in); break;
      case 39: testgraph("/data/zwang358/semisort/graph/twitter.txt", in); break;
      case 40: testgraph("/data/zwang358/semisort/graph/yahoo_g9.txt", in); break;
      case 41: testgraph("/data/zwang358/semisort/graph/sd_arc.txt", in); break;
      case 42: testgraph("/data/zwang358/semisort/graph/enwiki.txt", in); break;
      case 43: testgraph("/data/zwang358/semisort/graph/webbase2001.txt", in); break;
      case 44: testgraph("/data/zwang358/semisort/graph/uk2002.txt", in); break;
      default: break;
    }
  }
  auto out = in;
  while (state.KeepRunningBatch(10)) {
    for (int i = 0; i < 10; i++) {
      COPY_NO_TIME(out, in);
      // parlay::internal::sample_sort_inplace(parlay::make_slice(out), std::less<T>());

      parlay::internal::integer_sort_inplace(parlay::make_slice(out),
                                             [](const auto& x) -> unsigned long { return x.first; });
    }
  }

  while (state.KeepRunningBatch(10)) {
    for (int i = 0; i < 10; i++) {
      RUN_AND_CLEAR(parlay::internal::integer_sort(parlay::make_slice(in), first, bits));
    }
  }

  REPORT_STATS(n, 0, 0);
}

template<typename T>
static void bench_integer_sort_pair(benchmark::State& state) {
  // size_t n = state.range(0);
  using par = std::pair<T, T>;
  // parlay::random r(0);
  size_t bits = sizeof(T) * 8;
  // auto S = parlay::tabulate(n, [&](size_t i) -> par { return par(r.ith_rand(i), i); });
  auto first = [](par a) { return a.first; };
  size_t n = 1000000000;
  size_t testid = state.range(0);
  parlay::sequence<std::pair<uint64_t, uint64_t>> in(n);
  if (testid > 3 && testid < 19)
    test_uniform(uniform[testid], testid, in);
  else if (testid > 29 && testid < 36)
    zipfian_generator_int64_(zipfan[testid - 29], testid, in);
  else if (testid > 20 && testid < 28) {
    exponential_generator_int64_(1000000, exp_lambda[testid - 21], testid, in);
  } else {
    switch (testid) {
      case 38: testgraph("/data/zwang358/semisort/graph/com-orkut.txt", in); break;
      case 39: testgraph("/data/zwang358/semisort/graph/twitter.txt", in); break;
      case 40: testgraph("/data/zwang358/semisort/graph/yahoo_g9.txt", in); break;
      case 41: testgraph("/data/zwang358/semisort/graph/sd_arc.txt", in); break;
      case 42: testgraph("/data/zwang358/semisort/graph/enwiki.txt", in); break;
      case 43: testgraph("/data/zwang358/semisort/graph/webbase2001.txt", in); break;
      case 44: testgraph("/data/zwang358/semisort/graph/uk2002.txt", in); break;
      default: break;
    }
  }
  while (state.KeepRunningBatch(10)) {
    for (int i = 0; i < 10; i++) {
      RUN_AND_CLEAR(parlay::internal::integer_sort(parlay::make_slice(in), first, bits));
    }
  }

  REPORT_STATS(n, 0, 0);
}

template<typename T>
static void bench_integer_sort(benchmark::State& state) {
  size_t n = state.range(0);
  parlay::random r(0);
  size_t bits = sizeof(T) * 8;
  auto S = parlay::tabulate(n, [&](size_t i) -> T { return r.ith_rand(i); });
  auto identity = [](T a) { return a; };

  while (state.KeepRunningBatch(10)) {
    for (int i = 0; i < 10; i++) {
      RUN_AND_CLEAR(parlay::internal::integer_sort(parlay::make_slice(S), identity, bits));
    }
  }

  REPORT_STATS(n, 0, 0);
}

template<typename T>
static void bench_integer_sort_128(benchmark::State& state) {
  size_t n = state.range(0);
  parlay::random r(0);
  size_t bits = parlay::log2_up(n);
  auto S = parlay::tabulate(
      n, [&](size_t i) -> __int128 { return r.ith_rand(2 * i) + (((__int128)r.ith_rand(2 * i + 1)) << 64); });
  auto identity = [](auto a) { return a; };

  while (state.KeepRunningBatch(10)) {
    for (int i = 0; i < 10; i++) {
      RUN_AND_CLEAR(parlay::internal::integer_sort(parlay::make_slice(S), identity, bits));
    }
  }

  REPORT_STATS(n, 0, 0);
}

template<typename T>
static void bench_stable_sort(benchmark::State& state) {
  // size_t n = state.range(0);
  // parlay::random r(0);
  // auto in = parlay::tabulate(n, [&](size_t i) -> T { return r.ith_rand(i) % n; });
  size_t n = 1000000000;
  size_t testid = state.range(0);
  parlay::sequence<std::pair<uint64_t, uint64_t>> in(n);
  if (testid > 3 && testid < 19)
    test_uniform(uniform[testid], testid, in);
  else if (testid > 29 && testid < 36)
    zipfian_generator_int64_(zipfan[testid - 29], testid, in);
  else if (testid > 20 && testid < 28) {
    exponential_generator_int64_(1000000, exp_lambda[testid - 21], testid, in);
  } else {
    switch (testid) {
      case 38: testgraph("/data/zwang358/semisort/graph/com-orkut.txt", in); break;
      case 39: testgraph("/data/zwang358/semisort/graph/twitter.txt", in); break;
      case 40: testgraph("/data/zwang358/semisort/graph/yahoo_g9.txt", in); break;
      case 41: testgraph("/data/zwang358/semisort/graph/sd_arc.txt", in); break;
      case 42: testgraph("/data/zwang358/semisort/graph/enwiki.txt", in); break;
      case 43: testgraph("/data/zwang358/semisort/graph/webbase2001.txt", in); break;
      case 44: testgraph("/data/zwang358/semisort/graph/uk2002.txt", in); break;
      default: break;
    }
  }
  auto less_pair = [&](std::pair<uint64_t, uint64_t> a, std::pair<uint64_t, uint64_t> b) { return a.first < b.first; };
  while (state.KeepRunningBatch(10)) {
    for (int i = 0; i < 10; i++) {
      RUN_AND_CLEAR(parlay::internal::sample_sort(parlay::make_slice(in), less_pair, true));
    }
  }

  REPORT_STATS(n, 0, 0);
}

template<typename T>
static void bench_unstable_sort(benchmark::State& state) {
  // size_t n = state.range(0);
  // parlay::random r(0);
  // auto in = parlay::tabulate(n, [&](size_t i) -> T { return r.ith_rand(i) % n; });
  size_t n = 1000000000;
  size_t testid = state.range(0);
  parlay::sequence<std::pair<uint64_t, uint64_t>> in(n);
  if (testid > 3 && testid < 19)
    test_uniform(uniform[testid], testid, in);
  else if (testid > 29 && testid < 36)
    zipfian_generator_int64_(zipfan[testid - 29], testid, in);
  else if (testid > 20 && testid < 28) {
    exponential_generator_int64_(1000000, exp_lambda[testid - 21], testid, in);
  } else {
    switch (testid) {
      case 38: testgraph("/data/zwang358/semisort/graph/com-orkut.txt", in); break;
      case 39: testgraph("/data/zwang358/semisort/graph/twitter.txt", in); break;
      case 40: testgraph("/data/zwang358/semisort/graph/yahoo_g9.txt", in); break;
      case 41: testgraph("/data/zwang358/semisort/graph/sd_arc.txt", in); break;
      case 42: testgraph("/data/zwang358/semisort/graph/enwiki.txt", in); break;
      case 43: testgraph("/data/zwang358/semisort/graph/webbase2001.txt", in); break;
      case 44: testgraph("/data/zwang358/semisort/graph/uk2002.txt", in); break;
      default: break;
    }
  }
  auto less_pair = [&](std::pair<uint64_t, uint64_t> a, std::pair<uint64_t, uint64_t> b) { return a.first < b.first; };
  while (state.KeepRunningBatch(10)) {
    for (int i = 0; i < 10; i++) {
      RUN_AND_CLEAR(parlay::internal::sample_sort(parlay::make_slice(in), less_pair, false));
    }
  }

  REPORT_STATS(n, 0, 0);
}

// template<>
// void bench_sort<parlay::sequence<char>>(benchmark::State& state) {
//   using T = parlay::sequence<char>;
//   size_t n = state.range(0);
//   ngram_table words;
//   auto in = parlay::tabulate(n, [&](size_t i) -> T { return words.word(i); });

//   while (state.KeepRunningBatch(10)) {
//     for (int i = 0; i < 10; i++) {
//       RUN_AND_CLEAR(parlay::internal::sample_sort(parlay::make_slice(in), std::less<T>()));
//     }
//   }

//   REPORT_STATS(n, 0, 0);
// }

template<typename T>
static void bench_sort_inplace(benchmark::State& state) {
  size_t n = 1000000000;
  size_t testid = state.range(0);
  parlay::sequence<std::pair<uint64_t, uint64_t>> in(n);
  if (testid > 3 && testid < 19)
    test_uniform(uniform[testid], testid, in);
  else if (testid > 29 && testid < 36)
    zipfian_generator_int64_(zipfan[testid - 29], testid, in);
  else if (testid > 20 && testid < 28) {
    exponential_generator_int64_(1000000, exp_lambda[testid - 21], testid, in);
  } else {
    switch (testid) {
      case 38: testgraph("/data/zwang358/semisort/graph/com-orkut.txt", in); break;
      case 39: testgraph("/data/zwang358/semisort/graph/twitter.txt", in); break;
      case 40: testgraph("/data/zwang358/semisort/graph/yahoo_g9.txt", in); break;
      case 41: testgraph("/data/zwang358/semisort/graph/sd_arc.txt", in); break;
      case 42: testgraph("/data/zwang358/semisort/graph/enwiki.txt", in); break;
      case 43: testgraph("/data/zwang358/semisort/graph/webbase2001.txt", in); break;
      case 44: testgraph("/data/zwang358/semisort/graph/uk2002.txt", in); break;
      default: break;
    }
  }
  auto less_pair = [&](std::pair<uint64_t, uint64_t> a, std::pair<uint64_t, uint64_t> b) { return a.first < b.first; };
  auto out = in;

  while (state.KeepRunningBatch(10)) {
    for (int i = 0; i < 10; i++) {
      COPY_NO_TIME(out, in);
      // parlay::internal::sample_sort_inplace(parlay::make_slice(out), std::less<T>());
      parlay::internal::sample_sort_inplace(parlay::make_slice(out), less_pair);
    }
  }

  REPORT_STATS(n, 0, 0);
}

template<typename T>
static void bench_merge(benchmark::State& state) {
  size_t n = state.range(0);
  auto in1 = parlay::tabulate(n / 2, [&](size_t i) -> T { return 2 * i; });
  auto in2 = parlay::tabulate(n - n / 2, [&](size_t i) -> T { return 2 * i + 1; });

  for (auto _ : state) {
    RUN_AND_CLEAR(parlay::merge(in1, in2, std::less<T>()));
  }

  REPORT_STATS(n, 2 * sizeof(T), sizeof(T));
}

template<typename T>
static void bench_merge_sort(benchmark::State& state) {
  size_t n = state.range(0);
  parlay::random r(0);
  auto in = parlay::tabulate(n, [&](size_t i) -> T { return r.ith_rand(i) % n; });
  auto out = in;

  while (state.KeepRunningBatch(10)) {
    for (int i = 0; i < 10; i++) {
      COPY_NO_TIME(out, in);
      parlay::internal::merge_sort_inplace(make_slice(out), std::less<T>());
    }
  }

  REPORT_STATS(n, 0, 0);
}

template<typename T>
static void bench_split3(benchmark::State& state) {
  size_t n = state.range(0);
  auto flags = parlay::tabulate(n, [](size_t i) -> unsigned char { return i % 3; });
  auto In = parlay::tabulate(n, [](size_t i) -> T { return i; });
  parlay::sequence<T> Out(n, 0);

  for (auto _ : state) {
    parlay::internal::split_three(make_slice(In), make_slice(Out), flags);
  }

  REPORT_STATS(n, 3 * sizeof(T), sizeof(T));
}

template<typename T>
static void bench_quicksort(benchmark::State& state) {
  size_t n = state.range(0);
  parlay::random r(0);
  auto in = parlay::tabulate(n, [&](size_t i) { return r.ith_rand(i) % n; });
  auto out = in;

  while (state.KeepRunningBatch(10)) {
    for (int i = 0; i < 10; i++) {
      COPY_NO_TIME(out, in);
      parlay::internal::p_quicksort_inplace(make_slice(out), std::less<T>());
    }
  }

  REPORT_STATS(n, 0, 0);
}

template<typename T>
static void bench_reduce_by_index_256(benchmark::State& state) {
  size_t n = state.range(0);
  using par = std::pair<T, T>;
  parlay::random r(0);
  size_t num_buckets = (1 << 8);
  auto S = parlay::tabulate(n, [&](size_t i) -> par { return par(r.ith_rand(i) % num_buckets, 1); });

  for (auto _ : state) {
    RUN_AND_CLEAR(parlay::reduce_by_index(S, num_buckets, parlay::addm<T>()));
  }

  REPORT_STATS(n, 0, 0);
}

template<typename T>
static void bench_reduce_by_index(benchmark::State& state) {
  size_t n = state.range(0);
  using par = std::pair<T, T>;
  parlay::random r(0);
  size_t num_buckets = n;
  auto S = parlay::tabulate(n, [&](size_t i) -> par { return par(r.ith_rand(i) % num_buckets, 1); });

  for (auto _ : state) {
    RUN_AND_CLEAR(parlay::reduce_by_index(S, num_buckets, parlay::addm<T>()));
  }

  REPORT_STATS(n, 0, 0);
}

template<typename T>
static void bench_remove_duplicate_integers(benchmark::State& state) {
  size_t n = state.range(0);
  parlay::random r(0);
  T num_buckets = n;
  auto S = parlay::tabulate(n, [&](size_t i) -> T { return r.ith_rand(i) % num_buckets; });

  for (auto _ : state) {
    RUN_AND_CLEAR(parlay::remove_duplicate_integers(S, num_buckets));
  }

  REPORT_STATS(n, 0, 0);
}

template<typename T>
static void bench_reduce_by_key(benchmark::State& state) {
  size_t n = state.range(0);
  using par = std::pair<T, T>;
  parlay::random r(0);
  auto S = parlay::tabulate(n, [&](size_t i) -> par { return par((T)r.ith_rand(i) % (n / 2), (T)1); });

  for (auto _ : state) {
    RUN_AND_CLEAR(parlay::reduce_by_key(S, parlay::addm<T>()));
  }

  REPORT_STATS(n, 0, 0);
}

template<typename T>
static void bench_histogram_by_key(benchmark::State& state) {
  size_t n = state.range(0);
  parlay::random r(0);
  auto S = parlay::tabulate(n, [&](size_t i) -> T { return r.ith_rand(i) % (n / 2); });

  for (auto _ : state) {
    RUN_AND_CLEAR(parlay::histogram_by_key<T>(S));
  }

  REPORT_STATS(n, 0, 0);
}

template<>
void bench_histogram_by_key<parlay::sequence<char>>(benchmark::State& state) {
  using T = parlay::sequence<char>;
  size_t n = state.range(0);
  ngram_table words;
  auto S = parlay::tabulate(n, [&](size_t i) { return words.word(i); });
  parlay::sequence<T> Tmp;
  for (auto _ : state) {
    COPY_NO_TIME(Tmp, S);
    RUN_AND_CLEAR(parlay::histogram_by_key(std::move(Tmp)));
  }

  REPORT_STATS(n, 0, 0);
}

template<typename T>
static void bench_remove_duplicates(benchmark::State& state) {
  size_t n = state.range(0);
  parlay::random r(0);
  auto S = parlay::tabulate(n, [&](size_t i) -> T { return r.ith_rand(i) % (n / 2); });

  for (auto _ : state) {
    RUN_AND_CLEAR(parlay::remove_duplicates(S));
  }

  REPORT_STATS(n, 0, 0);
}

template<>
void bench_remove_duplicates<parlay::sequence<char>>(benchmark::State& state) {
  using T = parlay::sequence<char>;
  size_t n = state.range(0);
  ngram_table words;
  auto S = parlay::tabulate(n, [&](size_t i) { return words.word(i); });
  parlay::sequence<T> Tmp;
  for (auto _ : state) {
    COPY_NO_TIME(Tmp, S);
    RUN_AND_CLEAR(parlay::remove_duplicates(std::move(Tmp)));
  }
  REPORT_STATS(n, 0, 0);
}

template<typename T>
static void bench_group_by_key(benchmark::State& state) {
  size_t n = 1000000000;
  size_t testid = state.range(0);
  parlay::sequence<std::pair<uint64_t, uint64_t>> S(n);
  if (testid > 3 && testid < 19)
    test_uniform(uniform[testid], testid, S);
  else if (testid > 29 && testid < 36)
    zipfian_generator_int64_(zipfan[testid - 29], testid, S);
  else if (testid > 20 && testid < 28) {
    exponential_generator_int64_(1000000, exp_lambda[testid - 21], testid, S);
  } else {
    switch (testid) {
      case 38: testgraph("/data/zwang358/semisort/graph/com-orkut.txt", S); break;
      case 39: testgraph("/data/zwang358/semisort/graph/twitter.txt", S); break;
      case 40: testgraph("/data/zwang358/semisort/graph/yahoo_g9.txt", S); break;
      case 41: testgraph("/data/zwang358/semisort/graph/sd_arc.txt", S); break;
      case 42: testgraph("/data/zwang358/semisort/graph/enwiki.txt", S); break;
      case 43: testgraph("/data/zwang358/semisort/graph/webbase2001.txt", S); break;
      case 44: testgraph("/data/zwang358/semisort/graph/uk2002.txt", S); break;
      default: break;
    }
  }
  parlay::sequence<mypair<uint64_t, uint64_t>> SS(n);
  parallel_for(0, n, [&](size_t i) {
    SS[i].first = S[i].first;
    SS[i].second = S[i].second;
  });
  while (state.KeepRunningBatch(10)) {
    for (int i = 0; i < 10; i++) {
      // RUN_AND_CLEAR(parlay::group_by_key(S));
      RUN_AND_CLEAR(parlay::group_by_key(parlay::make_slice(S)));
          // parlay::make_slice(SS), [&](mypair<uint64_t, uint64_t> a) { return a.first ; }));
    }
  }
  REPORT_STATS(n, 0, 0);
}

template<>
void bench_group_by_key<parlay::sequence<char>>(benchmark::State& state) {

  using T = parlay::sequence<char>;
  using par = std::pair<T, size_t>;
  size_t n = state.range(0);
  ngram_table words;
  auto S = parlay::tabulate(n, [&](size_t i) { return par(words.word(i), i); });
  parlay::sequence<par> Tmp;
  for (auto _ : state) {
    COPY_NO_TIME(Tmp, S);
    RUN_AND_CLEAR(parlay::group_by_key(std::move(Tmp)));
  }

  REPORT_STATS(n, 0, 0);
}

template<typename T>
static void bench_group_by_key_sorted(benchmark::State& state) {
  size_t n = 1000000000;
  size_t testid = state.range(0);
  // parlay::random r(0);
  // using par = std::pair<T,T>;
  // auto S = parlay::tabulate(n, [&] (size_t i) -> par {
  //     return par(r.ith_rand(i) % (n/20), i);});
  parlay::sequence<std::pair<uint64_t, uint64_t>> S(n);
  if (testid > 3 && testid < 19)
    test_uniform(uniform[testid], testid, S);
  else if (testid > 29 && testid < 36)
    zipfian_generator_int64_(zipfan[testid - 29], testid, S);
  else if (testid > 20 && testid < 28) {
    exponential_generator_int64_(1000000, exp_lambda[testid - 21], testid, S);
  } else {
    switch (testid) {
      case 38: testgraph("/data/zwang358/semisort/graph/com-orkut.txt", S); break;
      case 39: testgraph("/data/zwang358/semisort/graph/twitter.txt", S); break;
      case 40: testgraph("/data/zwang358/semisort/graph/yahoo_g9.txt", S); break;
      case 41: testgraph("/data/zwang358/semisort/graph/sd_arc.txt", S); break;
      case 42: testgraph("/data/zwang358/semisort/graph/enwiki.txt", S); break;
      case 43: testgraph("/data/zwang358/semisort/graph/webbase2001.txt", S); break;
      case 44: testgraph("/data/zwang358/semisort/graph/uk2002.txt", S); break;
      default: break;
    }
  }

  while (state.KeepRunningBatch(10)) {
    for (int i = 0; i < 10; i++) {
      RUN_AND_CLEAR(parlay::group_by_key_sorted(S));
    }
  }
  REPORT_STATS(n, 0, 0);
}

template<typename T>
static void bench_group_by_index(benchmark::State& state) {
  size_t n = state.range(0);
  parlay::random r(0);
  using par = std::pair<T, T>;
  T num_buckets = (n / 20);
  auto S = parlay::tabulate(n, [&](size_t i) -> par { return par(r.ith_rand(i) % num_buckets, i); });

  for (auto _ : state) {
    RUN_AND_CLEAR(parlay::group_by_index(S, num_buckets));
  }

  REPORT_STATS(n, 0, 0);
}

template<typename T>
static void bench_group_by_index_256(benchmark::State& state) {
  size_t n = state.range(0);
  parlay::random r(0);
  using par = std::pair<T, T>;
  T num_buckets = 256;
  auto S = parlay::tabulate(n, [&](size_t i) -> par { return par(r.ith_rand(i) % num_buckets, i); });

  for (auto _ : state) {
    RUN_AND_CLEAR(parlay::group_by_index(S, num_buckets));
  }

  REPORT_STATS(n, 0, 0);
}

// ------------------------- Registration -------------------------------

#define BENCH(NAME, T, args...)                                                                                        \
  BENCHMARK_TEMPLATE(bench_##NAME, T)->UseRealTime()->Unit(benchmark::kMillisecond)->Args({args});

// BENCH(map, long, 100000000);
// BENCH(tabulate, long, 100000000);
// BENCH(reduce_add, long, 100000000);
// BENCH(scan_add, long, 100000000);
// BENCH(pack, long, 100000000);
// BENCH(gather, long, 100000000);
// BENCH(scatter, long, 100000000);
// BENCH(scatter, int, 100000000);
// BENCH(write_add, long, 100000000);
// BENCH(write_min, long, 100000000);
// BENCH(count_sort, long, 100000000, 4);
// BENCH(count_sort, long, 100000000, 8);
// BENCH(integer_sort, unsigned long, 100000000);
// BENCH(integer_sort_pair, unsigned long, 4);
// BENCH(integer_sort_pair, unsigned long, 5);
// BENCH(integer_sort_pair, unsigned long, 6);
// BENCH(integer_sort_pair, unsigned long, 7);
// BENCH(integer_sort_pair, unsigned long, 8);
// BENCH(integer_sort_pair, unsigned long, 9);
// BENCH(integer_sort_pair, unsigned long, 10);
// BENCH(integer_sort_pair, unsigned long, 11);
// BENCH(integer_sort_pair, unsigned long, 12);
// BENCH(integer_sort_pair, unsigned long, 13);
// BENCH(integer_sort_pair, unsigned long, 14);
// BENCH(integer_sort_pair, unsigned long, 15);
// BENCH(integer_sort_pair, unsigned long, 16);
// BENCH(integer_sort_pair, unsigned long, 17);
// BENCH(integer_sort_pair, unsigned long, 18);
// BENCH(integer_sort_pair, unsigned long, 21);
// BENCH(integer_sort_pair, unsigned long, 22);
// BENCH(integer_sort_pair, unsigned long, 23);
// BENCH(integer_sort_pair, unsigned long, 24);
// BENCH(integer_sort_pair, unsigned long, 25);
// BENCH(integer_sort_pair, unsigned long, 26);
// BENCH(integer_sort_pair, unsigned long, 27);
// BENCH(integer_sort_pair, unsigned long, 30);
// BENCH(integer_sort_pair, unsigned long, 31);
// BENCH(integer_sort_pair, unsigned long, 32);
// BENCH(integer_sort_pair, unsigned long, 33);
// BENCH(integer_sort_pair, unsigned long, 34);
// BENCH(integer_sort_pair, unsigned long, 35);
// BENCH(integer_sort_pair, unsigned long, 38);
// BENCH(integer_sort_pair, unsigned long, 39);
// BENCH(integer_sort_pair, unsigned long, 40);
// BENCH(integer_sort_pair, unsigned long, 41);
// BENCH(integer_sort_pair, unsigned long, 42);
// BENCH(integer_sort_pair, unsigned long, 43);

// // BENCH(integer_sort_128, __int128, 100000000);
// BENCH(unstable_sort, unsigned long, 4);
// BENCH(unstable_sort, unsigned long, 5);
// BENCH(unstable_sort, unsigned long, 6);
// BENCH(unstable_sort, unsigned long, 7);
// BENCH(unstable_sort, unsigned long, 8);
// BENCH(unstable_sort, unsigned long, 9);
// BENCH(unstable_sort, unsigned long, 10);
// BENCH(unstable_sort, unsigned long, 11);
// BENCH(unstable_sort, unsigned long, 12);
// BENCH(unstable_sort, unsigned long, 13);
// BENCH(unstable_sort, unsigned long, 14);
// BENCH(unstable_sort, unsigned long, 15);
// BENCH(unstable_sort, unsigned long, 16);
// BENCH(unstable_sort, unsigned long, 17);
// BENCH(unstable_sort, unsigned long, 18);
// BENCH(unstable_sort, unsigned long, 21);
// BENCH(unstable_sort, unsigned long, 22);
// BENCH(unstable_sort, unsigned long, 23);
// BENCH(unstable_sort, unsigned long, 24);
// BENCH(unstable_sort, unsigned long, 25);
// BENCH(unstable_sort, unsigned long, 26);
// BENCH(unstable_sort, unsigned long, 27);
// BENCH(unstable_sort, unsigned long, 30);
// BENCH(unstable_sort, unsigned long, 31);
// BENCH(unstable_sort, unsigned long, 32);
// BENCH(unstable_sort, unsigned long, 33);
// BENCH(unstable_sort, unsigned long, 34);
// BENCH(unstable_sort, unsigned long, 35);
// BENCH(unstable_sort, unsigned long, 38);
// BENCH(unstable_sort, unsigned long, 39);
// BENCH(unstable_sort, unsigned long, 40);
// BENCH(unstable_sort, unsigned long, 41);
// BENCH(unstable_sort, unsigned long, 42);
// BENCH(unstable_sort, unsigned long, 43);
// BENCH(unstable_sort, unsigned long, 44);

// BENCH(stable_sort, unsigned long, 4);
// BENCH(stable_sort, unsigned long, 5);
// BENCH(stable_sort, unsigned long, 6);
// BENCH(stable_sort, unsigned long, 7);
// BENCH(stable_sort, unsigned long, 8);
// BENCH(stable_sort, unsigned long, 9);
// BENCH(stable_sort, unsigned long, 10);
// BENCH(stable_sort, unsigned long, 11);
// BENCH(stable_sort, unsigned long, 12);
// BENCH(stable_sort, unsigned long, 13);
// BENCH(stable_sort, unsigned long, 14);
// BENCH(stable_sort, unsigned long, 15);
// BENCH(stable_sort, unsigned long, 16);
// BENCH(stable_sort, unsigned long, 17);
// BENCH(stable_sort, unsigned long, 18);
// BENCH(stable_sort, unsigned long, 21);
// BENCH(stable_sort, unsigned long, 22);
// BENCH(stable_sort, unsigned long, 23);
// BENCH(stable_sort, unsigned long, 24);
// BENCH(stable_sort, unsigned long, 25);
// BENCH(stable_sort, unsigned long, 26);
// BENCH(stable_sort, unsigned long, 27);
// BENCH(stable_sort, unsigned long, 30);
// BENCH(stable_sort, unsigned long, 31);
// BENCH(stable_sort, unsigned long, 32);
// BENCH(stable_sort, unsigned long, 33);
// BENCH(stable_sort, unsigned long, 34);
// BENCH(stable_sort, unsigned long, 35);
// BENCH(stable_sort, unsigned long, 38);
// BENCH(stable_sort, unsigned long, 39);
// BENCH(stable_sort, unsigned long, 40);
// BENCH(stable_sort, unsigned long, 41);
// BENCH(stable_sort, unsigned long, 42);
// BENCH(stable_sort, unsigned long, 43);
// BENCH(stable_sort, unsigned long, 44);
// BENCH(sort, long, 100000000);
// BENCH(sort, __int128, 100000000);
// BENCH(sort, parlay::sequence<char>, 100000000);
// BENCH(sort_inplace, unsigned long, 4);
// BENCH(sort_inplace, unsigned long, 5);
// BENCH(sort_inplace, unsigned long, 6);
// BENCH(sort_inplace, unsigned long, 7);
// BENCH(sort_inplace, unsigned long, 8);
// BENCH(sort_inplace, unsigned long, 9);
// BENCH(sort_inplace, unsigned long, 10);
// BENCH(sort_inplace, unsigned long, 11);
// BENCH(sort_inplace, unsigned long, 12);
// BENCH(sort_inplace, unsigned long, 13);
// BENCH(sort_inplace, unsigned long, 14);
// BENCH(sort_inplace, unsigned long, 15);
// BENCH(sort_inplace, unsigned long, 16);
// BENCH(sort_inplace, unsigned long, 17);
// BENCH(sort_inplace, unsigned long, 18);
// BENCH(sort_inplace, unsigned long, 21);
// BENCH(sort_inplace, unsigned long, 22);
// BENCH(sort_inplace, unsigned long, 23);
// BENCH(sort_inplace, unsigned long, 24);
// BENCH(sort_inplace, unsigned long, 25);
// BENCH(sort_inplace, unsigned long, 26);
// BENCH(sort_inplace, unsigned long, 27);
// BENCH(sort_inplace, unsigned long, 30);
// BENCH(sort_inplace, unsigned long, 31);
// BENCH(sort_inplace, unsigned long, 32);
// BENCH(sort_inplace, unsigned long, 33);
// BENCH(sort_inplace, unsigned long, 34);
// BENCH(sort_inplace, unsigned long, 35);
// BENCH(sort_inplace, unsigned long, 38);
// BENCH(sort_inplace, unsigned long, 39);
// BENCH(sort_inplace, unsigned long, 40);
// BENCH(sort_inplace, unsigned long, 41);
// BENCH(sort_inplace, unsigned long, 42);
// BENCH(sort_inplace, unsigned long, 43);
// BENCH(sort_inplace, unsigned long, 44);
// BENCH(sort_inplace, long, 100000000);
// BENCH(sort_inplace, __int128, 100000000);
// BENCH(merge, long, 100000000);
// BENCH(merge_sort, long, 100000000);
// BENCH(quicksort, long, 100000000);
// BENCH(random_shuffle, long, 100000000);
// BENCH(histogram, unsigned int, 100000000);
// BENCH(histogram_same, unsigned int, 100000000);
// BENCH(histogram_few, unsigned int, 100000000);
// BENCH(reduce_by_index_256, unsigned int, 100000000);
// BENCH(reduce_by_index, unsigned int, 100000000);
// BENCH(remove_duplicate_integers, unsigned int, 100000000);
// BENCH(group_by_index_256, unsigned int, 100000000);
// BENCH(group_by_index, unsigned int, 100000000);
// BENCH(reduce_by_key, unsigned long, 100000000);
// BENCH(histogram_by_key, unsigned long, 100000000);
// BENCH(remove_duplicates, unsigned long, 100000000);
// BENCH(group_by_key, unsigned long, 4);
// BENCH(group_by_key, unsigned long, 5);
// BENCH(group_by_key, unsigned long, 6);
// BENCH(group_by_key, unsigned long, 7);
// BENCH(group_by_key, unsigned long, 8);
// BENCH(group_by_key, unsigned long, 9);
// BENCH(group_by_key, unsigned long, 10);
// BENCH(group_by_key, unsigned long, 11);
// BENCH(group_by_key, unsigned long, 12);
// BENCH(group_by_key, unsigned long, 13);
// BENCH(group_by_key, unsigned long, 14);
// BENCH(group_by_key, unsigned long, 15);
// BENCH(group_by_key, unsigned long, 16);
// BENCH(group_by_key, unsigned long, 17);
// BENCH(group_by_key, unsigned long, 18);
// BENCH(group_by_key, unsigned long, 21);
// BENCH(group_by_key, unsigned long, 22);
// BENCH(group_by_key, unsigned long, 23);
// BENCH(group_by_key, unsigned long, 24);
// BENCH(group_by_key, unsigned long, 25);
// BENCH(group_by_key, unsigned long, 26);
// BENCH(group_by_key, unsigned long, 27);
BENCH(group_by_key, unsigned long, 30);
BENCH(group_by_key, unsigned long, 31);
BENCH(group_by_key, unsigned long, 32);
BENCH(group_by_key, unsigned long, 33);
BENCH(group_by_key, unsigned long, 34);
BENCH(group_by_key, unsigned long, 35);
// BENCH(group_by_key, unsigned long, 38);
// BENCH(group_by_key, unsigned long, 39);
// BENCH(group_by_key, unsigned long, 40);
// BENCH(group_by_key, unsigned long, 41);
// BENCH(group_by_key, unsigned long, 42);
// BENCH(group_by_key, unsigned long, 43);
// BENCH(group_by_key, unsigned long, 44);
// BENCH(group_by_key_sorted, unsigned long)->Args(4)
//                                         ->Args(5);
// BENCH(group_by_key_sorted, unsigned long, 4);
// BENCH(group_by_key_sorted, unsigned long, 5);
// BENCH(group_by_key_sorted, unsigned long, 6);
// BENCH(group_by_key_sorted, unsigned long, 7);
// BENCH(group_by_key_sorted, unsigned long, 8);
// BENCH(group_by_key_sorted, unsigned long, 9);
// BENCH(group_by_key_sorted, unsigned long, 10);
// BENCH(group_by_key_sorted, unsigned long, 11);
// BENCH(group_by_key_sorted, unsigned long, 12);
// BENCH(group_by_key_sorted, unsigned long, 13);
// BENCH(group_by_key_sorted, unsigned long, 14);
// BENCH(group_by_key_sorted, unsigned long, 15);
// BENCH(group_by_key_sorted, unsigned long, 16);
// BENCH(group_by_key_sorted, unsigned long, 17);
// BENCH(group_by_key_sorted, unsigned long, 18);
BENCH(group_by_key_sorted, unsigned long, 21);
// BENCH(group_by_key_sorted, unsigned long, 22);
// BENCH(group_by_key_sorted, unsigned long, 23);
// BENCH(group_by_key_sorted, unsigned long, 24);
// BENCH(group_by_key_sorted, unsigned long, 25);
// BENCH(group_by_key_sorted, unsigned long, 26);
// BENCH(group_by_key_sorted, unsigned long, 27);
// BENCH(group_by_key_sorted, unsigned long, 30);
// BENCH(group_by_key_sorted, unsigned long, 31);
// BENCH(group_by_key_sorted, unsigned long, 32);
// BENCH(group_by_key_sorted, unsigned long, 33);
// BENCH(group_by_key_sorted, unsigned long, 34);
// BENCH(group_by_key_sorted, unsigned long, 35);
// BENCH(group_by_key_sorted, unsigned long, 38);
// BENCH(group_by_key_sorted, unsigned long, 39);
// BENCH(group_by_key_sorted, unsigned long, 40);
// BENCH(group_by_key_sorted, unsigned long, 41);
// BENCH(group_by_key_sorted, unsigned long, 42);
// BENCH(group_by_key_sorted, unsigned long, 43);
// BENCH(group_by_key_sorted, unsigned long, 44);
// BENCH(group_by_key_sorted, unsigned long, 30);
// BENCH(group_by_key_sorted, unsigned long, 37);
// BENCH(histogram_by_key, parlay::sequence<char>, 100000000);
// BENCH(remove_duplicates, parlay::sequence<char>, 100000000);
// BENCH(group_by_key, parlay::sequence<char>, 100000000);

BENCH(integer_sort_inplace, unsigned long, 4);
BENCH(integer_sort_inplace, unsigned long, 5);
BENCH(integer_sort_inplace, unsigned long, 6);
BENCH(integer_sort_inplace, unsigned long, 7);
BENCH(integer_sort_inplace, unsigned long, 8);
BENCH(integer_sort_inplace, unsigned long, 9);
BENCH(integer_sort_inplace, unsigned long, 10);
BENCH(integer_sort_inplace, unsigned long, 11);
BENCH(integer_sort_inplace, unsigned long, 12);
BENCH(integer_sort_inplace, unsigned long, 13);
BENCH(integer_sort_inplace, unsigned long, 14);
BENCH(integer_sort_inplace, unsigned long, 15);
BENCH(integer_sort_inplace, unsigned long, 16);
BENCH(integer_sort_inplace, unsigned long, 17);
BENCH(integer_sort_inplace, unsigned long, 18);
BENCH(integer_sort_inplace, unsigned long, 21);
BENCH(integer_sort_inplace, unsigned long, 22);
BENCH(integer_sort_inplace, unsigned long, 23);
BENCH(integer_sort_inplace, unsigned long, 24);
BENCH(integer_sort_inplace, unsigned long, 25);
BENCH(integer_sort_inplace, unsigned long, 26);
BENCH(integer_sort_inplace, unsigned long, 27);
BENCH(integer_sort_inplace, unsigned long, 30);
BENCH(integer_sort_inplace, unsigned long, 31);
BENCH(integer_sort_inplace, unsigned long, 32);
BENCH(integer_sort_inplace, unsigned long, 33);
BENCH(integer_sort_inplace, unsigned long, 34);
BENCH(integer_sort_inplace, unsigned long, 35);