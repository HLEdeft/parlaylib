#include "gtest/gtest.h"

#include <algorithm>
#include <cstddef>
#include <deque>
#include <fstream>
#include <iostream>
#include <numeric>
#include <parlay/internal/group_by.h>
#include <parlay/primitives.h>
#include <parlay/random.h>
#include <parlay/sequence.h>
#include <parlay/slice.h>
#include <parlay/type_traits.h>
#include <string>

#include <parlay/internal/sample_sort.h>

#include "sorting_utils.h"
using parlay::parallel_for;
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
    {1000000000, "zipfan_1000000000"}};
void scan_inplace_(uint32_t *in, uint32_t n) {
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
        nums[i] = (uint32_t)((double)n * (exp(-exp_lambda * i) * (1 - exp(-exp_lambda))));
      },
      1);
  uint32_t offset = parlay::reduce(
      nums, parlay::addm<uint32_t>()); // cout << "offset = " << offset << endl;
  nums[0] += (n - offset);
  uint32_t *addr = new uint32_t[exp_cutoff];
  parlay::parallel_for(
      0, exp_cutoff, [&](uint32_t i) { addr[i] = nums[i]; }, 1);
  scan_inplace_(addr, exp_cutoff); // store all addresses into addr[]

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
  scan_inplace_(addr, zipf_s); // store all addresses into addr[]
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
void exponential_generator_int64_yunshu(uint32_t n, int exp_cutoff, double exp_lambda,parlay::sequence<std::pair<uint64_t, uint64_t>> &C) {
  parlay::sequence<uint32_t> nums(exp_cutoff);
  parlay::sequence<std::pair<uint64_t, uint64_t>> B(n);
  // double base = 1 - exp(-1);
  // cout << "1 - e^-1 = " << base << endl;
  // cout << "e^-2 = " << exp(-2) << endl;

  /* 1. making nums[] array */
  parallel_for(
      0, exp_cutoff,
      [&](int i) {
        nums[i] = (uint32_t)((double)n * (exp(-exp_lambda * i) * (1 - exp(-exp_lambda))));
      },
      1);

  uint32_t offset = parlay::reduce(nums, parlay::addm<uint32_t>());
  nums[0] += (n - offset);
  // cout << "offset/n = " << offset << "/" << n << endl;
  // checking if the sum of nums[] equals to n
  if (parlay::reduce(nums, parlay::addm<uint32_t>()) == (uint32_t)n) {
    std::cout << "sum of nums[] == n" << std::endl;
  }

  /* 2. scan to calculate position */
  uint32_t *addr = new uint32_t[exp_cutoff];
  parallel_for(
      0, exp_cutoff, [&](uint32_t i) { addr[i] = nums[i]; }, 1);
  scan_inplace_(addr, exp_cutoff); // store all addresses into addr[]

  /* 3. distribute random numbers into A[i].first */
  parallel_for(
      0, exp_cutoff,
      [&](size_t i) {
        size_t st = (i == 0) ? 0 : addr[i - 1],
               ed = (i == (uint32_t)exp_cutoff - 1) ? n : addr[i];
        for (size_t j = st; j < ed; j++) {
          B[j].first = parlay::hash64_2(i);
        }
      },
      1);
  parallel_for(
      0, n, [&](size_t i) { B[i].second = parlay::hash64_2(i); }, 1);

  /* 4. shuffle the keys */
  C = parlay::random_shuffle(B, n);

  // parallel_for(0, n, [&](size_t i) { A[i] = C[i]; });

  delete[] addr;
}
TEST(TestSemiSort, TestExp) {
  uint32_t n = 1000000000;
  parlay::sequence<std::pair<uint64_t, uint64_t>> A(n);
  parlay::sequence<std::pair<uint64_t, uint64_t>> B(n);
  exponential_generator_int64_(1000000, testcase[15].first / 100000.0, A);
  exponential_generator_int64_yunshu(n,1000000, testcase[15].first / 100000.0, B);
  std::cout << A[1].first << "  " << A[1].second << std::endl;
  std::cout << A[n/10].first << "  " << A[n/10].second << std::endl;
  std::cout << A[n/5].first << "  " << A[n/5].second << std::endl;
  std::cout << A[n/2].first << "  " << A[n/2].second << std::endl;
  std::cout << B[1].first << "  " << B[1].second << std::endl;
  std::cout << B[n/10].first << "  " << B[n/10].second << std::endl;
  std::cout << B[n/5].first << "  " << B[n/5].second << std::endl;
  std::cout << B[n/2].first << "  " << B[n/2].second << std::endl;
  ASSERT_EQ(A, B);
}
TEST(TestSemiSort, TestParlaySort) {
  for (size_t i = 0; i < testcase.size(); i++) {
    // for (size_t i = 20; i < 21; i++) {
    std::ifstream foutput;
    std::ifstream finput;
    size_t n = 1000000000;
    parlay::sequence<uint64_t> In(n / 10);
    parlay::sequence<uint64_t> Out(n / 10);
    finput.open("/data/zwang358/verification/" + testcase[i].second + ".in",
                std::ios::in | std::ios::binary);
    foutput.open("/data/zwang358/verification/" + testcase[i].second + ".out",
                 std::ios::in | std::ios::binary);
    size_t j = 0;
    while (finput.read((char *)&In[j], sizeof(In[j])))
      j++;
    j = 0;
    while (foutput.read((char *)&Out[j], sizeof(Out[j])))
      j++;
    // size_t j = 0;
    // while (finput.read((char *)&In[j], sizeof(In[j]))) { //一直读到文件结束
    //   j += 10;
    // }
    // j = 0;
    // while (foutput.read((char *)&Out[j], sizeof(Out[j]))) {
    // //一直读到文件结束
    //   j += 10;
    // }
    parlay::sequence<std::pair<uint64_t, uint64_t>> A(n);
    if (i < 15)
      test_uniform(testcase[i].first, A);
    else if (i < 22)
      exponential_generator_int64_(1000000, testcase[i].first / 100000.0, A);
    else if (i < 28)
      zipfian_generator_int64_(testcase[i].first, A);

    auto in =
        parlay::tabulate(n / 10, [&](uint32_t i) { return A[static_cast<size_t>(i ) * 10].first; });
    // for (size_t j = 0; j < n; j += 10)
    //   ASSERT_EQ(A[j].first, In[j]);
    ASSERT_EQ(in, In);
    std::cout << "start testing " << testcase[i].second << std::endl;
    // Modify Here
    {
      // Copy from test_primitives.cpp
      std::cout << "integer_sort_pair\n";
      auto s = parlay::tabulate(n, [&](uint32_t i) -> UnstablePair {
        UnstablePair x;
        x.x = A[i].first;
        x.y = A[i].second;
        return x;
      });
      size_t bits = sizeof(uint64_t) * 8;
      auto first = [](const auto &x) -> uint64_t { return x.x; };
      auto sorted =
          parlay::internal::integer_sort(parlay::make_slice(s), first, bits);
      // Assert Here
      auto out =
          parlay::tabulate(n / 10, [&](size_t j) { return sorted[j * 10].x; });
      ASSERT_EQ(out, Out);
      // std::cout << "Success!\n";
    }
    {
      std::cout << "integer_sort_inplace_pair\n";
      auto s = parlay::tabulate(n, [&](uint32_t i) -> UnstablePair {
        UnstablePair x;
        x.x = A[i].first;
        x.y = A[i].second;
        return x;
      });
      size_t bits = sizeof(uint64_t) * 8;
      auto first = [](const auto &x) -> uint64_t { return x.x; };
      parlay::internal::integer_sort_inplace(parlay::make_slice(s), first,
                                             bits);
      auto out =
          parlay::tabulate(n / 10, [&](size_t j) { return s[j * 10].x; });
      ASSERT_EQ(out, Out);
      // std::cout << "Success! \n";
    }
    {
      std::cout << "Sample_Sort_Stable\n";
      auto s = parlay::tabulate(n, [&](uint32_t i) -> UnstablePair {
        UnstablePair x;
        x.x = A[i].first;
        x.y = A[i].second;
        return x;
      });
      auto sorted = parlay::internal::sample_sort(
          parlay::make_slice(s), std::less<UnstablePair>(), true);
      auto out =
          parlay::tabulate(n / 10, [&](size_t j) { return sorted[j * 10].x; });
      ASSERT_EQ(out, Out);
      // std::cout << "Success! \n";
    }
    {
      std::cout << "Sample_Sort_Unstable\n";
      auto s = parlay::tabulate(n, [&](uint32_t i) -> UnstablePair {
        UnstablePair x;
        x.x = A[i].first;
        x.y = A[i].second;
        return x;
      });
      auto sorted = parlay::internal::sample_sort(
          parlay::make_slice(s), std::less<UnstablePair>(), false);
      auto out =
          parlay::tabulate(n / 10, [&](size_t j) { return sorted[j * 10].x; });
      ASSERT_EQ(out, Out);
      // std::cout << "Success! \n";
    }
    {
      std::cout << "Sample_Sort_Unstable_Inplace\n";
      auto s = parlay::tabulate(n, [&](uint32_t i) -> UnstablePair {
        UnstablePair x;
        x.x = A[i].first;
        x.y = A[i].second;
        return x;
      });
      parlay::internal::sample_sort_inplace(parlay::make_slice(s),
                                            std::less<UnstablePair>());
      auto out =
          parlay::tabulate(n / 10, [&](size_t j) { return s[j * 10].x; });
      ASSERT_EQ(out, Out);
      // std::cout << "Success! \n";
    }
    // {
    //   std::cout << "Group_By_Key_Sorted\n";
    //   auto s = parlay::tabulate(n, [&](uint32_t i) -> UnstablePair {
    //     UnstablePair x;
    //     x.x = A[i].first;
    //     x.y = A[i].second;
    //     return x;
    //   });
    //   auto key_vals = parlay::map(s,[&](auto x){ return std::make_pair(x.x,
    //   x.y);});
    //   // auto key_vals = parlay::map(s,[&](auto x){ UnstablePair pair; pair.x
    //   = x.x; pair.y = x.y; return pair;}); auto sorted =
    //   parlay::group_by_key_sorted(key_vals);
    // }
    foutput.close();
    finput.close();
  }
}

// TEST(TestSampleSort, TestStableSortInplacePair) {
//   auto s = parlay::tabulate(100000, [](int i) -> UnstablePair {
//     UnstablePair x;
//     x.x = (53 * i + 61) % (1 << 10);
//     x.y = i;
//     return x;
//   });
//   auto sorted = s;
//   parlay::internal::sample_sort_inplace(parlay::make_slice(sorted),
//                                         std::greater<UnstablePair>());
//   ASSERT_EQ(s.size(), sorted.size());
//   std::stable_sort(std::rbegin(s), std::rend(s), std::less<UnstablePair>());
//   ASSERT_EQ(s, sorted);
//   ASSERT_TRUE(std::is_sorted(std::rbegin(sorted), std::rend(sorted)));
// }

// TEST(TestSampleSort, TestSort) {
//   auto s = parlay::tabulate(100000, [](long long i) -> long long {
//     return (50021 * i + 61) % (1 << 20);
//   });
//   auto sorted = parlay::internal::sample_sort(parlay::make_slice(s),
//                                               std::less<long long>());
//   ASSERT_EQ(s.size(), sorted.size());
//   std::sort(std::begin(s), std::end(s));
//   ASSERT_EQ(s, sorted);
//   ASSERT_TRUE(std::is_sorted(std::begin(sorted), std::end(sorted)));
// }

// TEST(TestSampleSort, TestSortCustomCompare) {
//   auto s = parlay::tabulate(100000, [](long long i) -> long long {
//     return (50021 * i + 61) % (1 << 20);
//   });
//   auto sorted = parlay::internal::sample_sort(parlay::make_slice(s),
//                                               std::greater<long long>());
//   ASSERT_EQ(s.size(), sorted.size());
//   std::sort(std::rbegin(s), std::rend(s));
//   ASSERT_EQ(s, sorted);
//   ASSERT_TRUE(std::is_sorted(std::rbegin(sorted), std::rend(sorted)));
// }

// TEST(TestSampleSort, TestStableSort) {
//   auto s = parlay::tabulate(100000, [](int i) -> UnstablePair {
//     UnstablePair x;
//     x.x = (53 * i + 61) % (1 << 10);
//     x.y = i;
//     return x;
//   });
//   auto sorted = parlay::internal::sample_sort(parlay::make_slice(s),
//                                               std::less<UnstablePair>(),
//                                               true);
//   ASSERT_EQ(s.size(), sorted.size());
//   std::stable_sort(std::begin(s), std::end(s));
//   ASSERT_EQ(s, sorted);
//   ASSERT_TRUE(std::is_sorted(std::begin(sorted), std::end(sorted)));
// }

// TEST(TestSampleSort, TestStableSortCustomCompare) {
//   auto s = parlay::tabulate(100000, [](int i) -> UnstablePair {
//     UnstablePair x;
//     x.x = (53 * i + 61) % (1 << 10);
//     x.y = i;
//     return x;
//   });
//   auto sorted = parlay::internal::sample_sort(
//       parlay::make_slice(s), std::greater<UnstablePair>(), true);
//   ASSERT_EQ(s.size(), sorted.size());
//   std::stable_sort(std::rbegin(s), std::rend(s));
//   ASSERT_EQ(s, sorted);
//   ASSERT_TRUE(std::is_sorted(std::rbegin(sorted), std::rend(sorted)));
// }

// TEST(TestSampleSort, TestSortInplace) {
//   auto s = parlay::tabulate(100000, [](long long i) -> long long {
//     return (50021 * i + 61) % (1 << 20);
//   });
//   auto s2 = s;
//   ASSERT_EQ(s, s2);
//   parlay::internal::sample_sort_inplace(parlay::make_slice(s),
//                                         std::less<long long>());
//   std::sort(std::begin(s2), std::end(s2));
//   ASSERT_EQ(s, s2);
//   ASSERT_TRUE(std::is_sorted(std::begin(s), std::end(s)));
// }

// TEST(TestSampleSort, TestSortInplaceCustomCompare) {
//   auto s = parlay::tabulate(100000, [](long long i) -> long long {
//     return (50021 * i + 61) % (1 << 20);
//   });
//   auto s2 = s;
//   ASSERT_EQ(s, s2);
//   parlay::internal::sample_sort_inplace(parlay::make_slice(s),
//                                         std::greater<long long>());
//   std::sort(std::rbegin(s2), std::rend(s2));
//   ASSERT_EQ(s, s2);
//   ASSERT_TRUE(std::is_sorted(std::rbegin(s), std::rend(s)));
// }

// TEST(TestSampleSort, TestSortInplaceUncopyable) {
//   auto s = parlay::tabulate(
//       100000, [](int i) -> UncopyableThing { return UncopyableThing(i); });
//   auto s2 = parlay::tabulate(
//       100000, [](int i) -> UncopyableThing { return UncopyableThing(i); });
//   ASSERT_EQ(s, s2);
//   parlay::internal::sample_sort_inplace(parlay::make_slice(s),
//                                         std::less<UncopyableThing>());
//   std::stable_sort(std::begin(s2), std::end(s2));
//   ASSERT_EQ(s, s2);
//   ASSERT_TRUE(std::is_sorted(std::begin(s), std::end(s)));
// }

// namespace parlay {
// // Specialize std::unique_ptr to be considered trivially relocatable
// template <typename T>
// struct is_trivially_relocatable<std::unique_ptr<T>> : public std::true_type
// {}; } // namespace parlay

// TEST(TestSampleSort, TestSortInplaceUniquePtr) {
//   auto s = parlay::tabulate(100000, [](long long int i) {
//     return std::make_unique<long long int>((50021 * i + 61) % (1 << 20));
//   });
//   auto s2 = parlay::tabulate(100000, [](long long int i) {
//     return std::make_unique<long long int>((50021 * i + 61) % (1 << 20));
//   });
//   parlay::internal::sample_sort_inplace(
//       parlay::make_slice(s),
//       [](const auto &a, const auto &b) { return *a < *b; });
//   std::stable_sort(std::begin(s2), std::end(s2),
//                    [](const auto &a, const auto &b) { return *a < *b; });
//   ASSERT_TRUE(
//       std::is_sorted(std::begin(s), std::end(s),
//                      [](const auto &a, const auto &b) { return *a < *b; }));
//   for (size_t i = 0; i < 100000; i++) {
//     ASSERT_EQ(*s[i], *s2[i]);
//   }
// }

// TEST(TestSampleSort, TestSortInplaceNonContiguous) {
//   auto ss = parlay::tabulate(100000, [](long long i) -> long long {
//     return (50021 * i + 61) % (1 << 20);
//   });
//   auto s = std::deque<long long>(ss.begin(), ss.end());
//   auto s2 = s;
//   ASSERT_EQ(s, s2);
//   parlay::internal::sample_sort_inplace(parlay::make_slice(s),
//                                         std::less<long long>());
//   std::sort(std::begin(s2), std::end(s2));
//   ASSERT_EQ(s, s2);
//   ASSERT_TRUE(std::is_sorted(std::begin(s), std::end(s)));
// }
