#include "gtest/gtest.h"

#include <deque>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include <iostream>
#include <parlay/internal/group_by.h>

#include "sorting_utils.h"

class TestGroupByP : public testing::TestWithParam<size_t> {};

TEST(TestGroupBy, TestHistogramByKey) {
  auto s =
      parlay::tabulate(357453, [](unsigned long long i) -> unsigned long long { return (243571 * i + 61) % (1 << 20); });
  size_t num_buckets = 19;
  std::map<unsigned long long, parlay::sequence<unsigned long long>> results;
  std::vector<std::vector<unsigned long long>> input(num_buckets);
  // std::map<unsigned long long, parlay::sequence<unsigned long long>> input;

  auto key_vals = parlay::map(s, [num_buckets](auto x) { return std::make_pair(x % num_buckets, (50021 * x + 61) % (1 << 20)); });

  for (const auto& result : key_vals) {
    input[result.first].push_back(result.second);
  }
  auto grouped = parlay::group_by_key(key_vals);
  //   std::stable_sort(std::begin(key_vals), std::end(key_vals));
  ASSERT_EQ(grouped.size(), num_buckets);

  for (const auto& result : grouped) {
    results.insert(result);
  }

  for (int i = 0; i < num_buckets; i++) {
    for (int j = 0; j < results[i].size(); j++) {
      ASSERT_EQ(results[i][j], input[i][j]);
    }
  }
}