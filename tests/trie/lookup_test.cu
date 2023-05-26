/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <utils.hpp>

#include <cuco/trie.cuh>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>

#include <catch2/catch_test_macros.hpp>

template <typename KeyType>
void generate_keys(thrust::host_vector<KeyType>& keys,
                   thrust::host_vector<uint64_t>& offsets,
                   size_t num_keys,
                   size_t max_key_value,
                   size_t max_key_length)
{
  for (size_t key_id = 0; key_id < num_keys; key_id++) {
    size_t cur_key_length = 1 + (std::rand() % max_key_length);
    offsets.push_back(cur_key_length);
    for (size_t pos = 0; pos < cur_key_length; pos++) {
      keys.push_back(1 + (std::rand() % max_key_value));
    }
  }

  // Add a dummy 0 to simplify subsequent scan
  offsets.push_back(0);
  thrust::exclusive_scan(offsets.begin(), offsets.end(), offsets.begin());  // in-place scan
}

TEST_CASE("Lookup test", "")
{
  using KeyType = int;
  cuco::experimental::trie<KeyType> trie;

  std::size_t num_keys = 64 * 1024;
  thrust::host_vector<KeyType> keys;
  thrust::host_vector<uint64_t> offsets;

  generate_keys(keys, offsets, num_keys, 1000, 32);

  {
    std::vector<std::vector<KeyType>> all_keys;
    for (size_t key_id = 0; key_id < num_keys; key_id++) {
      std::vector<KeyType> cur_key;
      for (size_t pos = offsets[key_id]; pos < offsets[key_id + 1]; pos++) {
        cur_key.push_back(keys[pos]);
      }
      all_keys.push_back(cur_key);
    }

    struct vectorKeyCompare {
      bool operator()(const std::vector<KeyType>& lhs, const std::vector<KeyType>& rhs)
      {
        for (size_t pos = 0; pos < min(lhs.size(), rhs.size()); pos++) {
          if (lhs[pos] < rhs[pos]) {
            return true;
          } else if (lhs[pos] > rhs[pos]) {
            return false;
          }
        }
        return lhs.size() <= rhs.size();
      }
    };
    sort(all_keys.begin(), all_keys.end(), vectorKeyCompare());

    for (auto key : all_keys) {
      trie.add(key);
    }
  }

  trie.build();

  {
    thrust::device_vector<uint64_t> lookup_result(num_keys, -1lu);
    thrust::device_vector<KeyType> device_keys     = keys;
    thrust::device_vector<uint64_t> device_offsets = offsets;

    trie.lookup(
      device_keys.begin(), device_offsets.begin(), device_offsets.end(), lookup_result.begin());

    thrust::host_vector<uint64_t> host_lookup_result = lookup_result;
    for (size_t key_id = 0; key_id < num_keys; key_id++) {
      REQUIRE(host_lookup_result[key_id] < num_keys);
    }
  }
}
