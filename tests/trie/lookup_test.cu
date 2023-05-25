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

TEST_CASE("Lookup test", "")
{
  using KeyType = int;
  cuco::experimental::trie<KeyType> trie;

  std::size_t num_keys              = 3;
  thrust::host_vector<KeyType> keys = std::vector<KeyType>{1, 2, 3, 1, 2, 4, 1, 4, 2};

  // Last length is 0 to simplify subsequent scan
  thrust::host_vector<uint64_t> lengths = std::vector<uint64_t>{3, 3, 3, 0};
  thrust::host_vector<uint64_t> offsets = lengths;
  thrust::exclusive_scan(offsets.begin(), offsets.end(), offsets.begin());  // in-place scan

  for (size_t key_id = 0; key_id < num_keys; key_id++) {
    std::vector<KeyType> cur_key;
    for (size_t pos = offsets[key_id]; pos < offsets[key_id + 1]; pos++) {
      cur_key.push_back(keys[pos]);
    }
    trie.add(cur_key);
  }
  trie.build();

  {
    thrust::device_vector<uint64_t> lookup_result(num_keys, -1lu);
    thrust::device_vector<KeyType> device_keys     = keys;
    thrust::device_vector<uint64_t> device_offsets = offsets;

    trie.lookup(
      device_keys.begin(), device_keys.end(), device_offsets.begin(), lookup_result.begin());

    thrust::host_vector<uint64_t> host_lookup_result = lookup_result;
    for (size_t key_id = 0; key_id < num_keys; key_id++) {
      REQUIRE(host_lookup_result[key_id] == key_id);
    }
  }
}
