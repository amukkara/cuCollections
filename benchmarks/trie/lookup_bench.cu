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

#include <defaults.hpp>
#include <utils.hpp>

#include "../../tests/trie/trie_utils.hpp"
#include <cuco/trie.cuh>
#include <cuco/utility/key_generator.hpp>

#include <nvbench/nvbench.cuh>

using namespace cuco::benchmark;
using namespace cuco::utility;

/**
 * @brief A benchmark evaluating `cuco::experimental::trie::lookup` performance
 */
void trie_lookup(nvbench::state& state)
{
  auto const num_keys       = 64 * 1024;
  auto const max_key_length = 6;

  using LabelType = int;
  cuco::experimental::trie<LabelType> trie;

  thrust::host_vector<LabelType> labels;
  thrust::host_vector<size_t> offsets;

  generate_labels(labels, offsets, num_keys, max_key_length);
  auto keys = sorted_keys(labels, offsets);
  for (auto key : keys) {
    trie.insert(key.begin(), key.end());
  }
  trie.build();

  const size_t query_size = num_keys / 10;
  thrust::device_vector<LabelType> inputs(labels.begin(), labels.begin() + offsets[query_size]);
  thrust::device_vector<size_t> d_offsets(offsets.begin(), offsets.begin() + query_size);
  thrust::device_vector<size_t> outputs(query_size);

  state.add_element_count(query_size);
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    trie.lookup(inputs.begin(), d_offsets.begin(), d_offsets.end(), outputs.begin());
  });
}

NVBENCH_BENCH(trie_lookup).set_name("trie_lookup").set_max_noise(defaults::MAX_NOISE);