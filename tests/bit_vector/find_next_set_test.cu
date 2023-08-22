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

#include <cuco/detail/trie/bit_vector/bit_vector.cuh>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>

#include <catch2/catch_test_macros.hpp>

template <class BitVectorRef, typename size_type>
__global__ void find_next_set_kernel(BitVectorRef ref, size_type n, size_type* output)
{
  size_t index  = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = gridDim.x * blockDim.x;
  while (index < n) {
    output[index] = ref.find_next_set(index);
    index += stride;
  }
}

extern bool modulo_bitgen(uint64_t i);

TEST_CASE("Find next set test", "")
{
  cuco::experimental::bit_vector bv;

  using size_type = cuco::experimental::bit_vector<>::size_type;
  constexpr size_type num_elements{400};

  for (size_type i = 0; i < num_elements; i++) {
    bv.append(modulo_bitgen(i));
  }
  bv.build();

  thrust::device_vector<size_type> device_result(num_elements);
  auto ref = bv.ref(cuco::experimental::bv_read);
  find_next_set_kernel<<<1, 1024>>>(
    ref, num_elements, thrust::raw_pointer_cast(device_result.data()));

  thrust::host_vector<size_type> host_result = device_result;
  size_type num_matches                      = 0;

  size_type next_set_pos = -1lu;
  do {
    next_set_pos++;
  } while (next_set_pos < num_elements and !modulo_bitgen(next_set_pos));

  for (size_type key = 0; key < num_elements; key++) {
    num_matches += host_result[key] == next_set_pos;

    if (key == next_set_pos) {
      do {
        next_set_pos++;
      } while (next_set_pos < num_elements and !modulo_bitgen(next_set_pos));
    }
  }
  REQUIRE(num_matches == num_elements);
}
