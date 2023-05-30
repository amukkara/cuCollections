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

#include <cuco/bit_vector.cuh>

#include <catch2/catch_test_macros.hpp>

TEST_CASE("Size computation", "")
{
  constexpr std::size_t num_elements{400};

  cuco::experimental::bit_vector bv;

  for (size_t i = 0; i < num_elements; i++) {
    bv.append(i % 2 == 0);  // Alternate 0s and 1s pattern
  }
  bv.build();

  auto const size = bv.size();
  REQUIRE(size == num_elements);
}
