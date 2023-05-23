/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <thrust/device_vector.h>

#include <cassert>
#include <iostream>
#include <queue>
#include <vector>

#include <cuco/bit_vector.cuh>

namespace cuco {
namespace experimental {

template <typename T>
class trie {
 public:
  trie();
  ~trie() noexcept(false);
  void add(const std::vector<T>& key);
  void build();

  void lookup(const T* queries, const uint64_t* offsets, uint64_t* ids, uint64_t num_queries) const;

  uint64_t n_keys() const { return n_keys_; }
  uint64_t memory_footprint() const { return footprint_; }

  struct Level {
    bit_vector<> louds;
    bit_vector<> outs;
    std::vector<T> labels;
    thrust::device_vector<T> d_labels;
    T* d_labels_ptr;
    uint64_t offset;

    Level()
      : louds(cuco::experimental::extent<std::size_t>{1000}),
        outs(cuco::experimental::extent<std::size_t>{1000}),
        labels(),
        offset(0)
    {
    }
    uint64_t memory_footprint() const
    {
      return louds.size() + outs.size() + sizeof(T) * labels.size();
    }
  };

 public:
  Level* d_levels_ptr_;

  static constexpr auto cg_size     = 1;
  static constexpr auto window_size = 1;
  using Extent                      = cuco::experimental::extent<std::size_t>;
  using Storage                     = cuco::experimental::aow_storage<1>;
  using value_type                  = uint64_t;
  using extent_type    = decltype(make_valid_extent<cg_size, window_size>(std::declval<Extent>()));
  using allocator_type = cuco::cuda_allocator<std::byte>;

  using storage_type     = detail::storage<Storage, value_type, extent_type, allocator_type>;
  using storage_ref_type = typename storage_type::ref_type;

  bit_vector_ref<storage_ref_type, bv_read_tag>* d_louds_refs_ptr_;
  bit_vector_ref<storage_ref_type, bv_read_tag>* d_outs_refs_ptr_;

 private:
  uint64_t num_levels_;
  std::vector<Level> levels_;

  uint64_t n_keys_;
  uint64_t n_nodes_;
  uint64_t footprint_;
  std::vector<T> last_key_;

  trie<T>* device_impl_;

  thrust::device_vector<bit_vector_ref<storage_ref_type, bv_read_tag>> d_louds_refs_, d_outs_refs_;
};

}  // namespace experimental
}  // namespace cuco

#include <cuco/detail/trie/trie.inl>
