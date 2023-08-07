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

#include <cuco/cuda_stream_ref.hpp>
#include <cuco/detail/trie/bit_vector/bit_vector.cuh>
#include <cuco/trie_ref.cuh>

#include <thrust/device_vector.h>

#include <cassert>
#include <iostream>
#include <queue>
#include <vector>

namespace cuco {
namespace experimental {

/**
 * @brief Trie class
 *
 * @tparam T type of individual characters of vector keys (eg. char or int)
 */
template <typename T>
class trie {
 public:
  trie();
  ~trie() noexcept(false);

  /**
   * @brief Insert new key into trie
   *
   * @param key Key to insert
   */
  void insert(const std::vector<T>& key);

  /**
   * @brief Build level-by-level trie indexes after inserting all keys
   *
   * In addition, a snapshot of current trie state is copied to device
   */
  void build();

  /**
   * @brief Bulk lookup vector of keys
   *
   * @tparam KeyIt Device-accessible iterator to individual characters of keys
   * @tparam OffsetIt Device-accessible iterator to positions of key boundaries
   * @tparam OutputIt Device-accessible iterator to lookup result
   *
   * @param keys_begin Begin iterator to individual key characters
   * @param offsets_begin Begin iterator to offsets of key boundaries
   * @param offsets_end End iterator to offsets
   * @param outputs_begin Begin iterator to results
   * @param stream Stream to execute lookup kernel
   */
  template <typename KeyIt, typename OffsetIt, typename OutputIt>
  void lookup(KeyIt keys_begin,
              OffsetIt offsets_begin,
              OffsetIt offsets_end,
              OutputIt outputs_begin,
              cuda_stream_ref stream = {}) const;

  /**
   * @brief Get number of keys inserted into trie
   *
   * @return Number of keys
   */
  uint64_t n_keys() const { return n_keys_; }

  uint64_t n_nodes() const { return n_nodes_; }

  using value_type     = uint64_t;  ///< bit_vector value type
  using extent_type    = decltype(make_valid_extent<1, 1>(
    std::declval<cuco::experimental::extent<std::size_t>>()));  ///< Extent type
  using allocator_type = cuco::cuda_allocator<std::byte>;          ///< Allocator type

  using storage_type = detail::storage<cuco::experimental::aow_storage<1>,
                                       value_type,
                                       extent_type,
                                       allocator_type>;  ///< bit_vector storage type
  using storage_ref_type =
    typename storage_type::ref_type;  ///< Non-owing bit_vector storage ref type

  template <typename... Operators>
  using ref_type =
    cuco::experimental::trie_ref<T, Operators...>;  ///< Non-owning container ref type

  /**
   * @brief Get device ref with operators.
   *
   * @tparam Operators Set of `cuco::op` to be provided by the ref
   *
   * @param ops List of operators, e.g., `cuco::bv_read`
   *
   * @return Device ref of the current `trie` object
   */
  template <typename... Operators>
  [[nodiscard]] auto ref(Operators... ops) const noexcept;

  bit_vector_ref<storage_ref_type, bv_read_tag>*
    d_louds_refs_ptr_;  ///< Array of refs of louds bitvectors of each level
  bit_vector_ref<storage_ref_type, bv_read_tag>*
    d_outs_refs_ptr_;  ///< Array of refs of out bitvectors of each level

  /**
   * @brief Struct to represent each trie level
   */
  struct level {
    level();

    bit_vector<> louds;  ///< Indicates links to next and previous level
    bit_vector<> outs;   ///< Indicates terminal nodes of valid keys

    std::vector<T> labels;              ///< Stores individual characters of keys
    thrust::device_vector<T> d_labels;  ///< Device-side copy of `labels`
    T* d_labels_ptr;                    ///< Raw pointer to d_labels

    uint64_t offset;  ///< Count of nodes in all parent levels
  };

  level* d_levels_ptr_;  ///< Device-side array of levels

 private:
  uint64_t num_levels_;        ///< Number of trie levels
  std::vector<level> levels_;  ///< Host-side array of levels

  uint64_t n_keys_;          ///< Number of keys inserted into trie
  uint64_t n_nodes_;         ///< Number of nodes in trie
  std::vector<T> last_key_;  ///< Last key inserted into trie

  trie<T>* device_ptr_;  ///< Device-side copy of trie structure

  thrust::device_vector<bit_vector_ref<storage_ref_type, bv_read_tag>> d_louds_refs_, d_outs_refs_;
};

}  // namespace experimental
}  // namespace cuco

#include <cuco/detail/trie/trie.inl>