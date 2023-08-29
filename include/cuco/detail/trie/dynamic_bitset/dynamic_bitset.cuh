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

#pragma once

#include <cuco/cuda_stream_ref.hpp>

#include <thrust/device_malloc_allocator.h>
#include <thrust/device_vector.h>

#include <cuda/std/array>

#include <climits>
#include <cstddef>

namespace cuco {
namespace experimental {
namespace detail {

/**
 * @brief Struct to store ranks of bits at 256-bit intervals
 */
struct rank {
  uint32_t abs_hi_;                    ///< Upper 32 bits of base
  uint8_t abs_lo_;                     ///< Lower 8 bits of base
  cuda::std::array<uint8_t, 3> rels_;  ///< Offsets for 64-bit sub-intervals

  /**
   * @brief Gets base rank of current 256-bit interval
   *
   * @return The base rank
   */
  __host__ __device__ uint64_t constexpr abs() const noexcept
  {
    return (static_cast<uint64_t>(abs_hi_) << 8) | abs_lo_;
  }

  /**
   * @brief Sets base rank of current 256-bit interval
   *
   * @param abs Base rank
   */
  __host__ __device__ void set_abs(uint64_t abs) noexcept
  {
    abs_hi_ = static_cast<uint32_t>(abs >> 8);
    abs_lo_ = static_cast<uint8_t>(abs);
  }
};

/**
 * @brief Bitvector class with rank and select index structures
 *
 * In addition to standard bitvector get/set operations, this class provides
 * rank and select operation API. It maintains index structures to make both these
 * new operations close to constant time.
 * Bitvector construction happens on host, after which the structures are moved to device.
 * All subsequent read-only operations access device structures only.
 *
 * @tparam Allocator Type of allocator used for device storage
 */
template <class Allocator = thrust::device_malloc_allocator<std::byte>>
class dynamic_bitset {
 public:
  using size_type = std::size_t;  ///< size type to specify bit index
  using slot_type = uint64_t;     ///< Slot type
  /// Type of the allocator to (de)allocate words
  using allocator_type = typename std::allocator_traits<Allocator>::rebind_alloc<slot_type>;

  static constexpr size_type words_per_block = 4;  ///< Tradeoff between space efficiency and perf.
  static constexpr size_type bits_per_word   = sizeof(slot_type) * CHAR_BIT;     ///< Bits in a word
  static constexpr size_type bits_per_block  = words_per_block * bits_per_word;  ///< Trivial

  /**
   * @brief Constructs an empty bitvector
   *
   * @param allocator Allocator used for allocating device storage
   */
  inline dynamic_bitset(Allocator const& allocator = Allocator{});
  dynamic_bitset(dynamic_bitset&&) = default;  ///< Move constructor
  inline ~dynamic_bitset();

  /**
   * @brief adds a new bit at the end
   *
   * Grows internal storage if needed
   *
   * @param bit Boolean value of new bit to be added
   */
  inline void append(bool bit) noexcept;

  /**
   * @brief Modifies a single bit
   *
   * @param index position of bit to be modified
   * @param bit new value of bit
   */
  inline void set(size_type index, bool bit) noexcept;

  /**
   * @brief Sets last bit to specified value
   *
   * @param bit new value of last bit
   */
  inline void set_last(bool bit) noexcept;

  /**
   * @brief Builds indexes for rank and select
   */
  inline void build() noexcept;

  /**
   * @brief Bulk get operation
   *
   * @tparam KeyIt Device-accessible iterator to keys
   * @tparam OutputIt Device-accessible iterator to outputs
   *
   * @param keys_begin Begin iterator to keys list whose values are queried
   * @param keys_end End iterator to keys list
   * @param outputs_begin Begin iterator to outputs of get operation
   * @param stream Stream to execute get kernel
   */
  template <typename KeyIt, typename OutputIt>
  void get(KeyIt keys_begin,
           KeyIt keys_end,
           OutputIt outputs_begin,
           cuda_stream_ref stream = {}) const noexcept;

  /**
   * @brief Bulk rank operation
   *
   * @tparam KeyIt Device-accessible iterator to keys
   * @tparam OutputIt Device-accessible iterator to output ranks
   *
   * @param keys_begin Begin iterator to keys list whose ranks are queried
   * @param keys_end End iterator to keys list
   * @param outputs_begin Begin iterator to outputs ranks list
   * @param stream Stream to execute ranks kernel
   */
  template <typename KeyIt, typename OutputIt>
  void ranks(KeyIt keys_begin,
             KeyIt keys_end,
             OutputIt outputs_begin,
             cuda_stream_ref stream = {}) const noexcept;

  /**
   * @brief Bulk select operation
   *
   * @tparam KeyIt Device-accessible iterator to keys
   * @tparam OutputIt Device-accessible iterator to outputs
   *
   * @param keys_begin Begin iterator to keys list whose select values are queried
   * @param keys_end End iterator to keys list
   * @param outputs_begin Begin iterator to outputs selects list
   * @param stream Stream to execute selects kernel
   */
  template <typename KeyIt, typename OutputIt>
  void selects(KeyIt keys_begin,
               KeyIt keys_end,
               OutputIt outputs_begin,
               cuda_stream_ref stream = {}) const noexcept;

  /**
   *@brief Struct to hold all storage refs needed by bitvector_ref
   */
  struct storage_ref_type {
    const slot_type* words_ref_;  ///< Words refs

    const rank* ranks_ref_;         ///< Ranks refs
    const size_type* selects_ref_;  ///< Selects refs

    const rank* ranks0_ref_;         ///< Ranks refs for 0 bits
    const size_type* selects0_ref_;  ///< Selects refs 0 bits
  };

  /**
   * @brief Device non-owning reference type of dynamic_bitset
   */
  class reference {
   public:
    /**
￼   * @brief Constructs dynamic_bitset_ref.
￼   *
￼   * @param storage Struct with non-owning refs to bitvector slot storages
￼   */
    __host__ __device__ explicit constexpr reference(storage_ref_type storage) noexcept;

    /**
     * @brief Access value of a single bit
     *
     * @param key Position of bit
     *
     * @return Value of bit at position specified by key
     */
    [[nodiscard]] __device__ bool get(size_type key) const noexcept;

    /**
     * @brief Access a single word of internal storage
     *
     * @param word_id Index of word
     *
     * @return Word at position specified by index
     */
    [[nodiscard]] __device__ slot_type get_word(size_type word_id) const noexcept;

    /**
     * @brief Find position of first set bit starting from a given position (inclusive)
     *
     * @param key Position of starting bit
     *
     * @return Index of next set bit
     */
    [[nodiscard]] __device__ size_type find_next_set(size_type key) const noexcept;

    /**
     * @brief Find number of set bits (rank) in all positions before the input position (exclusive)
     *
     * @param key Input bit position
     *
     * @return Rank of input position
     */
    [[nodiscard]] __device__ size_type rank(size_type key) const noexcept;

    /**
     * @brief Find position of Nth set (1) bit counting from start of bitvector
     *
     * @param count Input N
     *
     * @return Position of Nth set bit
     */
    [[nodiscard]] __device__ size_type select(size_type count) const noexcept;

    /**
     * @brief Find position of Nth not-set (0) bit counting from start of bitvector
     *
     * @param count Input N
     *
     * @return Position of Nth not-set bit
     */
    [[nodiscard]] __device__ size_type select0(size_type count) const noexcept;

   private:
    /**
     * @brief Helper function for select operation that computes an initial rank estimate
     *
     * @param count Input count for which select operation is being performed
     * @param selects Selects array
     * @param ranks Ranks array
     *
     * @return index in ranks which corresponds to highest rank less than count (least upper bound)
     */
    template <typename SelectsRef, typename RanksRef>
    [[nodiscard]] __device__ size_type get_initial_rank_estimate(
      size_type count, const SelectsRef& selects, const RanksRef& ranks) const noexcept;

    /**
     * @brief Subtract rank estimate from input count and return an increment to word_id
     *
     * @tparam Rank type
     *
     * @param count Input count that will be updated
     * @param rank  Initial rank estimate for count
     *
     * @return Increment to word_id based on rank values
     */
    template <typename Rank>
    [[nodiscard]] __device__ size_type subtract_rank_from_count(size_type& count,
                                                                Rank rank) const noexcept;

    /**
     * @brief Find position of Nth set bit in a 64-bit word
     *
     * @param N Input count
     *
     * @return Position of Nth set bit
     */
    [[nodiscard]] __device__ size_type select_bit_in_word(size_type N,
                                                          slot_type word) const noexcept;

    storage_ref_type storage_;  ///< Non-owning storage
  };

  using ref_type = reference;  ///< Non-owning container ref type

  /**
   * @brief Gets non-owning device ref of the current object
   *
   * @return Device ref of the current `dynamic_bitset` object
   */
  [[nodiscard]] ref_type ref() const noexcept;

  /**
   * @brief Gets the number of bits dynamic_bitset holds
   *
   * @return Number of bits dynamic_bitset holds
   */
  [[nodiscard]] constexpr size_type size() const noexcept;

 private:
  /// Type of the allocator to (de)allocate ranks
  using rank_allocator_type = typename std::allocator_traits<Allocator>::rebind_alloc<rank>;
  /// Type of the allocator to (de)allocate indices
  using size_allocator_type = typename std::allocator_traits<Allocator>::rebind_alloc<size_type>;

  allocator_type allocator_;  ///< Words allocator
  size_type n_bits_;          ///< Number of bits dynamic_bitset currently holds

  /// Words vector that represents all bits
  thrust::device_vector<slot_type, allocator_type> words_;
  /// Rank values for every 256-th bit (4-th word)
  thrust::device_vector<rank, rank_allocator_type> ranks_;
  /// Same as ranks_ but for `0` bits
  thrust::device_vector<rank, rank_allocator_type> ranks0_;
  /// Block indices of (0, 256, 512...)th `1` bit
  thrust::device_vector<size_type, size_allocator_type> selects_;
  /// Same as selects_, but for `0` bits
  thrust::device_vector<size_type, size_allocator_type> selects0_;

  /**
   * @brief Populates rank and select indexes on device
   *
   * @param ranks Output array of ranks
   * @param selects Output array of selects
   * @param flip_bits If true, negate bits to construct indexes for `0` bits
   */
  void build_ranks_and_selects(thrust::device_vector<rank, rank_allocator_type>& ranks,
                               thrust::device_vector<size_type, size_allocator_type>& selects,
                               bool flip_bits) noexcept;

  /**
   * @brief Helper function to calculate grid size for simple kernels
   *
   * @param num_elements Elements being processed by kernel
   *
   * @return grid size
   */
  // TODO: to be moved to the CUDA utility header
  size_type constexpr default_grid_size(size_type num_elements) const noexcept;
};

}  // namespace detail
}  // namespace experimental
}  // namespace cuco

#include <cuco/detail/trie/dynamic_bitset/dynamic_bitset.inl>