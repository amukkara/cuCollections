#pragma once

#include <cuco/operator.hpp>

namespace cuco {
namespace experimental {

template <typename T>
class trie;

template <typename T, typename... Operators>
class trie_ref : public detail::operator_impl<Operators, trie_ref<T, Operators...>>... {
 public:
  /**
   * @brief Constructs trie_ref.
   *
   */
  __host__ __device__ explicit constexpr trie_ref(const trie<T>*) noexcept;

 private:
  const trie<T>* trie_;

  // Mixins need to be friends with this class in order to access private members
  template <typename Op, typename Ref>
  friend class detail::operator_impl;
};

}  // namespace experimental
}  // namespace cuco

#include <cuco/detail/trie/trie_ref.inl>
