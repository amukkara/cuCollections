#include <cuco/trie_ref.cuh>

namespace cuco {
namespace experimental {

template <typename T, typename... Operators>
__host__ __device__ constexpr trie_ref<T, Operators...>::trie_ref(const trie<T>* trie) noexcept
  : trie_(trie)
{
}

namespace detail {

template <typename T, typename... Operators>
class operator_impl<op::trie_lookup_tag, trie_ref<T, Operators...>> {
  using ref_type = trie_ref<T, Operators...>;

 public:
  template <typename KeyIt>
  [[nodiscard]] __device__ uint64_t lookup_key(KeyIt key, uint64_t length) const noexcept
  {
    auto const& ref_ = static_cast<ref_type const&>(*this);

    uint32_t node_id = 0;
    for (uint32_t cur_depth = 1; cur_depth <= length; cur_depth++) {
      if (!binary_search_labels_array((T)key[cur_depth - 1], node_id, cur_depth)) { return -1lu; }
    }

    uint64_t leaf_level_id = length;
    if (!ref_.trie_->d_outs_refs_ptr_[leaf_level_id].get(node_id)) { return -1lu; }

    auto offset = ref_.trie_->d_levels_ptr_[leaf_level_id].offset;
    auto rank   = ref_.trie_->d_outs_refs_ptr_[leaf_level_id].rank(node_id);
    return offset + rank;
  }

  template <typename BitVectorRef>
  [[nodiscard]] __device__ uint32_t find_end_pos(BitVectorRef louds_ref,
                                                 uint32_t& node_id) const noexcept
  {
    uint32_t node_pos = 0;
    if (node_id != 0) {
      node_pos = louds_ref.select(node_id - 1) + 1;
      node_id  = node_pos - node_id;
    }
    uint32_t pos_end = louds_ref.find_next_set(node_pos);
    uint32_t end     = node_id + (pos_end - node_pos);
    return end;
  }

  [[nodiscard]] __device__ bool binary_search_labels_array(T target,
                                                           uint32_t& node_id,
                                                           uint32_t level_id) const noexcept
  {
    auto const& ref_ = static_cast<ref_type const&>(*this);
    auto louds_ref   = ref_.trie_->d_louds_refs_ptr_[level_id];

    uint32_t end   = find_end_pos(louds_ref, node_id);
    uint32_t begin = node_id;  // Do not move this before find_end_pos call

    const auto& level = ref_.trie_->d_levels_ptr_[level_id];
    while (begin < end) {
      node_id    = (begin + end) / 2;
      auto label = level.d_labels_ptr[node_id];
      if (target < label) {
        end = node_id;
      } else if (target > label) {
        begin = node_id + 1;
      } else {
        break;
      }
    }
    return begin < end;
  }
};

}  // namespace detail
}  // namespace experimental
}  // namespace cuco
