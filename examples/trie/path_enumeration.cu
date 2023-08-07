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

#include "cub/block/block_run_length_decode.cuh"
#include "cub/cub.cuh"
#include "input_utils.hpp"
#include <cuco/trie.cuh>
#include <cuco/trie_ref.cuh>
#include <limits>

const uint32_t FIND_PATH_BLOCK_SIZE = 1024;

const uint32_t MAX_PATH_BUFFER_SIZE = 100 * 1000;  // 8192
const uint32_t MAX_FRONTIER_SIZE    = 100 * 1000;

const uint32_t TOPK_KEYS_PER_LEVEL = 256;
const bool CHECK_FIND_PATHS_RESULT = false;

const uint32_t num_topk_id = 100;
const uint32_t max_depth   = 16;
const uint32_t max_paths   = 100;

struct Edge {
  uint32_t node_id;
  float score;
};

struct Path {
  uint32_t node_id;
  uint32_t level_id;
};

const float score_sentinel       = -std::numeric_limits<float>::max();
const float node_score_threshold = -11;

struct State {
  uint32_t node_id;
  float score;
};

template <typename T>
class PathEnumeration {
 public:
  PathEnumeration();
  ~PathEnumeration() noexcept(false);

  void find_paths(const cuco::experimental::trie<T>* trie,
                  const uint32_t* keys,
                  const float* scores,
                  uint32_t max_depth,
                  uint32_t max_paths,
                  uint32_t stream_id);
  void sync_streams();

 private:
  uint32_t num_levels_;

  const uint32_t num_streams = 1;
  std::vector<cudaStream_t> streams;

  State** frontiers;
  State** next_frontiers;

  uint32_t* num_paths_outs;

  Path** path_buffers;
  float** score_buffers;

  T** path_values;
  uint32_t** path_offsets;

  void** sort_paths_temp_storage;
  size_t sort_paths_temp_storage_bytes;

 private:
  void sort_paths(uint32_t stream_id) const;
  void check_result(uint32_t stream_id, uint32_t max_paths) const;
};

template <typename T>
PathEnumeration<T>::PathEnumeration()
{
  CUCO_CUDA_TRY(cudaMalloc(&num_paths_outs, sizeof(uint32_t) * num_streams));

  frontiers      = (State**)malloc(sizeof(State*) * num_streams);
  next_frontiers = (State**)malloc(sizeof(State*) * num_streams);

  path_buffers  = (Path**)malloc(sizeof(Path*) * 2 * num_streams);
  score_buffers = (float**)malloc(sizeof(float*) * 2 * num_streams);

  path_values  = (T**)malloc(sizeof(T*) * num_streams);
  path_offsets = (uint32_t**)malloc(sizeof(uint32_t*) * num_streams);

  sort_paths_temp_storage = (void**)malloc(sizeof(void*) * num_streams);

  sort_paths_temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortPairsDescending(nullptr,
                                            sort_paths_temp_storage_bytes,
                                            score_buffers[0],
                                            score_buffers[1],
                                            path_buffers[0],
                                            path_buffers[1],
                                            MAX_PATH_BUFFER_SIZE);

  streams.resize(num_streams);
  for (size_t i = 0; i < num_streams; i++) {
    CUCO_CUDA_TRY(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));

    CUCO_CUDA_TRY(cudaMalloc(&frontiers[i], sizeof(State) * MAX_FRONTIER_SIZE));
    CUCO_CUDA_TRY(cudaMalloc(&next_frontiers[i], sizeof(State) * MAX_FRONTIER_SIZE));

    CUCO_CUDA_TRY(cudaMalloc(&path_buffers[i], sizeof(Path) * MAX_PATH_BUFFER_SIZE));
    CUCO_CUDA_TRY(cudaMalloc(&path_buffers[i + num_streams], sizeof(Path) * MAX_PATH_BUFFER_SIZE));
    CUCO_CUDA_TRY(cudaMalloc(&score_buffers[i], sizeof(float) * MAX_PATH_BUFFER_SIZE));
    CUCO_CUDA_TRY(
      cudaMalloc(&score_buffers[i + num_streams], sizeof(float) * MAX_PATH_BUFFER_SIZE));

    CUCO_CUDA_TRY(cudaMalloc(&path_values[i], sizeof(T) * 1000 * 100));
    CUCO_CUDA_TRY(cudaMalloc(&path_offsets[i], sizeof(uint32_t) * 1000));

    CUCO_CUDA_TRY(cudaMalloc(&sort_paths_temp_storage[i], sort_paths_temp_storage_bytes));
  }
}

template <typename T>
PathEnumeration<T>::~PathEnumeration() noexcept(false)
{
  for (size_t i = 0; i < num_streams; i++) {
    CUCO_CUDA_TRY(cudaFree(frontiers[i]));
    CUCO_CUDA_TRY(cudaFree(next_frontiers[i]));
    CUCO_CUDA_TRY(cudaFree(path_buffers[2 * i]));
    CUCO_CUDA_TRY(cudaFree(path_buffers[2 * i + 1]));
    CUCO_CUDA_TRY(cudaFree(score_buffers[2 * i]));
    CUCO_CUDA_TRY(cudaFree(score_buffers[2 * i + 1]));
    CUCO_CUDA_TRY(cudaFree(path_values[i]));
    CUCO_CUDA_TRY(cudaFree(path_offsets[i]));
    CUCO_CUDA_TRY(cudaFree(sort_paths_temp_storage[i]));
    CUCO_CUDA_TRY(cudaStreamDestroy(streams[i]));
  }
  free(frontiers);
  free(next_frontiers);
  free(path_buffers);
  free(score_buffers);
  free(path_values);
  free(path_offsets);
  free(sort_paths_temp_storage);

  CUCO_CUDA_TRY(cudaFree(num_paths_outs));
}

template <typename T>
void PathEnumeration<T>::sync_streams()
{
  for (auto& stream : streams) {
    CUCO_CUDA_TRY(cudaStreamSynchronize(stream));
  }
}

template <typename T>
__global__ void __launch_bounds__(FIND_PATH_BLOCK_SIZE, 1)
  find_paths_kernel(const cuco::experimental::trie<T>* t,
                    const uint32_t* keys,
                    const float* scores,
                    State* frontier_,
                    State* next_frontier_,
                    uint32_t* num_paths_out,
                    Path* path_buffer,
                    float* score_buffer,
                    uint32_t max_depth,
                    uint32_t max_paths);

template <typename T>
void PathEnumeration<T>::find_paths(const cuco::experimental::trie<T>* trie,
                                    const uint32_t* keys,
                                    const float* scores,
                                    uint32_t max_depth,
                                    uint32_t max_paths,
                                    uint32_t stream_id)
{
  assert(stream_id < streams.size());
  auto& stream = streams[stream_id];

  num_levels_ = trie->num_levels();
  max_depth   = min(num_levels_ - 1, max_depth);

  find_paths_kernel<<<1, FIND_PATH_BLOCK_SIZE, 0, stream>>>(trie->device_ptr_,
                                                            keys,
                                                            scores,
                                                            frontiers[stream_id],
                                                            next_frontiers[stream_id],
                                                            num_paths_outs + stream_id,
                                                            path_buffers[stream_id],
                                                            score_buffers[stream_id],
                                                            max_depth,
                                                            max_paths);
  CUCO_CUDA_TRY(cudaStreamSynchronize(stream));

  sort_paths(stream_id);
  CUCO_CUDA_TRY(cudaStreamSynchronize(stream));

  generate_full_paths<<<1, min(max_paths, 1024), 0, stream>>>(trie->device_ptr_,
                                                              path_buffers[stream_id],
                                                              score_buffers[stream_id],
                                                              path_values[stream_id],
                                                              path_offsets[stream_id],
                                                              max_paths);
  CUCO_CUDA_TRY(cudaStreamSynchronize(stream));

  if (CHECK_FIND_PATHS_RESULT) { check_result(stream_id, max_paths); }
}

__device__ float score_node(const uint32_t* keys, const float* scores, uint32_t label, bool& match);

template <typename BV>
__device__ uint32_t init_node_pos(const BV& louds, uint32_t& node_id)
{
  uint32_t node_pos = 0;
  if (node_id != 0) {
    node_pos = louds.select(node_id - 1) + 1;
    node_id  = node_pos - node_id;
  }
  return node_pos;
}

template <typename T>
__global__ void __launch_bounds__(FIND_PATH_BLOCK_SIZE, 1)
  find_paths_kernel(const cuco::experimental::trie<T>* t,
                    const uint32_t* keys,
                    const float* scores,
                    State* frontier_,
                    State* next_frontier_,
                    uint32_t* num_paths_out,
                    Path* path_buffer,
                    float* score_buffer,
                    uint32_t max_depth,
                    uint32_t max_paths)
{
  uint32_t offset = threadIdx.x;
  while (offset < MAX_PATH_BUFFER_SIZE) {
    score_buffer[offset] = score_sentinel;
    offset += blockDim.x;
  }

  State* frontier      = frontier_;
  State* next_frontier = next_frontier_;
  uint32_t cur_depth   = 1;

  __shared__ uint32_t frontier_size;
  __shared__ uint32_t next_frontier_index;
  __shared__ uint32_t path_buffer_index;
  __shared__ uint32_t level_keys[TOPK_KEYS_PER_LEVEL];

  if (threadIdx.x == 0) {
    next_frontier_index = 0;
    path_buffer_index   = 0;

    frontier[0]   = {0, 0.0};
    frontier_size = 1;
  }
  __syncthreads();

  while (cur_depth <= max_depth and frontier_size > 0) {
    const auto& level = t->d_levels_ptr_[cur_depth];
    const auto& louds = t->d_louds_refs_ptr_[cur_depth];
    const auto& outs  = t->d_outs_refs_ptr_[cur_depth];

    uint32_t num_iters      = (frontier_size - 1) / blockDim.x + 1;
    uint32_t frontier_index = threadIdx.x;
    for (uint32_t iter = 0; iter < num_iters; iter++) {
      Edge parent     = {0, 0};
      uint32_t degree = 0;

      if (frontier_index < frontier_size) {
        const auto s      = frontier[frontier_index];
        uint32_t node_id  = s.node_id;
        uint32_t node_pos = init_node_pos(louds, node_id);
        uint32_t pos_end  = louds.find_next_set(node_pos);
        uint32_t node_end = node_id + (pos_end - node_pos);

        parent = {node_id, s.score};
        degree = node_end - node_id;
      }

      auto process_edge = [&](uint32_t node_id, float score, uint32_t label, bool terminal_path) {
        bool valid_path;
        if (cur_depth == 1) {
          valid_path = label == keys[0];
          score += scores[0];
        } else {
          uint32_t offset = (cur_depth - 1) * TOPK_KEYS_PER_LEVEL;
          score += score_node(level_keys, scores + offset, label, valid_path);
        }

        if (valid_path) {
          uint32_t insert_index = min(MAX_FRONTIER_SIZE - 1, atomicAdd(&next_frontier_index, 1));
          next_frontier[insert_index] = {node_id, score};

          if (terminal_path) {
            uint32_t insert_index = min(MAX_PATH_BUFFER_SIZE - 1, atomicAdd(&path_buffer_index, 1));
            path_buffer[insert_index]  = {node_id, cur_depth};
            score_buffer[insert_index] = score;
          }
        }
      };

      using RunItemT                              = Edge;
      using RunLengthT                            = uint32_t;
      constexpr uint32_t BLOCK_DIM_X              = FIND_PATH_BLOCK_SIZE;
      constexpr uint32_t RUNS_PER_THREAD          = 1;
      constexpr uint32_t DECODED_ITEMS_PER_THREAD = 3;

      using BlockRunLengthDecodeT =
        cub::BlockRunLengthDecode<RunItemT, BLOCK_DIM_X, RUNS_PER_THREAD, DECODED_ITEMS_PER_THREAD>;
      __shared__ typename BlockRunLengthDecodeT::TempStorage temp_storage;

      RunItemT run_values[RUNS_PER_THREAD];
      RunLengthT run_lengths[RUNS_PER_THREAD];
      run_values[0]  = parent;
      run_lengths[0] = degree;

      uint32_t total_decoded_size = 0;
      BlockRunLengthDecodeT block_rld(temp_storage, run_values, run_lengths, total_decoded_size);

      uint32_t decoded_window_offset = 0U;
      while (decoded_window_offset < total_decoded_size) {
        RunItemT decoded_items[DECODED_ITEMS_PER_THREAD];
        RunLengthT relative_offsets[DECODED_ITEMS_PER_THREAD];

        int num_valid_items = total_decoded_size - decoded_window_offset;
        block_rld.RunLengthDecode(decoded_items, relative_offsets, decoded_window_offset);
        decoded_window_offset += BLOCK_DIM_X * DECODED_ITEMS_PER_THREAD;

        uint32_t labels[DECODED_ITEMS_PER_THREAD];
        bool terminal_paths[DECODED_ITEMS_PER_THREAD];

        uint32_t start_offset       = DECODED_ITEMS_PER_THREAD * threadIdx.x;
        uint32_t thread_valid_items = 0;
        // Manual loop unrolling from 0 to DECODED_ITEMS_PER_THREAD - 1
        thread_valid_items += start_offset + 0 < num_valid_items;
        thread_valid_items += start_offset + 1 < num_valid_items;
        thread_valid_items += start_offset + 2 < num_valid_items;

        for (uint32_t item = 0; item < thread_valid_items; item++) {
          auto node_id         = decoded_items[item].node_id + relative_offsets[item];
          labels[item]         = level.d_labels_ptr[node_id];
          terminal_paths[item] = outs.get(node_id);
        }

        for (uint32_t item = 0; item < thread_valid_items; item++) {
          auto node_id = decoded_items[item].node_id + relative_offsets[item];
          process_edge(node_id, decoded_items[item].score, labels[item], terminal_paths[item]);
        }
      }

      frontier_index += blockDim.x;
      __syncthreads();
    }  // Iters end

    State* temp   = frontier;
    frontier      = next_frontier;
    next_frontier = temp;

    cur_depth++;

    if (cur_depth <= max_depth) {
      uint32_t offset = (cur_depth - 1) * TOPK_KEYS_PER_LEVEL;
      for (uint32_t pos = threadIdx.x; pos < TOPK_KEYS_PER_LEVEL; pos += blockDim.x) {
        level_keys[pos] = keys[offset + pos];
      }
    }

    if (threadIdx.x == 0) {
      frontier_size       = min(MAX_FRONTIER_SIZE, next_frontier_index);
      next_frontier_index = 0;
    }

    __syncthreads();
  }  // Level end

  if (threadIdx.x == 0) { *num_paths_out = path_buffer_index; }
}

template <typename T>
void PathEnumeration<T>::sort_paths(uint32_t stream_id) const
{
  auto& stream            = streams[stream_id];
  auto temp_storage_bytes = sort_paths_temp_storage_bytes;
  cub::DeviceRadixSort::SortPairsDescending(sort_paths_temp_storage[stream_id],
                                            temp_storage_bytes,
                                            score_buffers[stream_id],
                                            score_buffers[num_streams + stream_id],
                                            path_buffers[stream_id],
                                            path_buffers[num_streams + stream_id],
                                            MAX_PATH_BUFFER_SIZE,
                                            0,
                                            sizeof(float) * 8,
                                            stream);
}

template <typename T>
void PathEnumeration<T>::check_result(uint32_t stream_id, uint32_t max_paths) const
{
  auto& stream = streams[stream_id];

  uint32_t num_paths = 0;
  cudaMemcpyAsync(
    &num_paths, num_paths_outs + stream_id, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
  CUCO_CUDA_TRY(cudaStreamSynchronize(stream));
  std::cout << "Num paths " << num_paths << std::endl;

  std::vector<Path> paths_out(max_paths);
  std::vector<float> scores_out(max_paths);
  cudaMemcpyAsync(&paths_out[0],
                  path_buffers[num_streams + stream_id],
                  sizeof(Path) * min(max_paths, MAX_PATH_BUFFER_SIZE),
                  cudaMemcpyDeviceToHost,
                  stream);
  cudaMemcpyAsync(&scores_out[0],
                  score_buffers[num_streams + stream_id],
                  sizeof(float) * min(max_paths, MAX_PATH_BUFFER_SIZE),
                  cudaMemcpyDeviceToHost,
                  stream);

  CUCO_CUDA_TRY(cudaStreamSynchronize(stream));

  for (uint32_t path_id = 0; path_id < 5; path_id++) {
    auto path = paths_out[path_id];
    std::cout << "Path " << path_id << ": " << path.node_id << " @ " << path.level_id << " "
              << scores_out[path_id] << std::endl;
  }
}

template <typename T>
__device__ void backtrace_path(const cuco::experimental::trie<T>* t,
                               int32_t level_id,
                               uint32_t start_node_id,
                               int32_t buffer_pos,
                               T* buffer)
{
  if (level_id == 0) { return; }
  uint32_t node_pos = t->d_louds_refs_ptr_[level_id].select0(start_node_id);
  for (; level_id >= 1; level_id--) {
    const auto& level    = t->d_levels_ptr_[level_id];
    uint32_t rank        = t->d_louds_refs_ptr_[level_id].rank(node_pos);
    uint32_t node_id     = node_pos - rank;
    buffer[--buffer_pos] = level.d_labels_ptr[node_id];  // insert in reverse order

    if (level_id > 1) { node_pos = t->d_louds_refs_ptr_[level_id - 1].select0(rank); }
  }
}

template <typename T>
__global__ void generate_full_paths(const cuco::experimental::trie<T>* t,
                                    Path* path_buffer,
                                    float* score_buffer,
                                    T* path_values,
                                    uint32_t* path_offsets,
                                    uint32_t num_paths)
{
  if (threadIdx.x < num_paths) { path_offsets[threadIdx.x] = path_buffer[threadIdx.x].level_id; }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (uint32_t id = 1; id < num_paths + 1; id++) {
      path_offsets[id] = path_offsets[id - 1] + path_offsets[id];
    }
  }
  __syncthreads();

  if (threadIdx.x < num_paths) {
    const Path p    = path_buffer[threadIdx.x];
    uint32_t offset = path_offsets[threadIdx.x + 1];
    backtrace_path(t, p.level_id, p.node_id, offset, path_values);
  }
}

__device__ __forceinline__ float score_node(const uint32_t* keys,
                                            const float* scores,
                                            uint32_t label,
                                            bool& match)
{
  uint32_t ret = (keys[128] <= label) * 128;
  ret += (keys[ret + 64] <= label) * 64;
  ret += (keys[ret + 32] <= label) * 32;
  ret += (keys[ret + 16] <= label) * 16;
  ret += (keys[ret + 8] <= label) * 8;
  ret += (keys[ret + 4] <= label) * 4;
  ret += (keys[ret + 2] <= label) * 2;
  ret += (keys[ret + 1] <= label) * 1;

  float val = scores[ret];
  match     = keys[ret] == label and val > node_score_threshold;
  return match * val + (1 - match) * score_sentinel;
}

int main(void)
{
  using KeyType = uint32_t;
  cuco::experimental::trie<KeyType> trie;

  for (auto key : read_dataset(10 * 1000 * 1000)) {
    trie.insert(key);
  }
  trie.build();

  std::cout << "#keys " << trie.n_keys() << "   "
            << "#nodes " << trie.n_nodes() << std::endl;

  vector<const uint32_t*> topk_keys;
  vector<const float*> topk_scores;
  read_topk_keys_and_scores(topk_keys, topk_scores, num_topk_id, max_depth);

  PathEnumeration<KeyType> pe;
  for (size_t topk_id = 0; topk_id < num_topk_id; topk_id++) {  // warmup
    pe.find_paths(&trie, topk_keys[topk_id], topk_scores[topk_id], max_depth, max_paths, 0);
  }
  pe.sync_streams();

  auto begin            = high_resolution_clock::now();
  uint32_t topk_repeats = 10;
  for (uint32_t repeat = 0; repeat < topk_repeats; repeat++) {
    //#pragma omp parallel for num_threads(num_streams)
    for (size_t topk_id = 0; topk_id < num_topk_id; topk_id++) {
      pe.find_paths(&trie, topk_keys[topk_id], topk_scores[topk_id], max_depth, max_paths, 0);
    }
    pe.sync_streams();
  }
  uint32_t total_calls = topk_repeats * num_topk_id;
  cout << "GPU find paths " << get_milliseconds(begin) / total_calls << " msec" << endl;

  return 0;
}
