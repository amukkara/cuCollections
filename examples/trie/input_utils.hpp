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

#include <omp.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cuco/detail/error.hpp>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <vector>

using namespace std;
using namespace std::chrono;
using Key = uint32_t;

double get_seconds(high_resolution_clock::time_point begin)
{
  return (double)duration_cast<seconds>(high_resolution_clock::now() - begin).count();
}

vector<string> read_input_keys(const char* filename, uint32_t max_keys, uint32_t start_offset);

void preprocess_input_keys(vector<string>& keys);

vector<vector<Key>> generate_split_keys(const vector<string>& keys);

vector<vector<Key>> read_dataset(int max_keys_per_file)
{
  int start_offset = 284580752;  // keys starting with 1040 begin at offset 284M

  // cout << "OMP threads " << omp_get_max_threads() << endl;
  vector<std::string> filenames{"dataset/bing/full.txt"};
  vector<string> keys;
  for (auto& name : filenames) {
    auto file_keys = read_input_keys(name.c_str(), start_offset, max_keys_per_file);
    keys.insert(keys.end(), file_keys.begin(), file_keys.end());
  }
  cout << "Total keys " << int(keys.size() / 1000000) << "M" << endl;

  preprocess_input_keys(keys);
  return generate_split_keys(keys);
}

template <typename T>
T* linearize_vector(const vector<vector<T>>& input)
{
  const uint32_t num_elems = input.size() * input[0].size();
  vector<T> host_vec(num_elems);
  uint32_t pos = 0;
  for (auto row : input) {
    for (auto elem : row) {
      host_vec[pos++] = elem;
    }
  }

  T* output;
  CUCO_CUDA_TRY(cudaMalloc((void**)&output, sizeof(T) * num_elems));
  CUCO_CUDA_TRY(cudaMemcpy(output, &host_vec[0], sizeof(T) * num_elems, cudaMemcpyHostToDevice));
  return output;
}

vector<string> read_input_keys(const char* filename, uint32_t start_offset, uint32_t max_keys)
{
  ifstream input_file(filename);
  vector<string> keys;
  string line;
  uint32_t i = 0;
  while (i < (start_offset + max_keys) and getline(input_file, line)) {
    if (i >= start_offset) { keys.push_back(line); }
    i++;
  }
  assert(keys.size() > 10);
  return keys;
}

void sort_input_keys(vector<string>& keys);

void preprocess_input_keys(vector<string>& keys)
{
  // auto begin = high_resolution_clock::now();
  // sort_input_keys(keys);
  keys.erase(unique(keys.begin(), keys.end()), keys.end());
  // cout << "preprocess time " << get_seconds(begin) << " sec" << endl;
}

vector<uint32_t> split_str_into_ints(const string& key)
{
  stringstream ss(key);
  vector<uint32_t> tokens;
  string buf;

  while (ss >> buf) {
    tokens.push_back(stoi(buf));
  }
  return tokens;
}

void sort_input_keys(vector<string>& keys)
{
  unordered_map<string, vector<Key>> split_keys;
  //#pragma omp parallel for
  for (size_t i = 0; i < keys.size(); i++) {
    auto s = split_str_into_ints(keys[i]);
    //#pragma omp critical
    split_keys[keys[i]] = s;
  }

  struct cmpFunc {
    cmpFunc(unordered_map<string, vector<Key>>& s) : split_keys(s) {}
    bool operator()(const string& lhs, const string& rhs)
    {
      return split_keys.find(lhs)->second < split_keys.find(rhs)->second;
    }
    const unordered_map<string, vector<Key>>& split_keys;
  };
  sort(keys.begin(), keys.end(), cmpFunc(split_keys));
}

vector<vector<Key>> generate_split_keys(const vector<string>& keys)
{
  vector<vector<Key>> split_keys(keys.size());
  //#pragma omp parallel for
  for (size_t i = 0; i < keys.size(); i++) {
    split_keys[i] = split_str_into_ints(keys[i]);
  }
  return split_keys;
}

template <typename T, typename Compare>
vector<std::size_t> sort_permutation(const vector<T>& vec, Compare compare)
{
  vector<std::size_t> p(vec.size());
  std::iota(p.begin(), p.end(), 0);
  std::sort(
    p.begin(), p.end(), [&](std::size_t i, std::size_t j) { return compare(vec[i], vec[j]); });
  return p;
}

template <typename T>
vector<T> apply_permutation(const vector<T>& vec, const vector<std::size_t>& p)
{
  vector<T> sorted_vec(vec.size());
  std::transform(p.begin(), p.end(), sorted_vec.begin(), [&](std::size_t i) { return vec[i]; });
  return sorted_vec;
}

void read_topk_keys_and_scores(vector<const uint32_t*>& keys_out,
                               vector<const float*>& scores_out,
                               size_t num_topk_id,
                               size_t max_depth)
{
  ifstream keys_file("dataset/bing/100-sev-b.txt");
  ifstream scores_file("dataset/bing/100-sev-a.txt");
  assert(keys_file.is_open());
  assert(scores_file.is_open());

  vector<vector<vector<uint32_t>>> keys;
  vector<vector<vector<float>>> scores;

  keys.resize(num_topk_id);
  scores.resize(num_topk_id);

  string keys_line, scores_line, buf;
  for (uint32_t topk_id = 0; topk_id < num_topk_id; topk_id++) {
    keys[topk_id].resize(max_depth);
    scores[topk_id].resize(max_depth);

    for (uint32_t depth = 0; depth < max_depth; depth++) {
      getline(keys_file, keys_line);
      stringstream keys_ss(keys_line);
      while (keys_ss >> buf) {
        keys[topk_id][depth].push_back(stof(buf));
      }
      assert(keys[topk_id][depth].size() == 300);

      getline(scores_file, scores_line);
      stringstream scores_ss(scores_line);
      while (scores_ss >> buf) {
        scores[topk_id][depth].push_back(stof(buf));
      }
      assert(scores[topk_id][depth].size() == 300);

      if (depth > 0) {
        assert(is_sorted(
          scores[topk_id][depth].begin(), scores[topk_id][depth].end(), std::greater<float>()));
        auto p                 = sort_permutation(keys[topk_id][depth], std::less<uint32_t>());
        keys[topk_id][depth]   = apply_permutation(keys[topk_id][depth], p);
        scores[topk_id][depth] = apply_permutation(scores[topk_id][depth], p);
      }

      // Truncate to 256 keys
      keys[topk_id][depth].resize(256);
      scores[topk_id][depth].resize(256);
    }
  }

  for (size_t topk_id = 0; topk_id < num_topk_id; topk_id++) {
    keys_out.push_back(linearize_vector(keys[topk_id]));
    scores_out.push_back(linearize_vector(scores[topk_id]));
  }
}
