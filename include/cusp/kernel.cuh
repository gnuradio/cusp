/* -*- c++ -*- */
/*
 * Copyright 2021 Josh Morman
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 */
 
#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <iostream>

namespace cusp {

class kernel {
protected:
  int _block_size;
  int _grid_size;
  cudaStream_t _stream = 0;

  int _min_block = -1;
  int _min_grid = -1;

public:
  kernel() = default;
  virtual ~kernel() = default;
  void set_stream(cudaStream_t stream) { _stream = stream; }
  void set_block_and_grid(int block_size, int grid_size) {
    _block_size = block_size;
    _grid_size = grid_size;
  }
  virtual cudaError_t launch(const std::vector<const void *> &inputs,
                             const std::vector<void *> &outputs,
                             size_t nitems) = 0;

  virtual cudaError_t
  launch_default_occupancy(const std::vector<const void *> &inputs,
                           const std::vector<void *> &outputs, size_t nitems) {
    if (_min_block < 0 || _min_grid < 0) {
      // throw std::runtime_error("occupancy() must be called prior to
      // launch_default_occupancy");
      occupancy_internal();
    }
    int gridSize = (nitems + _min_block - 1) / _min_block;
    set_block_and_grid(_min_block, gridSize);
    return launch(inputs, outputs, nitems);
  }

  virtual cudaError_t occupancy(int *minBlock, int *minGrid) = 0;
  cudaError_t occupancy_internal() { return occupancy(&_min_block, &_min_grid); }
};

} // namespace cusp