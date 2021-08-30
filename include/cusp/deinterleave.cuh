/* -*- c++ -*- */
/*
 * Copyright 2021 Josh Morman
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#pragma once

#include <cusp/kernel.cuh>
#include <thrust/complex.h>

namespace cusp {
class deinterleave : public kernel {
private:
  int _nstreams;
  int _itemsperstream;
  int _itemsize;
  void **_dev_ptr_array;

public:
  deinterleave(int nstreams, int itemsperstream, int itemsize);
  cudaError_t launch(const uint8_t *input, const std::vector<void *> &outputs,
                     int nstreams, int itemsperstream, int itemsize,
                     int grid_size, int block_size, size_t nitems,
                     cudaStream_t stream = 0);
  virtual cudaError_t launch(const std::vector<const void *> &inputs,
                             const std::vector<void *> &outputs,
                             size_t nitems) override;
  virtual cudaError_t occupancy(int *minBlock, int *minGrid);

protected:
  virtual size_t threads_per_item() override { return _itemsize; }
};

} // namespace cusp