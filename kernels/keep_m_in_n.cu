/* -*- c++ -*- */
/*
 * Copyright 2021 Mark Bauer
 * Copyright 2022 Josh Morman
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 *
 */

#include <cuda.h>
#include <cuda_runtime.h>

#include <complex>
#include <cusp/keep_m_in_n.cuh>

namespace cusp {

// This is going to run with M*itemsize threads where M is a multiple of m
template <typename T>
__global__ void kernel_keep_m_in_n(const T *in, T *out, int m, int n, int itemsize,
                                   int offset, int M) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < M*itemsize) {
    int window_num = i / (m*itemsize);
    int midx = (i / itemsize)%m;
    int itemidx = i%itemsize;
    int idx = midx*itemsize + itemidx + offset * itemsize;
    int excess = (offset + m - n) * itemsize;
    if (excess <= 0) {
      out[i] = in[(window_num * n)*itemsize + idx];
    } else {
      if (idx <= excess) {
        out[i] = in[(window_num * n)*itemsize + i % (m*itemsize)];
      } else {
        out[i] = in[(window_num * n)*itemsize + excess + offset*itemsize + i % (m*itemsize)];
      }
    }
  }
}

template <typename T>
cudaError_t keep_m_in_n<T>::launch(const T *in, T *out, int m, int n, int itemsize,
                                   int offset, int grid_size,
                                   int block_size, int M, cudaStream_t stream) {
  if (stream) {
    kernel_keep_m_in_n<<<grid_size, block_size, 0, stream>>>(in, out, m, n, itemsize,
                                                             offset, M);
  } else {
    kernel_keep_m_in_n<<<grid_size, block_size>>>(in, out, m, n, itemsize, offset, M);
  }
  return cudaPeekAtLastError();
}

template <typename T>
cudaError_t keep_m_in_n<T>::launch(const std::vector<const void *> &inputs,
                                   const std::vector<void *> &outputs,
                                   size_t nitems) {
  return launch((const T *)inputs[0], (T *)outputs[0], _m, _n, _itemsize, _offset,
                _grid_size, _block_size, nitems, _stream);
}

template <typename T>
cudaError_t keep_m_in_n<T>::occupancy(int *minBlock, int *minGrid) {
  return cudaOccupancyMaxPotentialBlockSize(minGrid, minBlock,
                                            kernel_keep_m_in_n<T>, 0, 0);
}

#define IMPLEMENT_KERNEL(T) template class keep_m_in_n<T>;

IMPLEMENT_KERNEL(uint8_t)
IMPLEMENT_KERNEL(uint16_t)
IMPLEMENT_KERNEL(uint32_t)
IMPLEMENT_KERNEL(uint64_t)

} // namespace cusp