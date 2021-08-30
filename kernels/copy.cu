/* -*- c++ -*- */
/*
 * Copyright 2021 Josh Morman
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#include <cuda.h>
#include <cuda_runtime.h>

#include <helper_cuda.h>
#include <cusp/copy.cuh>

namespace cusp {

template <typename T> __global__ void kernel_copy(const T *in, T *out, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    out[i] = in[i];
  }
}

template <typename T>
cudaError_t copy<T>::launch(const T *in, T *out, int N, int grid_size, int block_size,
                  cudaStream_t stream) {

  kernel_copy<<<grid_size, block_size, 0, stream>>>(in, out, N);
  return cudaPeekAtLastError();

}

template <typename T>
cudaError_t copy<T>::launch(const std::vector<const void *>& inputs,
                  const std::vector<void *>& outputs, size_t nitems) {
  return launch((const T*)inputs[0], (T*)outputs[0], nitems, _grid_size, _block_size, _stream);
}

template <typename T> cudaError_t copy<T>::occupancy(int *minBlock, int *minGrid) {
  return cudaOccupancyMaxPotentialBlockSize(minGrid, minBlock, kernel_copy<T>, 0, 0);
}

#define IMPLEMENT_KERNEL(T) template class copy<T>;

// The Copy Kernel is only implemented for different underlying memcpy sizes
//  Not for each type that might be used since it is type agnostic
IMPLEMENT_KERNEL(uint8_t)
IMPLEMENT_KERNEL(uint16_t)
IMPLEMENT_KERNEL(uint32_t)
IMPLEMENT_KERNEL(uint64_t)

} // namespace cusp
