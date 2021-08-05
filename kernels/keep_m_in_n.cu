/* -*- c++ -*- */
/*
 * Copyright 2021 Mark Bauer
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 *
 */

#include <cuda.h>
#include <cuda_runtime.h>

#include <cusp/keep_m_in_n.cuh>
#include <complex>

namespace cusp {

template <typename T>
__global__ void kernel_keep_m_in_n(const T *in, T *out, int m, int window, int N) {
  
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        if (i % window == 0) {
            int window_number = 0;
            if (i > window - 1) {
                window_number = i / window;
            }
            for (int j = 0; j < m; j++) {
              if (i + j < N) out[j + window_number * m] = in[i + j];
            }
        }
    }
}

template <>
__global__ void kernel_keep_m_in_n<thrust::complex<float>>(
  const thrust::complex<float> *in, 
  thrust::complex<float> *out,
  int m, int window, int N) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < N) {
        if (i % window == 0) {
            int window_number = 0;
            if (i > window - 1) {
                window_number = i / window;
            }
            for (int j = 0; j < m; j++) {
              if (i + j < N) out[j + window_number * m] = in[i + j];
            }
        }
    }
}

template <typename T>
cudaError_t keep_m_in_n<T>::launch(const T *in, T *out, int m, int window, int N,
                                   int grid_size, int block_size, cudaStream_t stream) {
  if (stream) {
    kernel_keep_m_in_n<<<grid_size, block_size, 0, stream>>>(in, out, m, window, N);
  } else {
    kernel_keep_m_in_n<<<grid_size, block_size>>>(in, out, m, window, N);
  }
  return cudaPeekAtLastError();
}

template <>
cudaError_t keep_m_in_n<std::complex<float>>::launch(
  const std::complex<float> *in, std::complex<float> *out,
  int m, int window, int N, int grid_size, int block_size,
  cudaStream_t stream) {

  if (stream) {
    kernel_keep_m_in_n<<<grid_size, block_size, 0, stream>>>(
      (const thrust::complex<float> *)in,
      (thrust::complex<float> *)out, m, window, N);
  } else {
    kernel_keep_m_in_n<<<grid_size, block_size>>>(
      (const thrust::complex<float> *)in,
      (thrust::complex<float> *)out, m, window, N);
  }
  return cudaPeekAtLastError();
}

template <typename T>
cudaError_t keep_m_in_n<T>::launch(const std::vector<const void *>& inputs,
                                 const std::vector<void *>& outputs,
                                 size_t nitems) {
    return launch(
        (const T *)inputs[0], (T *)outputs[0], _m, _window,
         nitems, _grid_size, _block_size, _stream
    );
}

template <typename T>
cudaError_t keep_m_in_n<T>::occupancy(int *minBlock, int *minGrid) {
  return cudaOccupancyMaxPotentialBlockSize(minGrid, minBlock,
                                            kernel_keep_m_in_n<T>, 0, 0);
}

template <>
cudaError_t keep_m_in_n<std::complex<float>>::occupancy(int *minBlock, int *minGrid) {
  return cudaOccupancyMaxPotentialBlockSize(minGrid, minBlock,
                                            kernel_keep_m_in_n<std::complex<float>>, 0, 0);
}

#define IMPLEMENT_KERNEL(T) template class keep_m_in_n<T>;

IMPLEMENT_KERNEL(int8_t)
IMPLEMENT_KERNEL(int16_t)
IMPLEMENT_KERNEL(int32_t)
IMPLEMENT_KERNEL(int64_t)
IMPLEMENT_KERNEL(float)
IMPLEMENT_KERNEL(std::complex<float>)

} // namespace cusp