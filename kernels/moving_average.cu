/* -*- c++ -*- */
/*
 * Copyright 2021 Mark Bauer
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 *
 */

#include <cuda.h>
#include <cuda_runtime.h>

#include <cusp/moving_average.cuh>
#include <complex>

namespace cusp {

template <typename T>
__global__ void kernel_moving_average(const T *in, T *out, int l, float s, int N) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    T sum = (T)0;

    if (i < l - 1) {
        out[i] = (T)0;
    }
    else if (i < N) {
        for (int j = 0; j < l; j++) {
            sum += in[i - j];
        }
        out[i] = sum * s;
    }
}

template <typename T>
cudaError_t moving_average<T>::launch(const T *in, T *out, int l, float s,
                                      int N, int grid_size, int block_size,
                                      cudaStream_t stream) {
  if (stream) {
    kernel_moving_average<<<grid_size, block_size, 0, stream>>>(in, out, l, s, N);
  } else {
    kernel_moving_average<<<grid_size, block_size>>>(in, out, l, s, N);
  }
  return cudaPeekAtLastError();
}

template <typename T>
cudaError_t moving_average<T>::launch(const std::vector<const void *>& inputs,
                                 const std::vector<void *>& outputs,
                                 size_t nitems) {
  return launch((const T *)inputs[0], (T *)outputs[0], _l, _s, nitems, _grid_size,
                _block_size, _stream);
}

template <typename T>
cudaError_t moving_average<T>::occupancy(int *minBlock, int *minGrid) {
  return cudaOccupancyMaxPotentialBlockSize(minGrid, minBlock,
                                            kernel_moving_average<T>, 0, 0);
}


#define IMPLEMENT_KERNEL(T) template class moving_average<T>;

IMPLEMENT_KERNEL(int8_t)
IMPLEMENT_KERNEL(int16_t)
IMPLEMENT_KERNEL(int32_t)
IMPLEMENT_KERNEL(int64_t)
IMPLEMENT_KERNEL(float)

} // namespace cusp