/* -*- c++ -*- */
/*
 * Copyright 2021 Mark Bauer
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 *
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

#include <complex>
#include <cusp/add.cuh>

namespace cusp {

template <typename T>
__global__ void kernel_add(const T **ins, T *out, int ninputs, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    T *in = (T *)(*ins);
    out[i] = in[i];
    for (int j = 1; j < ninputs; j++) {
      in = (T*)(*(ins+j));
      out[i] += in[i];
    }
  }
}

template <>
__global__ void kernel_add<thrust::complex<float>>(const thrust::complex<float> **ins,
                                                   thrust::complex<float> *out,
                                                   int ninputs, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    thrust::complex<float> *in = (thrust::complex<float> *)(*ins);
    out[i] = in[i];
    for (int j = 1; j < ninputs; j++) {
      in = (thrust::complex<float>*)(*(ins+j));
      out[i] += in[i];
    }
  }
}

template <typename T> add<T>::add(int ninputs) : _ninputs(ninputs) {
  checkCudaErrors(cudaMalloc(&_dev_ptr_array, sizeof(void *) * _ninputs));
}

template <typename T>
cudaError_t add<T>::launch(const std::vector<const void *>& inputs, T *output,
                           int ninputs, int grid_size, int block_size,
                           size_t nitems, cudaStream_t stream) {

  // There is a better way to do this here - just getting the pointers into
  // device memory
  checkCudaErrors(cudaMemcpy(_dev_ptr_array, inputs.data(), sizeof(void *) * ninputs,
             cudaMemcpyHostToDevice));

  if (stream) {
    kernel_add<<<grid_size, block_size, 0, stream>>>(
        (const T **)_dev_ptr_array,
         output, ninputs, nitems
    );
  } else {
    kernel_add<<<grid_size, block_size>>>(
        (const T **)_dev_ptr_array, output,
         ninputs, nitems
    );
  }
  return cudaPeekAtLastError();
}

template <>
cudaError_t add<std::complex<float>>::launch(
    const std::vector<const void *>& inputs,
    std::complex<float> *output, int ninputs,
    int grid_size, int block_size, size_t nitems,
    cudaStream_t stream) {

  // There is a better way to do this here - just getting the pointers into
  // device memory
  checkCudaErrors(cudaMemcpy(_dev_ptr_array, inputs.data(), sizeof(void *) * ninputs,
             cudaMemcpyHostToDevice));

  if (stream) {
    kernel_add<<<grid_size, block_size, 0, stream>>>(
        (const thrust::complex<float> **)_dev_ptr_array,
        (thrust::complex<float> *)output, ninputs, nitems
    );
  } else {
    kernel_add<<<grid_size, block_size>>>(
        (const thrust::complex<float> **)_dev_ptr_array,
        (thrust::complex<float> *) output, ninputs, nitems
    );
  }
  return cudaPeekAtLastError();
}

template <typename T>
cudaError_t add<T>::launch(const std::vector<const void *>& inputs,
                           const std::vector<void *>& outputs, size_t nitems) {
  return launch(inputs, (T *)outputs[0], _ninputs, _grid_size, _block_size,
                nitems, _stream);
}

template <typename T>
cudaError_t add<T>::occupancy(int *minBlock, int *minGrid) {
  return cudaOccupancyMaxPotentialBlockSize(minGrid, minBlock, kernel_add<T>, 0,
                                            0);
}

template <>
cudaError_t add<std::complex<float>>::occupancy(int *minBlock, int *minGrid) {
  return cudaOccupancyMaxPotentialBlockSize(minGrid, minBlock,
                                            kernel_add<thrust::complex<float>>,
                                            0, 0);
}

#define IMPLEMENT_KERNEL(T) template class add<T>;

IMPLEMENT_KERNEL(int8_t)
IMPLEMENT_KERNEL(int16_t)
IMPLEMENT_KERNEL(int32_t)
IMPLEMENT_KERNEL(int64_t)
IMPLEMENT_KERNEL(float)
IMPLEMENT_KERNEL(std::complex<float>)

} // namespace cusp