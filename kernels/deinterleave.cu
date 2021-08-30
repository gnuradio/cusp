/* -*- c++ -*- */
/*
 * Copyright 2021 Josh Morman
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#include <complex>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cusp/deinterleave.cuh>
#include <helper_cuda.h>

namespace cusp {

// Might want to expand uint8_t interface to maximize bus throughput
__global__ void kernel_deinterleave(const uint8_t *in, uint8_t **outs,
                                    int nstreams, int itemsperblock,
                                    int itemsize, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    int a = itemsize * itemsperblock;
    int stream_idx = (i / a) % nstreams;
    int out_idx = ((i / a) / nstreams) * a + i % a;
    // ((uint8_t*)(*(outs+stream_idx)))[out_idx] = 3; //in[i];
    outs[stream_idx][out_idx] = in[i];
  }
}

deinterleave::deinterleave(int nstreams, int itemsperstream, int itemsize)
    : _nstreams(nstreams), _itemsperstream(itemsperstream),
      _itemsize(itemsize) {
  checkCudaErrors(cudaMalloc(&_dev_ptr_array, sizeof(void *) * _nstreams));
}

cudaError_t deinterleave::launch(const uint8_t *input,
                                 const std::vector<void *> &outputs,
                                 int nstreams, int itemsperstream, int itemsize,
                                 int grid_size, int block_size, size_t nitems,
                                 cudaStream_t stream) {

  if (stream) {
    // There is a better way to do this here - just getting the pointers into
    // device memory
    checkCudaErrors(cudaMemcpyAsync(_dev_ptr_array, outputs.data(),
                                    sizeof(void *) * nstreams,
                                    cudaMemcpyHostToDevice, stream));
    kernel_deinterleave<<<grid_size, block_size, 0, stream>>>(
        input, (uint8_t **)_dev_ptr_array, nstreams, itemsperstream, itemsize,
        nitems * itemsize);
  } else {
    // There is a better way to do this here - just getting the pointers into
    // device memory
    checkCudaErrors(cudaMemcpy(_dev_ptr_array, outputs.data(),
                               sizeof(void *) * nstreams,
                               cudaMemcpyHostToDevice));
    kernel_deinterleave<<<grid_size, block_size>>>(
        input, (uint8_t **)_dev_ptr_array, nstreams, itemsperstream, itemsize,
        nitems * itemsize);
  }
  return cudaPeekAtLastError();
}

cudaError_t deinterleave::launch(const std::vector<const void *> &inputs,
                                 const std::vector<void *> &outputs,
                                 size_t nitems) {
  return launch((const uint8_t *)inputs[0], outputs, _nstreams, _itemsperstream,
                _itemsize, _grid_size, _block_size, nitems, _stream);
}

cudaError_t deinterleave::occupancy(int *minBlock, int *minGrid) {
  return cudaOccupancyMaxPotentialBlockSize(minGrid, minBlock,
                                            kernel_deinterleave, 0, 0);
}

} // namespace cusp