/* -*- c++ -*- */
/*
 * Copyright 2021 Josh Morman
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#include <helper_cuda.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cusp/convolve.cuh>

namespace cusp {

extern "C" __global__ void __launch_bounds__(512)
    _cupy_convolve_float32(const float *__restrict__ inp, const int inpW,
                           const float *__restrict__ kernel, const int kerW,
                           const int mode, const bool swapped_inputs,
                           float *__restrict__ out, const int outW);

extern "C" __global__ void __launch_bounds__(512) _cupy_convolve_complex64(
    thrust::complex<float> *__restrict__ inp, const int inpW,
    thrust::complex<float> *__restrict__ kernel, const int kerW, const int mode,
    const bool swapped_inputs, thrust::complex<float> *__restrict__ out,
    const int outW);

template <typename T, typename T_TAPS>
convolve<T, T_TAPS>::convolve(const std::vector<T_TAPS> &taps,
                              const convolve_mode_t mode)
    : _taps(taps), _mode(mode) {
  checkCudaErrors(cudaMalloc(&_dev_taps, taps.size() * sizeof(T)));
  checkCudaErrors(cudaMemcpy(_dev_taps, taps.data(), taps.size() * sizeof(T),
                             cudaMemcpyHostToDevice));
  occupancy_internal();
};

template <>
cudaError_t convolve<float, float>::launch(const float *in, float *out, int N,
                                           int grid_size, int block_size,
                                           cudaStream_t stream) {

  auto N_out = output_length(N);

  if (stream) {
    _cupy_convolve_float32<<<grid_size, block_size, 0, stream>>>(
        in, N, _dev_taps, _taps.size(), (int)_mode, false, out, N_out);
  } else {
    _cupy_convolve_float32<<<grid_size, block_size>>>(
        in, N, _dev_taps, _taps.size(), (int)_mode, false, out, N_out);
  }
  return cudaPeekAtLastError();
}

template <>
cudaError_t convolve<std::complex<float>, std::complex<float>>::launch(
    const std::complex<float> *in, std::complex<float> *out, int N,
    int grid_size, int block_size, cudaStream_t stream) {

  auto N_out = output_length(N);
  auto mode = _mode == convolve_mode_t::FULL_TRUNC ? convolve_mode_t::FULL : _mode;
  
  if (stream) {
    _cupy_convolve_complex64<<<grid_size, block_size, 0, stream>>>(
        (thrust::complex<float> *)in, N, (thrust::complex<float>*)_dev_taps, _taps.size(), (int)mode,
        false, (thrust::complex<float> *)out, N_out);
  } else {
    _cupy_convolve_complex64<<<grid_size, block_size>>>(
        (thrust::complex<float> *)in, N, (thrust::complex<float>*)_dev_taps, _taps.size(), (int)mode,
        false, (thrust::complex<float> *)out, N_out);
  }
  return cudaPeekAtLastError();
}

template <typename T, typename T_TAPS>
cudaError_t convolve<T, T_TAPS>::launch(const std::vector<const void *> &inputs,
                                        const std::vector<void *> &outputs,
                                        size_t nitems) {
  return launch((const T *)inputs[0], (T *)outputs[0], nitems, _grid_size,
                _block_size, _stream);
}

template <>
cudaError_t convolve<float, float>::occupancy(int *minBlock, int *minGrid) {
  auto rc = cudaOccupancyMaxPotentialBlockSize(minGrid, minBlock,
                                               _cupy_convolve_float32, 0, 0);
  *minBlock =
      std::min(*minBlock, 512); // Convolve kernels are limited to 512 threads

  return rc;
}

template <>
cudaError_t
convolve<std::complex<float>, std::complex<float>>::occupancy(int *minBlock,
                                                              int *minGrid) {
  auto rc = cudaOccupancyMaxPotentialBlockSize(minGrid, minBlock,
                                               _cupy_convolve_complex64, 0, 0);
  *minBlock =
      std::min(*minBlock, 512); // Convolve kernels are limited to 512 threads

  return rc;
}

template <typename T, typename T_TAPS>
int convolve<T, T_TAPS>::output_length(int input_length) {
  int N_out = input_length;
  if (_mode == convolve_mode_t::VALID) {
    N_out = input_length - _taps.size() + 1;
  } else if (_mode == convolve_mode_t::FULL) {
    N_out = input_length + _taps.size() - 1;
  }
  return N_out;
}

#define IMPLEMENT_KERNEL(T, T_TAPS) template class convolve<T, T_TAPS>;

// IMPLEMENT_KERNEL(int8_t)
// IMPLEMENT_KERNEL(int16_t)
// IMPLEMENT_KERNEL(int32_t)
// IMPLEMENT_KERNEL(int64_t)
IMPLEMENT_KERNEL(float, float)
IMPLEMENT_KERNEL(std::complex<float>, std::complex<float>)

} // namespace cusp