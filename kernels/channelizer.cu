/* -*- c++ -*- */
/*
 * Copyright 2021 Josh Morman
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#include <cuComplex.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cusp/channelizer.cuh>
#include <helper_cuda.h>

namespace cusp {

extern "C" __global__ void __launch_bounds__(64)
    _cupy_channelizer_8x8_complex64_complex64(
        const int n_chans, const int n_taps, const int n_pts,
        const cuFloatComplex *__restrict__ x,
        const cuFloatComplex *__restrict__ h, cuFloatComplex *__restrict__ y);

extern "C" __global__ void __launch_bounds__(256)
    _cupy_channelizer_16x16_complex64_complex64(
        const int n_chans, const int n_taps, const int n_pts,
        const cuFloatComplex *__restrict__ x,
        const cuFloatComplex *__restrict__ h, cuFloatComplex *__restrict__ y);

extern "C" __global__ void __launch_bounds__(1024)
    _cupy_channelizer_32x32_complex64_complex64(
        const int n_chans, const int n_taps, const int n_pts,
        const cuFloatComplex *__restrict__ x,
        const cuFloatComplex *__restrict__ h, cuFloatComplex *__restrict__ y);

template <typename T>
channelizer<T>::channelizer(const std::vector<T> &taps, const size_t nchans)
    : _taps(taps), _nchans(nchans) {

  std::cout << taps.size() << " " << nchans << std::endl;
  _ntaps = taps.size() / nchans;
  std::cout << _ntaps << std::endl;
  if (_ntaps > 32) {
    throw std::invalid_argument(
        "Number of Taps / Number of Chans must be <= 32");
  }

  checkCudaErrors(cudaMalloc(&_dev_taps, taps.size() * sizeof(T)));
  checkCudaErrors(cudaMemcpy(_dev_taps, taps.data(), taps.size() * sizeof(T),
                             cudaMemcpyHostToDevice));
  occupancy_internal();

  // checkCudaErrors(cufftPlan1d(&_plan, _nchans, CUFFT_C2C, 10000000 / _nchans ));
};

template <>
cudaError_t channelizer<std::complex<float>>::launch(
    const std::complex<float> *in, std::complex<float> *out, int N,
    int grid_size, int block_size, cudaStream_t stream) {

  if (_ntaps <= 8) {
    // std::cout << " launch 1" << std::endl;
    _cupy_channelizer_8x8_complex64_complex64<<<grid_size, block_size, 0,
                                                stream>>>(
        _nchans, _ntaps, N, (const cuFloatComplex *)in,
        (const cuFloatComplex *)_dev_taps, (cuFloatComplex *)out);
  } else if (_ntaps <= 16) {
    // std::cout << " launch 2" << std::endl;
    _cupy_channelizer_16x16_complex64_complex64<<<grid_size, block_size, 0,
                                                  stream>>>(
        _nchans, _ntaps, N, (const cuFloatComplex *)in,
        (const cuFloatComplex *)_dev_taps, (cuFloatComplex *)out);
  } else {
    // std::cout << " launch 3" << std::endl;

    // threadsperblock = (32, 32)
    // blockspergrid = ((n_chans + 31) // 32, _get_numSM() * 2)

    // std::cout << grid_size << " " << block_size << std::endl;
    _cupy_channelizer_32x32_complex64_complex64<<<
        dim3((_nchans + 31) / 32, 40 * 2, 1), dim3(32, 32, 1), 0, stream>>>(
        _nchans, _ntaps, N, (const cuFloatComplex *)in,
        (const cuFloatComplex *)_dev_taps, (cuFloatComplex *)out);
  }

  checkCudaErrors(cudaPeekAtLastError());

  // std::cout << "fft with " << _nchans << " / " << N << std::endl;
  checkCudaErrors(cufftPlan1d(&_plan, _nchans, CUFFT_C2C, N ));
  checkCudaErrors(cufftSetStream(_plan, stream));
  checkCudaErrors (cufftExecC2C(_plan, (cufftComplex *) out, (cufftComplex *) out, CUFFT_FORWARD) );
  _conj_kernel.launch_default_occupancy({out},{out}, N*_nchans);
  cufftDestroy(_plan);

  return cudaPeekAtLastError();
} 

template <typename T>
cudaError_t channelizer<T>::launch(const std::vector<const void *> &inputs,
                                   const std::vector<void *> &outputs,
                                   size_t nitems) {
  return launch((const T *)inputs[0], (T *)outputs[0], nitems, _grid_size,
                _block_size, _stream);
}

template <typename T>
cudaError_t channelizer<T>::occupancy(int *minBlock, int *minGrid) {
  cudaError_t rc;

  if (_ntaps <= 8) {
    rc = cudaOccupancyMaxPotentialBlockSize(
        minGrid, minBlock, _cupy_channelizer_8x8_complex64_complex64, 0, 0);
  } else if (_ntaps <= 16) {
    rc = cudaOccupancyMaxPotentialBlockSize(
        minGrid, minBlock, _cupy_channelizer_16x16_complex64_complex64, 0, 0);
  } else {
    rc = cudaOccupancyMaxPotentialBlockSize(
        minGrid, minBlock, _cupy_channelizer_32x32_complex64_complex64, 0, 0);
  }

  return rc;
}

#define IMPLEMENT_KERNEL(T) template class channelizer<T>;

IMPLEMENT_KERNEL(std::complex<float>)

} // namespace cusp