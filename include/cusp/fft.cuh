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

#include <cufft.h>
#include <map>

namespace cusp {

template <class T, bool forward> struct fft_inbuf { typedef T type; };

template <> struct fft_inbuf<float, false> {
  typedef std::complex<float> type;
};

template <class T, bool forward> struct fft_outbuf { typedef T type; };

template <> struct fft_outbuf<float, true> {
  typedef std::complex<float> type;
};

template <typename T, bool forward> class fft : public kernel {
private:
  size_t _fft_size;
  size_t _batch_size;

  cufftHandle _plan; // if batch size is specified
  std::map<size_t, cufftHandle> _plan_cache;

public:
  fft(size_t fft_size, size_t batch_size = 0);
  cudaError_t launch(const typename fft_inbuf<T, forward>::type *in,
                     typename fft_outbuf<T, forward>::type *out, size_t batch_size,
                     cudaStream_t stream = 0);
  virtual cudaError_t launch(const std::vector<const void *> &inputs,
                             const std::vector<void *> &outputs,
                             size_t nitems) override;
  virtual cudaError_t occupancy(int *minBlock, int *minGrid);
};
} // namespace cusp