/* -*- c++ -*- */
/*
 * Copyright 2021 Josh Morman
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cusp/fft.cuh>
#include <helper_cuda.h>

namespace cusp {

template <>
fft<std::complex<float>, true>::fft(size_t fft_size, size_t batch_size)
    : _fft_size(fft_size), _batch_size(batch_size) {
  // If a batch_size is specified, go ahead and create a plan
  if (_batch_size) {
    if (cufftPlan1d(&_plan, _fft_size, CUFFT_C2C, _batch_size) !=
        CUFFT_SUCCESS) {
      throw std::runtime_error("CUFFT error: plan creation failed");
    }
  }
};

template <>
fft<std::complex<float>, false>::fft(size_t fft_size, size_t batch_size)
    : _fft_size(fft_size), _batch_size(batch_size) {
  // If a batch_size is specified, go ahead and create a plan
  if (_batch_size) {
    if (cufftPlan1d(&_plan, _fft_size, CUFFT_C2C, _batch_size) !=
        CUFFT_SUCCESS) {
      throw std::runtime_error("CUFFT error: plan creation failed");
    }
  }
};

template <>
fft<float, true>::fft(size_t fft_size, size_t batch_size)
    : _fft_size(fft_size), _batch_size(batch_size) {
  // If a batch_size is specified, go ahead and create a plan
  if (_batch_size) {
    if (cufftPlan1d(&_plan, _fft_size, CUFFT_R2C, _batch_size) !=
        CUFFT_SUCCESS) {
      throw std::runtime_error("CUFFT error: plan creation failed");
    }
  }
};

template <>
fft<float, false>::fft(size_t fft_size, size_t batch_size)
    : _fft_size(fft_size), _batch_size(batch_size) {
  // If a batch_size is specified, go ahead and create a plan
  if (_batch_size) {
    if (cufftPlan1d(&_plan, _fft_size, CUFFT_R2C, _batch_size) !=
        CUFFT_SUCCESS) {
      throw std::runtime_error("CUFFT error: plan creation failed");
    }
  }
};

template <>
cudaError_t
fft<std::complex<float>, true>::launch(const std::complex<float> *in,
                                       std::complex<float> *out,
                                       size_t batch_size, cudaStream_t stream) {
  cufftHandle plan;
//   std::cout << "launching with batch_size = " << batch_size;
  if (!_batch_size) {
    if (_plan_cache.count(batch_size) > 0) {
      plan = _plan_cache[batch_size];
    //   std::cout << "plan found = " << batch_size;
    } else {
      checkCudaErrors(cufftPlan1d(&_plan, _fft_size, CUFFT_C2C, batch_size));
      checkCudaErrors(cufftSetStream(_plan, stream));
      _plan_cache[batch_size] = _plan;
      plan = _plan;
    }
  } else {
    plan = _plan;
  }

  if (auto err = cufftExecC2C(plan, (cufftComplex *)in, (cufftComplex *)out,
                              CUFFT_FORWARD) != CUFFT_SUCCESS) {
    throw std::runtime_error("CUFFT error: execution failed with error: " +
                             std::to_string(err));
  };
  return cudaSuccess;
}

template <>
cudaError_t fft<std::complex<float>, false>::launch(
    const std::complex<float> *in, std::complex<float> *out, size_t batch_size,
    cudaStream_t stream) {
  cufftHandle plan;
  if (!_batch_size) {
    if (_plan_cache.count(batch_size) > 0) {
      plan = _plan_cache[batch_size];
    } else {
      checkCudaErrors(cufftPlan1d(&_plan, _fft_size, CUFFT_C2C, batch_size));
      checkCudaErrors(cufftSetStream(_plan, stream));
      _plan_cache[batch_size] = _plan;
      plan = _plan;
    }
  } else {
    plan = _plan;
  }

  if (auto err = cufftExecC2C(plan, (cufftComplex *)in, (cufftComplex *)out,
                              CUFFT_INVERSE) != CUFFT_SUCCESS) {
    throw std::runtime_error("CUFFT error: execution failed with error: " +
                             std::to_string(err));
  };
  return cudaSuccess;
}

template <>
cudaError_t fft<float, true>::launch(const float *in, std::complex<float> *out,
                                     size_t batch_size, cudaStream_t stream) {
  cufftHandle plan;
  if (!_batch_size) {
    if (_plan_cache.count(batch_size) > 0) {
      plan = _plan_cache[batch_size];
    } else {
      checkCudaErrors(cufftPlan1d(&_plan, _fft_size, CUFFT_R2C, batch_size));
      checkCudaErrors(cufftSetStream(_plan, stream));
      _plan_cache[batch_size] = _plan;
      plan = _plan;
    }
  } else {
    plan = _plan;
  }

  if (auto err = cufftExecR2C(plan, (cufftReal *)in, (cufftComplex *)out) != CUFFT_SUCCESS) {
    throw std::runtime_error("CUFFT error: execution failed with error: " +
                             std::to_string(err));
  };
  return cudaSuccess;
}

template <>
cudaError_t fft<float, false>::launch(const std::complex<float> *in, float *out,
                                      size_t batch_size, cudaStream_t stream) {
  cufftHandle plan;
  if (!_batch_size) {
    if (_plan_cache.count(batch_size) > 0) {
      plan = _plan_cache[batch_size];
    } else {
      checkCudaErrors(cufftPlan1d(&_plan, _fft_size, CUFFT_R2C, batch_size));
      checkCudaErrors(cufftSetStream(_plan, stream));
      _plan_cache[batch_size] = _plan;
      plan = _plan;
    }
  } else {
    plan = _plan;
  }
  if (auto err = cufftExecC2R(plan, (cufftComplex *)in, (cufftReal *)out) != CUFFT_SUCCESS) {
    throw std::runtime_error("CUFFT error: execution failed with error: " +
                             std::to_string(err));
  };
  return cudaSuccess;
}

template <typename T, bool FORWARD>
cudaError_t fft<T, FORWARD>::launch(const std::vector<const void *> &inputs,
                                    const std::vector<void *> &outputs,
                                    size_t nitems) {
  if (_batch_size && nitems != _batch_size) {
    throw std::runtime_error(
        "Batch size must match the requested nitems to use generic launch method");
  } else {
    return launch((typename fft_inbuf<T,FORWARD>::type *)inputs[0], (typename fft_outbuf<T,FORWARD>::type *)outputs[0],
                  nitems, _stream);
  }
}

template <typename T, bool FORWARD>
cudaError_t fft<T, FORWARD>::occupancy(int *minBlock, int *minGrid) {
  throw std::runtime_error("Occupance not defined for cufft wrapped kernel");
}

#define IMPLEMENT_KERNEL(T, FORWARD) template class fft<T, FORWARD>;

IMPLEMENT_KERNEL(std::complex<float>, true)
IMPLEMENT_KERNEL(std::complex<float>, false)
IMPLEMENT_KERNEL(float, true)
IMPLEMENT_KERNEL(float, false)

} // namespace cusp