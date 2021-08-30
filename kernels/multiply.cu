#include <helper_cuda.h>
#include <complex>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cusp/multiply.cuh>

namespace cusp {

template <typename T>
__global__ void kernel_multiply(const T **ins, T *out, int ninputs, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    T *in = (T *)(*ins);
    out[i] = in[i];
    for (int j = 1; j < ninputs; j++) {
      in = (T *)(*(ins + j));
      out[i] *= in[i]; //(*(in + j))[i];
    }
  }
}

template <>
__global__ void
kernel_multiply<thrust::complex<float>>(const thrust::complex<float> **ins,
                                        thrust::complex<float> *out,
                                        int ninputs, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    thrust::complex<float> *in = (thrust::complex<float> *)(*ins);
    out[i] = in[i];
    for (int j = 1; j < ninputs; j++) {
      in = (thrust::complex<float> *)(*(ins + j));
      out[i] *= in[i]; //(*(in + j))[i];
    }
  }
}

template <typename T>
__global__ void kernel_multiply2(const T *in1, const T *in2, T *out,
                                 int ninputs, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    out[i] = in1[i] * in2[i];
  }
}

template <>
__global__ void kernel_multiply2<thrust::complex<float>>(
    const thrust::complex<float> *in1, const thrust::complex<float> *in2,
    thrust::complex<float> *out, int ninputs, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    out[i] = in1[i] * in2[i];
  }
}

template <typename T> multiply<T>::multiply(int ninputs) : _ninputs(ninputs) {
  checkCudaErrors(cudaMalloc(&_dev_ptr_array, sizeof(void *) * _ninputs));
}

template <typename T>
cudaError_t multiply<T>::launch(const std::vector<const void *> &inputs,
                                T *output, int ninputs, int grid_size,
                                int block_size, size_t nitems,
                                cudaStream_t stream) {

  if (ninputs == 2) {
    if (stream) {
      kernel_multiply2<<<grid_size, block_size, 0, stream>>>(
          (const T *)inputs[0], (const T *)inputs[1], (T *)output, ninputs,
          nitems);
    } else {
      kernel_multiply2<<<grid_size, block_size>>>((const T *)inputs[0],
                                                  (const T *)inputs[1],
                                                  (T *)output, ninputs, nitems);
    }
    return cudaPeekAtLastError();
  }

  // There is a better way to do this here - just getting the pointers into
  // device memory
  checkCudaErrors(cudaMemcpy(_dev_ptr_array, inputs.data(),
                             sizeof(void *) * ninputs, cudaMemcpyHostToDevice));

  if (stream) {
    kernel_multiply<<<grid_size, block_size, 0, stream>>>(
        (const T **)_dev_ptr_array, output, ninputs, nitems);
  } else {
    kernel_multiply<<<grid_size, block_size>>>((const T **)_dev_ptr_array,
                                               output, ninputs, nitems);
  }
  return cudaPeekAtLastError();
}

template <>
cudaError_t
multiply<std::complex<float>>::launch(const std::vector<const void *> &inputs,
                                      std::complex<float> *output, int ninputs,
                                      int grid_size, int block_size,
                                      size_t nitems, cudaStream_t stream) {

  if (ninputs == 2) {
    if (stream) {
      kernel_multiply2<<<grid_size, block_size, 0, stream>>>(
          (const thrust::complex<float> *)inputs[0],
          (const thrust::complex<float> *)inputs[1],
          (thrust::complex<float> *)output, ninputs, nitems);
    } else {
      kernel_multiply2<<<grid_size, block_size>>>(
          (const thrust::complex<float> *)inputs[0],
          (const thrust::complex<float> *)inputs[1],
          (thrust::complex<float> *)output, ninputs, nitems);
    }
    return cudaPeekAtLastError();
  }

  // There is a better way to do this here - just getting the pointers into
  // device memory
  checkCudaErrors(cudaMemcpy(_dev_ptr_array, inputs.data(),
                             sizeof(void *) * ninputs, cudaMemcpyHostToDevice));

  if (stream) {
    kernel_multiply<<<grid_size, block_size, 0, stream>>>(
        (const thrust::complex<float> **)_dev_ptr_array,
        (thrust::complex<float> *)output, ninputs, nitems);
  } else {
    kernel_multiply<<<grid_size, block_size>>>(
        (const thrust::complex<float> **)_dev_ptr_array,
        (thrust::complex<float> *)output, ninputs, nitems);
  }
  return cudaPeekAtLastError();
}

template <typename T>
cudaError_t multiply<T>::launch(const std::vector<const void *> &inputs,
                                const std::vector<void *> &outputs,
                                size_t nitems) {
  return launch(inputs, (T *)outputs[0], _ninputs, _grid_size, _block_size,
                nitems, _stream);
}

template <typename T>
cudaError_t multiply<T>::occupancy(int *minBlock, int *minGrid) {
  return cudaOccupancyMaxPotentialBlockSize(minGrid, minBlock,
                                            kernel_multiply<T>, 0, 0);
}

template <>
cudaError_t multiply<std::complex<float>>::occupancy(int *minBlock,
                                                     int *minGrid) {
  return cudaOccupancyMaxPotentialBlockSize(
      minGrid, minBlock, kernel_multiply<thrust::complex<float>>, 0, 0);
}

#define IMPLEMENT_KERNEL(T) template class multiply<T>;

IMPLEMENT_KERNEL(int8_t)
IMPLEMENT_KERNEL(int16_t)
IMPLEMENT_KERNEL(int32_t)
IMPLEMENT_KERNEL(int64_t)
IMPLEMENT_KERNEL(float)
IMPLEMENT_KERNEL(std::complex<float>)

} // namespace cusp