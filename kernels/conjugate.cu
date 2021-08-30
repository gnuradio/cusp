#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cusp/conjugate.cuh>
#include <thrust/complex.h>

namespace cusp {

__global__ void kernel_conjugate(const thrust::complex<float> *in, thrust::complex<float> *out, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    out[i] = thrust::complex<float>(in[i].real(), -1.0 * in[i].imag());
  }
}

cudaError_t conjugate::launch(const std::complex<float> *in, std::complex<float> *out, int N, int grid_size, int block_size,
                  cudaStream_t stream) {
  if (stream) {
    kernel_conjugate<<<grid_size, block_size, 0, stream>>>((const thrust::complex<float> *)in, 
                                                           (thrust::complex<float> *)out, N);
  } else {
    kernel_conjugate<<<grid_size, block_size>>>((const thrust::complex<float> *)in, 
                                                (thrust::complex<float> *)out, N);
  }
  return cudaPeekAtLastError();
}

cudaError_t conjugate::launch(const std::vector<const void *>& inputs,
                  const std::vector<void *>& outputs, size_t nitems) {
  return launch((const std::complex<float> *)inputs[0], (std::complex<float> *)outputs[0], nitems, _grid_size, _block_size, _stream);
}

cudaError_t conjugate::occupancy(int *minBlock, int *minGrid) {
  return cudaOccupancyMaxPotentialBlockSize(minGrid, minBlock,
                                            kernel_conjugate, 0, 0);
}


} // namespace cusp
