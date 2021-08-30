#include <cuda.h>
#include <complex>
#include <thrust/complex.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <cusp/complex_to_mag_squared.cuh>

namespace cusp {

__global__ void kernel_mag_squared(const thrust::complex<float> *in, float *out, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    out[i] = in[i].real() * in[i].real() + in[i].imag() * in[i].imag();
  }
}

cudaError_t complex_to_mag_squared::launch(const std::complex<float> *in, float *out, int N, int grid_size, int block_size,
                  cudaStream_t stream) {
  if (stream) {
    kernel_mag_squared<<<grid_size, block_size, 0, stream>>>((const thrust::complex<float> *)in, 
                                                     out, N);
  } else {
    kernel_mag_squared<<<grid_size, block_size>>>((const thrust::complex<float> *)in, 
                                          out, N);
  }
  return cudaPeekAtLastError();
}

cudaError_t complex_to_mag_squared::launch(const std::vector<const void *>& inputs,
                  const std::vector<void *>& outputs, size_t nitems) {
  return launch((const std::complex<float>*)inputs[0], (float*)outputs[0], nitems, _grid_size, _block_size, _stream);
}

cudaError_t complex_to_mag_squared::occupancy(int *minBlock, int *minGrid) {
  return cudaOccupancyMaxPotentialBlockSize(minGrid, minBlock,
                                            kernel_mag_squared,
                                            0, 0);
}

} // namespace cusp
