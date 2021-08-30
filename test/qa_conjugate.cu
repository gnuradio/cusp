
#include <gtest/gtest.h>
#include <complex>
#include <cusp/conjugate.cuh>
#include <cmath>

using namespace cusp;

template <typename T> 
void run_test(int N)
{
    std::vector<std::complex<float>> host_input_data(N);
    std::vector<std::complex<float>> expected_output_data(N);
    for (int i = 0; i < N; i++) {
      host_input_data[i] = std::complex<float>(float(i), float(i * 2));
      expected_output_data[i] = std::complex<float>(host_input_data[i].real(),
                                                    -1.0f * host_input_data[i].imag());
    }
    std::vector<std::complex<float>> host_output_data(N);
  
    void *dev_input_data;
    void *dev_output_data;
  
    cudaMalloc(&dev_input_data, N * sizeof(std::complex<float>));
    cudaMalloc(&dev_output_data, N * sizeof(std::complex<float>));
  
    cudaMemcpy(dev_input_data, host_input_data.data(),
               N * sizeof(std::complex<float>), cudaMemcpyHostToDevice);
  
    cusp::conjugate op;
    op.launch_default_occupancy({dev_input_data}, {dev_output_data}, N);
  
    cudaDeviceSynchronize();
    cudaMemcpy(host_output_data.data(), dev_output_data,
               N * sizeof(std::complex<float>), cudaMemcpyDeviceToHost);
  
    //EXPECT_EQ(expected_output_data, host_output_data);
    for (int i = 0; i < (int)expected_output_data.size(); i++) {
      EXPECT_NEAR(expected_output_data[i].real(),
                  host_output_data[i].real(),
                  expected_output_data[i].real() / 10000);
    }
}


TEST(ConjugateKernel, Basic) {
  int N = 1024 * 100;

  run_test<std::complex<float>>(N);
}