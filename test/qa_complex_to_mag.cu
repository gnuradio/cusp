
#include <gtest/gtest.h>
#include <complex>
#include <cusp/complex_to_mag.cuh>
#include <cmath>

using namespace cusp;

void run_test(int N)
{
    std::vector<std::complex<float>> host_input_data(N);
    std::vector<float> expected_output_data(N);
    for (int i = 0; i < N; i++) {
      host_input_data[i] = std::complex<float>(float(i), float(i * 2));
      expected_output_data[i] = sqrtf(powf(host_input_data[i].real(), 2) + powf(host_input_data[i].imag(), 2));
    }
    std::vector<float> host_output_data(N);
  
    void *dev_input_data;
    void *dev_output_data;
  
    cudaMalloc(&dev_input_data, N * sizeof(std::complex<float>));
    cudaMalloc(&dev_output_data, N * sizeof(float));
  
    cudaMemcpy(dev_input_data, host_input_data.data(),
               N * sizeof(std::complex<float>), cudaMemcpyHostToDevice);
  
    cusp::complex_to_mag op;
    op.launch_default_occupancy({dev_input_data}, {dev_output_data}, N);
  
    cudaDeviceSynchronize();
    cudaMemcpy(host_output_data.data(), dev_output_data,
               N * sizeof(float), cudaMemcpyDeviceToHost);
  
    // EXPECT_EQ(expected_output_data, host_output_data);
    for (size_t i = 0; i < expected_output_data.size(); i++) {
      EXPECT_NEAR(expected_output_data[i],
                  host_output_data[i],
                  expected_output_data[i] / 10000);
    }
}


TEST(ComplexToMagKernel, Basic) {
  int N = 1024 * 100;

  run_test(N);
}