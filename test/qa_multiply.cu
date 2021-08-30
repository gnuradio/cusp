
#include <gtest/gtest.h>
#include <complex>
#include <cusp/multiply.cuh>

using namespace cusp;

template <typename T> 
void run_test(int N, int num_inputs)
{
    std::vector<T> host_input_data(N);
    std::vector<T> expected_output_data(N);

    for (int i = 0; i < N; i++) {
      host_input_data[i] = (T)(i + 1);
      T out = host_input_data[i];
      for (int j = 0; j < num_inputs - 1; j++) {
        out *= host_input_data[i];
      }
      expected_output_data[i] = out;
    }
    std::vector<T> host_output_data(N);
  
    void *dev_input_data;
    void **dev_output_data;
  
    cudaMalloc(&dev_input_data, N * sizeof(T));
    cudaMalloc(&dev_output_data, N * sizeof(T));

    cudaMemcpy(dev_input_data, host_input_data.data(),
               N * sizeof(T), cudaMemcpyHostToDevice);
  
    cusp::multiply<T> op(num_inputs);

    std::vector<const void *> input_data_pointer_vec(num_inputs);
    for (int i=0; i<num_inputs; i++)
    {
      input_data_pointer_vec[i] = dev_input_data;
    }

    op.launch_default_occupancy({input_data_pointer_vec}, {dev_output_data}, N);
  
    cudaDeviceSynchronize();
    cudaMemcpy(host_output_data.data(), dev_output_data,
               N * sizeof(T), cudaMemcpyDeviceToHost);
  
    EXPECT_EQ(expected_output_data, host_output_data);
}

template <> 
void run_test<std::complex<float>>(int N, int num_inputs)
{
    std::vector<std::complex<float>> host_input_data(N);
    std::vector<std::complex<float>> expected_output_data(N);

    for (int i = 0; i < N; i++) {
      host_input_data[i] = std::complex<float>(float(i + 1), float(i + 1));
      std::complex<float> out = host_input_data[i];
      for (int j = 0; j < num_inputs - 1; j++) {
        out *= host_input_data[i];
      }
      expected_output_data[i] = out;
    }
    std::vector<std::complex<float>> host_output_data(N);
  
    void *dev_input_data;
    void **dev_output_data;
  
    cudaMalloc(&dev_input_data, N * sizeof(std::complex<float>));
    cudaMalloc(&dev_output_data, N * sizeof(std::complex<float>));

    cudaMemcpy(dev_input_data, host_input_data.data(),
               N * sizeof(std::complex<float>), cudaMemcpyHostToDevice);
  
    cusp::multiply<std::complex<float>> op(num_inputs);
    std::vector<const void *> input_data_pointer_vec(num_inputs);
    for (int i=0; i<num_inputs; i++)
    {
      input_data_pointer_vec[i] = dev_input_data;
    }

    op.launch_default_occupancy({input_data_pointer_vec}, {dev_output_data}, N);
  
    cudaDeviceSynchronize();
    cudaMemcpy(host_output_data.data(), dev_output_data,
               N * sizeof(std::complex<float>), cudaMemcpyDeviceToHost);
  
    //EXPECT_EQ(expected_output_data, host_output_data);

    for (int i = 0; i < (int)expected_output_data.size(); i++) {

      // Also add a test case to check for imaginary component

      EXPECT_NEAR(expected_output_data[i].real(),
                  host_output_data[i].real(),
                  abs(expected_output_data[i].real() / 10000));

      EXPECT_NEAR(expected_output_data[i].imag(),
                  host_output_data[i].imag(),
                  abs(expected_output_data[i].imag() / 10000));
    }
}

TEST(MultiplyKernel, Basic) {
  int N = 1024 * 100;

  run_test<int16_t>(N, 3);
  run_test<float>(N, 3);
  run_test<std::complex<float>>(N, 3);
}