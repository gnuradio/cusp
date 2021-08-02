/* -*- c++ -*- */
/*
 * Copyright 2021 Josh Morman
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 *
 */

#include <complex>
#include <cusp/copy.cuh>
#include <cusp/cusp.cuh>
#include <gtest/gtest.h>

template <typename T> void run_test(int N)
{
    std::vector<std::complex<float>> host_input_data(N);
    for (int i = 0; i < N; i++) {
      host_input_data[i] = std::complex<float>(i, -i);
    }
    std::vector<std::complex<float>> host_output_data(N);
  
    void *dev_input_data;
    void *dev_output_data;
  
    cudaMalloc(&dev_input_data, N * sizeof(std::complex<float>));
    cudaMalloc(&dev_output_data, N * sizeof(std::complex<float>));
  
    cudaMemcpy(dev_input_data, host_input_data.data(),
               N * sizeof(std::complex<float>), cudaMemcpyHostToDevice);
  
    int ncopies = N * sizeof(std::complex<float>) / sizeof(T);
    cusp::copy<T> op;
    
    op.launch_default_occupancy({dev_input_data}, {dev_output_data}, ncopies);
  
    cudaDeviceSynchronize();
    cudaMemcpy(host_output_data.data(), dev_output_data,
               N * sizeof(std::complex<float>), cudaMemcpyDeviceToHost);
  
    EXPECT_EQ(host_input_data, host_output_data);
}

TEST(CopyKernel, Basic) {
  int N = 1024 * 1000;

  run_test<uint64_t>(N);
  run_test<uint8_t>(N);
}
