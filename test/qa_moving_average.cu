/* -*- c++ -*- */
/*
 * Copyright 2021 Mark Bauer
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 *
 */

#include <gtest/gtest.h>
#include <complex>
#include <cusp/moving_average.cuh>
#include <cmath>

using namespace cusp;

template <typename T> 
void run_test(int N, int l, float s)
{
    std::vector<T> host_input_data(N);
    std::vector<T> expected_output_data(N);

    for (int i = 0; i < N; i++) {
        host_input_data[i] = (T)i;
        
        if (i >= l - 1) {
            for (int j = 0; j < l; j++) {
                expected_output_data[i] += host_input_data[i - j];
            }
        }
        expected_output_data[i] = (T)(expected_output_data[i] * s);
    }
    std::vector<T> host_output_data(N);
  
    void *dev_input_data;
    void *dev_output_data;
  
    cudaMalloc(&dev_input_data, N * sizeof(T));
    cudaMalloc(&dev_output_data, N * sizeof(T));
  
    cudaMemcpy(dev_input_data, host_input_data.data(),
               N * sizeof(T), cudaMemcpyHostToDevice);
  
    cusp::moving_average<T> op(l, s);
    op.launch_default_occupancy({dev_input_data}, {dev_output_data}, N);
  
    cudaDeviceSynchronize();
    cudaMemcpy(host_output_data.data(), dev_output_data,
               N * sizeof(T), cudaMemcpyDeviceToHost);
  
    EXPECT_EQ(expected_output_data, host_output_data);
}


TEST(MovingAverageKernel, Basic) {
    int N = 1024 * 100;
    int l = 50;
    int s = 2.0;

    run_test<int16_t>(N, l, s);
    run_test<int32_t>(N, l, s);
    run_test<float>(N, l, s);
}