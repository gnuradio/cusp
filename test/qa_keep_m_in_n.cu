/* -*- c++ -*- */
/*
 * Copyright 2021 Mark Bauer
 *
 * SPDX-License-Identifier: LGPL-3.0-or-later
 *
 */

#include <cmath>
#include <complex>
#include <cusp/keep_m_in_n.cuh>
#include <gtest/gtest.h>

using namespace cusp;

template <typename T> void run_test(int N, int m, int n) {
  int M = m * ((N + n - 1) / n);
  std::vector<T> host_input_data(N);
  std::vector<T> expected_output_data(M);

  for (int i = 0; i < N; i++) {
    host_input_data[i] = (T)i;

    if (i % n == 0) {
      int window_number = 0;

      if (i > n - 1) {
        window_number = i / n;
      }
      for (int j = 0; j < m; j++) {
        if (i + j < N) {
          expected_output_data[j + window_number * m] = T(i + j);
        }
      }
    }
  }
  std::vector<T> host_output_data(M);

  void *dev_input_data;
  void *dev_output_data;

  cudaMalloc(&dev_input_data, N * sizeof(T));
  cudaMalloc(&dev_output_data, N * sizeof(T));

  cudaMemcpy(dev_input_data, host_input_data.data(), N * sizeof(T),
             cudaMemcpyHostToDevice);

  cusp::keep_m_in_n<uint8_t> op(m, n, sizeof(T));
  op.launch_default_occupancy({dev_input_data}, {dev_output_data}, M*sizeof(T));

  cudaDeviceSynchronize();
  cudaMemcpy(host_output_data.data(), dev_output_data, M * sizeof(T),
             cudaMemcpyDeviceToHost);

  EXPECT_EQ(expected_output_data, host_output_data);
}

template <> void run_test<std::complex<float>>(int N, int m, int n) {
  int M = m * ((N + n - 1) / n);
  std::vector<std::complex<float>> host_input_data(N);
  std::vector<std::complex<float>> expected_output_data(M);
  for (int i = 0; i < N; i++) {
    host_input_data[i] = std::complex<float>(float(i), float(i));

    if (i % n == 0) {
      int window_number = 0;

      if (i > n - 1) {
        window_number = i / n;
      }
      for (int j = 0; j < m; j++) {
        if (i + j < N) {
          expected_output_data[j + window_number * m] =
              std::complex<float>(float(i + j), float(i + j));
        }
      }
    }
  }
  std::vector<std::complex<float>> host_output_data(M);

  void *dev_input_data;
  void *dev_output_data;

  cudaMalloc(&dev_input_data, N * sizeof(std::complex<float>));
  cudaMalloc(&dev_output_data, N * sizeof(std::complex<float>));

  cudaMemcpy(dev_input_data, host_input_data.data(),
             N * sizeof(std::complex<float>), cudaMemcpyHostToDevice);

  cusp::keep_m_in_n<uint8_t> op(m, n, sizeof(std::complex<float>));
  op.launch_default_occupancy({dev_input_data}, {dev_output_data}, M*sizeof(std::complex<float>));

  cudaDeviceSynchronize();
  cudaMemcpy(host_output_data.data(), dev_output_data,
             M * sizeof(std::complex<float>), cudaMemcpyDeviceToHost);

  EXPECT_EQ(expected_output_data, host_output_data);
}

TEST(KeepMInNKernel, Basic) {
  int N = 1024 * 100;

  run_test<int32_t>(N, 7, 13);
  run_test<float>(N, 8, 16);
  run_test<std::complex<float>>(N, 10, 20);
}