/* -*- c++ -*- */
/*
 * Copyright 2021 Josh Morman
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#include <cmath>
#include <complex>
#include <cusp/convolve.cuh>
#include <gtest/gtest.h>

#include <helper_cuda.h>
using namespace cusp;

template <typename T>
void run_test(int N, const std::vector<T> &taps, convolve_mode_t mode) {
  std::vector<T> host_input_data(N);
  std::vector<T> expected_output_data(N);
  for (int i = 0; i < N; i++) {
    host_input_data[i] = T(i);
  }

  void *dev_input_data;

  void *dev_output_data;

  checkCudaErrors(cudaMalloc(&dev_input_data, N * sizeof(T)));

  checkCudaErrors(cudaMemcpy(dev_input_data, host_input_data.data(),
                             N * sizeof(T), cudaMemcpyHostToDevice));

  cusp::convolve<T, T> op(taps, mode);

  int N_out = op.output_length(N);
  std::vector<T> host_output_data(N_out);
  checkCudaErrors(cudaMalloc(&dev_output_data, N_out * sizeof(T)));

  checkCudaErrors(op.launch_default_occupancy({dev_input_data}, {dev_output_data}, N));

  cudaDeviceSynchronize();

  checkCudaErrors(cudaMemcpy(host_output_data.data(), dev_output_data,
                             N_out * sizeof(T), cudaMemcpyDeviceToHost));

  // EXPECT_EQ(expected_output_data, host_output_data);

  for (auto &x : host_output_data) {
    std::cout << x << ' ';
  }
  std::cout << std::endl;
}

TEST(ConvolveKernel, Basic) {
  int N = 20 * 1;

  std::vector<float> ftaps{1, 1, 1};
  run_test<float>(N, ftaps, convolve_mode_t::FULL);
  run_test<float>(N, ftaps, convolve_mode_t::VALID);
  run_test<float>(N, ftaps, convolve_mode_t::SAME);

  std::vector<std::complex<float>> ctaps{std::complex<float>(1, 2),
                                      std::complex<float>(3, 4),
                                      std::complex<float>(5, 6)};
  run_test<std::complex<float>>(N, ctaps, convolve_mode_t::FULL);
  run_test<std::complex<float>>(N, ctaps, convolve_mode_t::VALID);
  run_test<std::complex<float>>(N, ctaps, convolve_mode_t::SAME);
}