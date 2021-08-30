/* -*- c++ -*- */
/*
 * Copyright 2021 Josh Morman
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#include <complex>
#include <cusp/deinterleave.cuh>
#include <gtest/gtest.h>
#include <helper_cuda.h>
using namespace cusp;

template <typename T> void run_test(int N, int nstreams, int blocksize) {
  size_t itemsize = sizeof(T);
  std::vector<T> host_input_data(N);
  std::vector<std::vector<T>> expected_output_data(nstreams);
  // std::vector<std::vector<T>> host_output_data(nstreams);
  for (int i = 0; i < nstreams; i++) {
    expected_output_data[i].resize(N / nstreams);
    // host_output_data.resize(N/nstreams);
  }
  for (int i = 0; i < N; i++) {
    host_input_data[i] = (T)i;
    expected_output_data[(i / blocksize) % nstreams]
                        [((i / blocksize) / nstreams) * blocksize +
                         i % blocksize] = (T)i;
  }


  cusp::deinterleave op(nstreams, blocksize, itemsize);
  // int minGrid, blockSize, gridSize;
  // op.occupancy(&blockSize, &minGrid);
  // gridSize = (N*itemsize + blockSize - 1) / blockSize;
  // op.set_block_and_grid(blockSize, gridSize);

  std::vector<void *> output_data_pointer_vec(nstreams);
  for (int i = 0; i < nstreams; i++) {
    void *tmp;
    checkCudaErrors(cudaMalloc(&tmp, N * sizeof(T)));
    output_data_pointer_vec[i] = tmp;
  }

  void *dev_input_data;
  cudaMalloc(&dev_input_data, N * sizeof(T));

  cudaMemcpy(dev_input_data, host_input_data.data(),
  N * sizeof(T), cudaMemcpyHostToDevice);

  // virtual cudaError_t launch(const std::vector<const void *> &inputs,
  //   const std::vector<void *> &outputs,
  //   size_t nitems) override;
  checkCudaErrors(op.launch_default_occupancy({dev_input_data}, output_data_pointer_vec, N));

  cudaDeviceSynchronize();
  for (int i=0; i<nstreams; i++)
  {
    std::vector<T> host_output_data(N/nstreams);
    cudaMemcpy(host_output_data.data(), output_data_pointer_vec[i], (N/nstreams) * sizeof(T),
                cudaMemcpyDeviceToHost);

    EXPECT_EQ(expected_output_data[i], host_output_data);
  }
}

TEST(deinterleave, Basic) {
  int N = 3000;

  run_test<int16_t>(3*(N / 3), 3, 2);
  run_test<float>(5*(N / 5), 5, 1);
}