/* -*- c++ -*- */
/*
 * Copyright 2021 Josh Morman
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#include <cmath>
#include <complex>
#include <cusp/channelizer.cuh>
#include <gtest/gtest.h>

#include <math.h>

#include <helper_cuda.h>
using namespace cusp;


#include <chrono>
#include <iostream>
#include <string>

template <typename T>
void run_test(const std::vector<T> &data, const std::vector<T> &taps, const size_t nchans) {

  void *dev_input_data;

  void *dev_output_data;
  auto N = data.size();

  // std::cout << "data = [";
  // for (int i=0; i<N; i++)
  // {
  //     std::cout << data[i].real() << "+" << data[i].imag() << "j" << ",";
  // }
  // std::cout << "];" << std::endl;

  checkCudaErrors(cudaMalloc(&dev_input_data, N * sizeof(T)));

  checkCudaErrors(cudaMemcpy(dev_input_data, data.data(),
                             N * sizeof(T), cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMalloc(&dev_output_data, N * sizeof(T)));


  cusp::channelizer<T> op(taps, nchans);

  std::vector<T> host_output_data(N);
  checkCudaErrors(cudaMalloc(&dev_output_data, N * sizeof(T)));

  auto t1 = std::chrono::steady_clock::now();

  int iters = 100;
  for (int i=0; i<iters; i++)
    checkCudaErrors(op.launch_default_occupancy({dev_input_data}, {dev_output_data}, N / nchans));

  cudaDeviceSynchronize();
  auto t2 = std::chrono::steady_clock::now();
  auto time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e9;
    std::cout << "[PROFILE_TIME]" << time << "[PROFILE_TIME]" << std::endl;
    std::cout << "[THROUGHPUT]" << (float)(N*iters) / time << "[THROUGHPUT]" << std::endl;

  checkCudaErrors(cudaMemcpy(host_output_data.data(), dev_output_data, sizeof(T)*N, cudaMemcpyDeviceToHost));


  // std::cout << "x = [";
  // for (int i=0; i<N; i++)
  // {
  //     std::cout << host_output_data[i].real() << "+" << host_output_data[i].imag() << "j" << ",";
  // }
  // std::cout << "];" << std::endl;

}

TEST(ChannelizerKernel, Basic) {
  std::vector<float> ftaps{
#include "data/pfb_taps.h"
  };

  std::vector<std::complex<float>> taps(ftaps.size());
  for (size_t i=0; i<ftaps.size(); i++)
  {
      taps[i] = std::complex<float>(ftaps[i],0.0);
  }


  std::vector<float> freqs{110., -513., 203., -230, 121};

  size_t nsamps = 1000000;
  // size_t nsamps = 1000;
  std::vector<std::complex<float>> in_data(nsamps);
  float samp_rate = 5000.0;
  float ifs = samp_rate * (float)freqs.size();
  for (size_t i = 0; i < nsamps; i++) {
    in_data[i] = {0,0};
    for (size_t j = 0; j < freqs.size(); j++) {
      float t = (float)i / ifs;
      float f = freqs[j] + j * samp_rate;
      in_data[i] += std::complex<float>(cos(2.0 * M_PI * f * t),
                                        sin(2.0 * M_PI * f * t));
    }
  }


  run_test<std::complex<float>>(in_data, taps, freqs.size());
}