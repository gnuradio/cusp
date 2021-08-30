/* -*- c++ -*- */
/*
 * Copyright 2021 Mark Bauer
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#pragma once

#include <cusp/kernel.cuh>
#include <thrust/complex.h>

namespace cusp
{

class complex_to_mag_squared : public kernel
{
public:
    complex_to_mag_squared() = default;
    cudaError_t launch(const std::complex<float> *in, float *out, int grid_size, int block_size,
        int N, cudaStream_t stream = 0);
    virtual cudaError_t launch(const std::vector<const void *>& inputs,
        const std::vector<void *>& outputs, size_t nitems) override;
    virtual cudaError_t occupancy(int *minBlock, int *minGrid);
};

}