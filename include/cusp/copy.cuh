/* -*- c++ -*- */
/*
 * Copyright 2021 Josh Morman
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#pragma once

#include <cusp/kernel.cuh>

namespace cusp
{

template <typename T>
class copy : public kernel
{
public:
    copy() = default;
    cudaError_t launch(const T *in, T *out, int grid_size, int block_size,
        int N, cudaStream_t stream = 0);
    virtual cudaError_t launch(const std::vector<const void *>& inputs,
        const std::vector<void *>& outputs, size_t nitems) override;
    virtual cudaError_t occupancy(int *minBlock, int *minGrid);
};

}