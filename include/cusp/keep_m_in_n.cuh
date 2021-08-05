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
    
template <typename T>
class keep_m_in_n : public kernel
{
private:
    int _window;
    int _m;
public:
    keep_m_in_n(int m, int window) : _m(m), _window(window) {};
    cudaError_t launch(const T *in, T *out, int m, int window, int grid_size,
        int block_size, int N, cudaStream_t stream = 0);
    virtual cudaError_t launch(const std::vector<const void *>& inputs,
        const std::vector<void *>& outputs, size_t nitems) override;
    virtual cudaError_t occupancy(int *minBlock, int *minGrid);
};

}