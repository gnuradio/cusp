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
    int _m;
    int _n;
    int _offset;
public:
    keep_m_in_n(int m, int n, int offset = 0) : _m(m), _n(n), _offset(offset) {};
    cudaError_t launch(const T *in, T *out, int m, int n, int cur_offset, int grid_size,
        int block_size, int N, cudaStream_t stream = 0);
    virtual cudaError_t launch(const std::vector<const void *>& inputs,
        const std::vector<void *>& outputs, size_t nitems) override;
    virtual cudaError_t occupancy(int *minBlock, int *minGrid);
};

}