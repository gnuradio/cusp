/* -*- c++ -*- */
/*
 * Copyright 2021 Josh Morman
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 */

#pragma once

#include <cusp/kernel.cuh>
#include <thrust/complex.h>
#include <cufft.h>
#include <cusp/conjugate.cuh>

namespace cusp
{
    template <typename T>
    class channelizer : public kernel
    {
    private:
        size_t _nchans;
        size_t _ntaps;
        std::vector<T> _taps;
        T *_dev_taps;
        cufftHandle _plan;

        cusp::conjugate _conj_kernel;
    public:
        channelizer(const std::vector<T>& taps, const size_t nchans);
        cudaError_t launch(const T *in, T *out, int grid_size, int block_size,
            int N, cudaStream_t stream = 0);
        virtual cudaError_t launch(const std::vector<const void *>& inputs,
            const std::vector<void *>& outputs, size_t nitems) override;
        virtual cudaError_t occupancy(int *minBlock, int *minGrid);
    };
}