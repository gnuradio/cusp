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

namespace cusp
{
    enum class convolve_mode_t 
    {
        VALID = 0,
        SAME = 1,
        FULL = 2,
        FULL_TRUNC = 3
    };

    template <typename T, typename T_TAPS>
    class convolve : public kernel
    {
    private:
        std::vector<T_TAPS> _taps;
        T *_dev_taps;
        convolve_mode_t _mode;
    public:
        convolve(const std::vector<T_TAPS>& taps, const convolve_mode_t mode = 2);
        cudaError_t launch(const T *in, T *out, int grid_size, int block_size,
            int N, cudaStream_t stream = 0);
        virtual cudaError_t launch(const std::vector<const void *>& inputs,
            const std::vector<void *>& outputs, size_t nitems) override;
        virtual cudaError_t occupancy(int *minBlock, int *minGrid);
        int output_length(int input_length);
    };
}