# cusp
Library of CUDA Kernels for Signal Processing

## Introduction
Access to signal processing and mathematical routines implemented in Python has been implemented in several solid libraries (cuPy, cuSignal), but utilizing this functionality from c++ applications such as GNU Radio which uses its own memory management has proved challenging.  The purpose of cusp (CUDA Signal Processing) is to provide a library of CUDA kernels that are wrapped in a manner that is easily callable from c++.  This library is analogous to VOLK but with CUDA hardware acceleration. 

## Dependencies

## Installation

Generally accomplished in the standard meson/ninja way:

```
meson setup build
cd build
ninja
ninja install
```

For Ubuntu 20.04 with CUDA installed via apt (CUDA v10.1), it is necessary to tell meson where to find CUDA

Also, for installing into a newsched prefix, set the `--prefix` and `--libdir`
```
CUDA_ROOT=/usr/ meson setup build --buildtype=debugoptimized --prefix=[PREFIX_DIR] -Denable_cuda=true --libdir=lib
```

## Usage

## Future Development

