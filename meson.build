# Copyright 2021 Josh Morman
#
# SPDX-License-Identifier: Apache-2.0

project('cusp', 'cpp', 
  version : '0.0.1',
  license : 'GPLv3',
  default_options : ['cpp_std=c++14'])

cmake = import('cmake')

cc = meson.get_compiler('cpp')
rt_dep = cc.find_library('rt', required : false)

cuda_available = add_languages('cuda', required : true)
cuda_dep = dependency('cuda', version : '>=10.1', required : true, modules : ['cufft'])
gtest_dep = dependency('gtest', main : true, version : '>=1.10', required : get_option('enable_testing'))

subdir('deps/cusignal')

subdir('include/cusp')
subdir('kernels')
subdir('test')

pkg = import('pkgconfig')
libs = []     # the library/libraries users need to link against
h = ['.'] # subdirectories of ${prefix}/${includedir} to add to header path
pkg.generate(libraries : libs,
             subdirs : h,
             version : meson.project_version(),
             name : 'libcusp',
             filebase : 'cusp',
             install_dir : get_option('prefix') / 'lib' / 'pkgconfig',
             description : 'CUDA Signal Processing Library')
