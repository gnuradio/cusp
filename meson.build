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


subdir('include/cusp')
subdir('kernels')
subdir('test')

prefix = get_option('prefix')
# Escape spaces
prefix_parts = prefix.split(' ')
prefix = '\ '.join(prefix_parts)

conf = configuration_data()
conf.set('prefix', prefix)
conf.set('exec_prefix', '${prefix}')
conf.set('libdir', join_paths('${prefix}',get_option('libdir')))
conf.set('includedir', join_paths('${prefix}',get_option('includedir')))
conf.set('LIBVER', '0.0.0')

cmake_conf = configuration_data()
cmake_conf.set('libdir', join_paths(prefix,get_option('libdir')))
cmake.configure_package_config_file(
  name : 'cusp',
  input : 'cmake/Modules/cuspConfig.cmake.in',
  install_dir : get_option('prefix') / 'lib' / 'cmake' / 'cusp',
  configuration : cmake_conf
)

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