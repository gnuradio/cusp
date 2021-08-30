# Copyright 2021 Josh Morman
#
# SPDX-License-Identifier: Apache-2.0

cusp_cu_files = files([
    'copy.cu',
    'add.cu',
    'moving_average.cu',
    'deinterleave.cu',
    'convolve.cu',
    'conjugate.cu',
    'complex_to_mag.cu',
    'complex_to_mag_squared.cu',
    'divide.cu',
    'multiply.cu',
    'keep_m_in_n.cu',
    'channelizer.cu' 
])

incdir = ['../include']

cusp = library('cusp', 
    cusp_cu_files, 
    include_directories : incdir, 
    install : true, 
    dependencies : [cuda_dep, cusignal_dep])

cusp_dep = declare_dependency(include_directories : incdir,
                    link_with : cusp,
                    dependencies : cuda_dep)
