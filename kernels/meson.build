# Copyright 2021 Josh Morman
#
# SPDX-License-Identifier: Apache-2.0

cusp_cu_files = files([
    'copy.cu', 
])

incdir = ['../include']

cusp = library('cusp', 
    cusp_cu_files, 
    include_directories : incdir, 
    install : true, 
    dependencies : [cuda_dep])

cusp_dep = declare_dependency(include_directories : incdir,
                    link_with : cusp,
                    dependencies : cuda_dep)