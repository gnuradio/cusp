# Copyright 2021 Josh Morman
#
# SPDX-License-Identifier: Apache-2.0

incdir = ['../include']

qa_srcs = ['qa_copy',
           'qa_moving_average',
           'qa_keep_m_in_n',
           'qa_add'
          ]

foreach qa : qa_srcs
    e = executable(qa, 
        qa + '.cu', 
        include_directories : incdir, 
        link_language : 'cpp',
        dependencies: [cusp_dep, gtest_dep], 
        install : false)
    test('Test ' + qa, e)
endforeach
