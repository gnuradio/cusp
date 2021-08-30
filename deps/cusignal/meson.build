# comp = find_program('custom_compiler')

# infile = 'source_code.txt'
# outfile = 'output.bin'

# mytarget = custom_target('targetname',
#   output : outfile,
#   input : infile,
#   command : [comp, '@INPUT@', '@OUTPUT@'],
#   install : true,
#   install_dir : 'subdir')

cusignal_cu_files = files([
    'cpp/src/convolution/_convolution.cu',
    'cpp/src/filtering/_channelizer.cu'
])

cusignal = library('cusignal', 
    cusignal_cu_files, 
    include_directories : [], 
    install : true, 
    dependencies : [cuda_dep])

cusignal_dep = declare_dependency(include_directories : [],
                    link_with : cusignal,
                    dependencies : cuda_dep)