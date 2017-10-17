set -e

make clean
make -j40
make py

#export LD_LIBRARY_PATH=/opt/intel/compilers_and_libraries_2017.4.196/linux/mkl/lib/intel64:$LD_LIBRARY_PAT
