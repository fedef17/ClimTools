#!/bin/bash
source activate ctl3
rm ctool.*.so
rm ctp.*.so
f2py3 --fcompiler=gfortran --f90flags="-fopenmp" -lgomp -c -m ctp cluster_toolkit_parallel.f90 only: clus_sig_p clus_sig_p_ncl
cp ctp.*.so ../ctp.so
f2py3 --fcompiler=gfortran -c -m ctool cluster_toolkit.f90 only: clus_sig clus_opt adran1 gausts tsstat
cp ctool.*.so ../ctool.so
