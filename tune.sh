#!/bin/bash

for val in $(seq 4 64); do
    g++ -O3 -g -I/usr/local/cuda-5.0/include -m64 -I/opt/intel/composerxe-2011.4.191/mkl/include -DHYBLEN=$val  -c -o driver.o driver.cpp
    make
    echo VALUE: $val
    ./driver mm/icos7.mm
done
