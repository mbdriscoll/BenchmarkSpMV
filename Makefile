ICC = icc
NVCC = nvcc

MIC_CXXFLAGS = -std=c99 -O2 -openmp -I$(MKLROOT)/include -mkl=parallel  -offload-option,mic,compiler,"" -offload-attribute-target=mic
MIC_LDFLAGS =

GPU_CXXFLAGS = -O3 -g -I/usr/local/cuda-5.0/include -m64 -I$(MKLROOT)/include
GPU_LDFLAGS = -L$(MKLROOT)/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -L/usr/local/cuda/lib64 -lcudart -lcusparse

default: usage

usage:
	# Make 'mic' or 'gpu' targets

mic: mic.cpp extra/mmio.o
	$(ICC) $(MIC_CXXFLAGS) $(MIC_LDFLAGS) $^ -o $@

gpu: gpu.cpp extra/mmio.o
	$(NVCC) $(GPU_CXXFLAGS) $(GPU_LDFLAGS) $^ -o $@

.PHONY: clean

clean:
	rm -rf mic gpu *.o extra/*.o
