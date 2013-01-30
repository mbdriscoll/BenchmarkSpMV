NVCC = nvcc
CXX = nvcc
CC = nvcc

CXXFLAGS = -O3 -g -I/usr/local/cuda-5.0/include -m64 -I$(MKLROOT)/include
LDFLAGS = -L$(MKLROOT)/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -L/usr/local/cuda/lib64 -lcudart -lcusparse

default: driver

driver: driver.o mmio.o kernel.o

kernel.o: kernel.cu
	$(NVCC) -c $< -o $@

.PHONY: clean

clean:
	rm -rf driver *.o
