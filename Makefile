CC = icc
CXX = icc

CXXFLAGS = -O3 -g -openmp -I. -I$(MKLROOT)/include -mkl=parallel  -offload-option,mic,compiler,"" -offload-attribute-target=mic
LDFLAGS = -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lpthread

default: driver

driver: driver.o mic.o mmio.o

kernel.o: kernel.cu
	$(NVCC) -c $< -o $@

.PHONY: clean

clean:
	rm -rf driver *.o
