CC = icc
CXX = icc

CXXFLAGS = -openmp -I$(MKLROOT)/include -offload-option,mic,compiler,"  -L$(MKLROOT)/lib/mic -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core" -offload-attribute-target=mic 
LDFLAGS =  -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lpthread -lm 

default: driver

driver: driver.o mic.o mmio.o

kernel.o: kernel.cu
	$(NVCC) -c $< -o $@

.PHONY: clean

clean:
	rm -rf driver *.o
