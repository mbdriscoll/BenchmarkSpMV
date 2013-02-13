CC = icc
CXX = icc

CXXFLAGS = -O2 -openmp -I$(MKLROOT)/include -offload-option,mic,compiler,"  -L$(MKLROOT)/lib/mic -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core" -offload-attribute-target=mic 
LDFLAGS =  -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lpthread -lm 

default: driver
driver: driver.cpp cpu.cpp mic.cpp mmio.c
	$(CXX) $^ -o $@ $(CXXFLAGS) $(LDFLAGS)

.PHONY: clean

clean:
	rm -rf driver *.o
