CC = icc

CFLAGS = -std=c99 -O2 -openmp -I$(MKLROOT)/include -offload-option,mic,compiler,"  -L$(MKLROOT)/lib/mic -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core" -offload-attribute-target=mic 
LDFLAGS =  -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lpthread -lm 

default: bench 

bench: mmio.o

.PHONY: clean

clean:
	rm -rf bench *.o
