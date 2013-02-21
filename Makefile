CC = icc
CFLAGS = -std=c99 -O2 -openmp -I$(MKLROOT)/include -mkl=parallel  -offload-option,mic,compiler,"" -offload-attribute-target=mic

default: bench 

bench: mmio.o

.PHONY: clean

clean:
	rm -rf bench *.o
