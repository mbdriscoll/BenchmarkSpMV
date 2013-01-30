default: driver

driver: kernel.o

%.o: %.cu
	nvcc -c $< -o $@
