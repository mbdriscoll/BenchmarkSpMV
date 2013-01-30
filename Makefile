default: driver

driver: mmio.o

%.o: %.cu
	nvcc -c $< -o $@
