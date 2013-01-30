#include <cstdio>
#include <cstdlib>

extern "C" {
#include "mmio.h"
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s MATRIX.mm\n", argv[0]);
        exit(1);
    }

    printf("Benchmarking.\n");

    printf("Done.\n");
    return 0;
}
